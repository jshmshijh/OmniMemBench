import base64
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark' / 'longctx'))
import run_longctx as runner
import utils
from omnimembench.engine.evaluate.scoring import qa_score, compute_nugget_metrics, retrieval_judgment


class JudgmentValidityTests(unittest.TestCase):
    def test_qa_scores_follow_the_discrete_rubric(self):
        for value in [0, 0.5, 1]:
            self.assertEqual(qa_score({'score': value}), value)
        for value in [-1, 0.2, 2, True, '0.5', None]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                qa_score({'score': value})
        rows = [{'status': 'ok', 'category': 'SH', 'judge': {'result': {'score': value}}}
                for value in [0, 1, 0.2, 2]]
        summary = utils.summarize_rows(rows)
        self.assertEqual(summary['metrics']['overall_score'], 0.5)
        self.assertEqual(summary['counts']['valid_judged'], 2)
        self.assertEqual(summary['counts']['unjudged_or_invalid'], 2)

    def test_sum_requires_all_reference_labels(self):
        nuggets = [{'nugget_id': 'n1'}, {'nugget_id': 'n2', 'source_type': 'multimodal'}]
        result = compute_nugget_metrics({'n1': 0, 'n2': 1}, nuggets)
        self.assertEqual(result['nugget_recall'], 0.5)
        self.assertEqual(result['mm_nugget_recall'], 1)
        self.assertEqual(compute_nugget_metrics({'n1': 0, 'n2': 0}, nuggets)['nugget_recall'], 0)
        for coverage in [{}, {'n1': 1}, {'n1': 1, 'n2': 0, 'extra': 1},
                         {'n1': 0.5, 'n2': 0}, {'n1': 'false', 'n2': 0}, []]:
            with self.subTest(coverage=coverage), self.assertRaises(ValueError):
                compute_nugget_metrics(coverage, nuggets)

    def test_retrieval_requires_one_binary_label_per_item(self):
        result = retrieval_judgment({'clue_coverage': [0, True], 'entry_relevance': [False]}, 2, 1)
        self.assertEqual(result['clue_coverage'], [False, True])
        for result in [
            {'clue_coverage': [1], 'entry_relevance': [0]},
            {'clue_coverage': [1, 0], 'entry_relevance': []},
            {'clue_coverage': ['false', 1], 'entry_relevance': [0]},
            {'clue_coverage': [1, 0], 'entry_relevance': [0.5]},
        ]:
            with self.subTest(result=result), self.assertRaises(ValueError):
                retrieval_judgment(result, 2, 1)

    def judge(self, category, responses):
        nuggets = [{'nugget_id': f'n{i}', 'content': f'Fact {i}'} for i in range(6)]
        row = {'status': 'ok', 'qa_index': 0, 'category': category, 'question': 'Question',
               'original_answer': 'Reference', 'system_answer': 'Saved answer', 'nuggets': nuggets}
        with patch.object(runner, 'llm_chat', side_effect=[(r, {}) for r in responses]) as chat, \
             patch.object(runner.time, 'sleep'):
            result = runner._run_judge_for_item(
                row, qa={'nuggets': nuggets}, api_key='test', base_url='https://judge.example/v1',
                judge_model='glm-5.2', judge_template='{{ground_truth}}',
                nugget_judge_template='{{nuggets_list}}', timeout_seconds=1, max_retries=len(responses))
        return result, chat.call_count

    def test_invalid_qa_and_sum_outputs_are_retried(self):
        for category, valid in [('SH', '{"score": 0}'),
                                ('SUM', json.dumps({f'n{i}': 0 for i in range(6)}))]:
            with self.subTest(category=category):
                row, calls = self.judge(category, ['{}', valid])
                self.assertEqual(calls, 2)
                summary = utils.summarize_rows([row])
                self.assertEqual(summary['counts']['valid_judged'], 1)
                self.assertEqual(summary['metrics']['overall_score'], 0)

    def test_exhausted_judge_preserves_answers_and_excludes_scores(self):
        for category in ['SH', 'SUM']:
            with self.subTest(category=category):
                row, calls = self.judge(category, ['not JSON', '{}'])
                self.assertEqual(calls, 2)
                self.assertEqual(row['status'], 'ok')
                self.assertEqual(row['system_answer'], 'Saved answer')
                self.assertIn('error', row['judge']['result'])
                summary = utils.summarize_rows([row])
                self.assertEqual(summary['counts']['valid_judged'], 0)
                self.assertIsNone(summary['metrics']['overall_score'])

    def test_resume_rejudges_failures_and_updates_metadata_without_new_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            data = folder / 'dataset.json'
            data.write_text(json.dumps({'QAs': [{'type': 'SH', 'question': 'Where?', 'answer': 'Paris'}]}))
            args = dict(data_path=data, run_output_dir=folder, run_id='00001', tier='128k', eval_type='omni',
                        api_key='test', base_url='https://answer.example/v1', model='answer-model',
                        judge_model='glm-5.2', sys_prompt='', judge_template='{{ground_truth}}',
                        nugget_judge_template='{{nuggets_list}}', temperature=0, max_tokens=20,
                        timeout_seconds=1, max_retries=2, max_workers=1, enable_judge=True,
                        resume=True, include_persona=False, heartbeat_seconds=0, print_every=0)
            with patch.object(runner, 'llm_chat', side_effect=[('Paris', {}), ('{}', {}), ('{}', {})]) as chat, \
                 patch.object(runner.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()):
                runner.eval_single_run(**args)
            self.assertEqual(chat.call_count, 3)
            self.assertEqual(json.loads((folder / 'summary.json').read_text())['summary']['counts']['valid_judged'], 0)
            (folder / 'meta.json').write_text(json.dumps({'judge_model': 'old-judge', 'model': 'answer-model'}))
            with patch.object(runner, 'llm_chat', return_value=('{"score": 1}', {})) as chat, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.eval_single_run(**args, judge_only=True)
                runner.eval_single_run(**args)
            self.assertEqual(chat.call_count, 1)
            self.assertEqual(utils.load_jsonl_rows(folder / 'results.jsonl')[0]['system_answer'], 'Paris')
            for name in ['meta.json', 'summary.json']:
                saved = json.loads((folder / name).read_text())
                self.assertEqual((saved.get('meta') or saved)['judge_model'], 'glm-5.2')
            self.assertEqual(saved['summary']['counts']['valid_judged'], 1)


class LocalMediaTests(unittest.TestCase):
    def test_local_image_audio_and_video_reach_the_api_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = []
            for modality, suffix in [('image', 'png'), ('audio', 'flac'), ('video', 'mp4')]:
                name = f'{modality}.{suffix}'
                (root / name).write_bytes(modality.encode())
                media.append({'type': modality, 'url': name, 'caption': f'{modality} caption'})
            dataset = {'multi_session_dialogues': [{'session_id': 'S1', 'dialogues': [
                {'user': '<image> <audio> <video>', 'assistant': 'Reply', 'mm': media}]}]}
            enabled = {'image': 'raw', 'audio': 'raw', 'video': 'raw'}
            contents = utils.build_contents(dataset, enabled, '', False, root)
            inline = [p for c in contents for p in c['parts'] if 'inlineData' in p]
            self.assertEqual(len(inline), 3)
            converted = [utils._gemini_part_to_openai(p) for p in inline]
            self.assertEqual(converted[0]['image_url']['url'], 'data:image/png;base64,' + base64.b64encode(b'image').decode())
            self.assertEqual(converted[1]['input_audio'], {'data': base64.b64encode(b'audio').decode(), 'format': 'flac'})
            self.assertEqual(converted[2]['video_url']['url'], 'data:video/mp4;base64,' + base64.b64encode(b'video').decode())
            (root / 'audio.flac').unlink()
            vl = utils.build_contents(dataset, dict(enabled, audio='caption'), '', False, root)
            self.assertEqual(sum('inlineData' in p for c in vl for p in c['parts']), 2)
            self.assertTrue(any('audio caption' in p.get('text', '') for c in vl for p in c['parts']))

    def test_local_question_image_is_included(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'query.png'
            image_path.write_bytes(b'image')
            media = utils.extract_qa_media({'question_image': str(image_path)})
            parts = utils.build_raw_parts(media, {'image': 'raw'})
            self.assertEqual(len(parts), 1)
            self.assertEqual(base64.b64decode(parts[0]['inlineData']['data']), b'image')

    def test_missing_local_question_image_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            media = utils.extract_qa_media({'question_image': str(Path(directory) / 'missing.png')})
            with self.assertRaises(FileNotFoundError):
                utils.build_raw_parts(media, {'image': 'raw'})

    def test_http_media_urls_are_preserved(self):
        url = 'https://media.example/image.png'
        part = utils._filedata_part(url, 'image/png')
        self.assertEqual(part, {'fileData': {'mimeType': 'image/png', 'fileUri': url}})
        self.assertEqual(utils._gemini_part_to_openai(part)['image_url']['url'], url)


if __name__ == '__main__':
    unittest.main()
