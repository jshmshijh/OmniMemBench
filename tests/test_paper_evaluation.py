import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmark' / 'longctx'))
import run_longctx as runner
import utils


class PaperEvaluationTests(unittest.TestCase):
    def process(self, **overrides):
        arguments = dict(
            qa_index=0, qa={'type': 'SH', 'question': 'Where?', 'answer': 'Paris', 'full_answer': 'Paris, France'},
            base_contents=[], input_form={'image': 'raw', 'audio': 'raw', 'video': 'raw'}, run_id='00001',
            api_key='answer-test-key', base_url='https://answer.example/v1', model='answer-model',
            judge_model=None, enable_judge=True,
            judge_template='{{question}} | {{ground_truth}} | {{model_output}}', nugget_judge_template='{{nuggets_list}}',
            temperature=0.7, max_tokens=100, timeout_seconds=10, max_retries=1,
            judge_api_key='judge-test-key', judge_base_url='https://judge.example/v1',
        )
        arguments.update(overrides)
        with patch.object(runner, 'llm_chat', side_effect=[('Paris', {}), ('{"score": 1}', {})]) as chat:
            result = runner.process_one(**arguments)
        return result, chat.call_args_list

    def test_answer_and_judge_use_separate_models_and_endpoints(self):
        result, calls = self.process()
        self.assertEqual(calls[0].kwargs['model'], 'answer-model')
        self.assertEqual(calls[0].kwargs['api_key'], 'answer-test-key')
        self.assertEqual(calls[1].kwargs['model'], 'glm-5.2')
        self.assertEqual(calls[1].kwargs['api_key'], 'judge-test-key')
        self.assertEqual(calls[1].kwargs['base_url'], 'https://judge.example/v1')
        self.assertEqual(calls[1].kwargs['temperature'], 0)
        self.assertEqual(result['judge']['model'], 'glm-5.2')

    def test_provider_model_id_override(self):
        result, calls = self.process(judge_model='provider/glm-5.2')
        self.assertEqual(calls[1].kwargs['model'], 'provider/glm-5.2')
        self.assertEqual(result['judge']['model'], 'provider/glm-5.2')

    def test_omni_and_vl_use_the_same_reference(self):
        prompts = []
        for eval_type in ('omni', 'vl'):
            _, calls = self.process(eval_type=eval_type)
            prompts.append(calls[1].kwargs['contents'][0]['parts'][0]['text'])
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(prompts[0], 'Where? | Paris | Paris')

    def test_overall_weights_questions_including_sum(self):
        rows = [
            {'status': 'ok', 'category': 'SH', 'judge': {'result': {'score': 1}}},
            {'status': 'ok', 'category': 'SUM', 'nuggets': [{'nugget_id': f'n{i}'} for i in range(8)],
             'judge': {'result': {'nugget_coverage': {f'n{i}': int(i < 2) for i in range(8)}}}},
            {'status': 'ok', 'category': 'MH', 'judge': {'result': {'error': 'invalid output'}}},
            {'status': 'error', 'category': 'SH', 'judge': {'result': {'score': 0}}},
        ]
        summary = utils.summarize_rows(rows)
        self.assertEqual(summary['metrics']['overall_score'], 0.625)
        self.assertEqual(summary['counts']['valid_judged'], 2)

    def test_judge_only_replaces_other_model_and_resumes_same_model(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.jsonl'
            row = {'status': 'ok', 'qa_index': 0, 'question': 'Where?', 'system_answer': 'Paris',
                   'original_answer': 'Paris', 'category': 'SH', 'eval_type': 'omni',
                   'judge': {'model': 'previous-judge', 'result': {'score': 0}}}
            output.write_text(json.dumps(row) + '\n')
            arguments = dict(jsonl_path=output, qas=[{'answer': 'Paris'}], run_id='00001',
                             api_key='judge-test-key', base_url='https://judge.example/v1', judge_model='glm-5.2',
                             judge_template='{{ground_truth}}', nugget_judge_template='',
                             timeout_seconds=10, max_retries=1, max_workers=1, print_every=0)
            with patch.object(runner, 'llm_chat', return_value=('{"score": 1}', {})) as chat:
                runner.backfill_judge(**arguments)
                runner.backfill_judge(**arguments)
            self.assertEqual(chat.call_count, 1)
            saved = utils.load_jsonl_rows(output)[0]
            self.assertEqual(saved['judge']['model'], 'glm-5.2')
            self.assertEqual(saved['judge']['result']['score'], 1)

    def test_judge_only_accepts_judge_credentials_without_answer_key(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'data' / '00001'
            folder.mkdir(parents=True)
            (folder / 'benchmark_128k.json').write_text('{"QAs": []}')
            arguments = ['run_longctx.py', '--judge_only', '--model', 'answer-model', '--tiers', '128k',
                         '--type', 'omni', '--input_dir', str(folder.parent), '--output_base', directory]
            environment = {'JUDGE_API_KEY': 'judge-test-key', 'JUDGE_API_BASE': 'https://judge.example/v1'}
            with patch.dict(os.environ, environment, clear=True), patch.object(sys, 'argv', arguments), \
                 patch.object(runner, 'eval_single_run') as evaluate, contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            self.assertEqual(evaluate.call_args.kwargs['judge_model'], 'glm-5.2')
            self.assertEqual(evaluate.call_args.kwargs['judge_api_key'], 'judge-test-key')

    def test_dry_run_does_not_evaluate(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / '00001'
            folder.mkdir()
            (folder / 'benchmark_128k.json').write_text('{"QAs": []}')
            arguments = ['run_longctx.py', '--dry_run', '--input_dir', directory]
            with patch.object(sys, 'argv', arguments), patch.object(runner, 'eval_single_run') as evaluate, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            evaluate.assert_not_called()

    def test_openai_judge_uses_bearer_authentication(self):
        response = subprocess.CompletedProcess([], 0, stdout=b'{"choices":[{"message":{"content":"ok"}}]}')
        with patch.object(utils.subprocess, 'run', return_value=response) as request:
            utils.llm_chat(api_key='judge-test-key', base_url='https://judge.example/v1', model='glm-5.2',
                           contents=[{'role': 'USER', 'parts': [{'text': 'score'}]}], temperature=0)
        self.assertIn('Authorization: Bearer judge-test-key', request.call_args.args[0])

    def test_global_judge_default_is_independent_of_answer_model(self):
        with patch.dict(os.environ, {'LLM_MODEL': 'answer-model', 'JUDGE_MODEL': ''}, clear=True):
            spec = importlib.util.spec_from_file_location('paper_defaults', ROOT / 'omnimembench/config/defaults.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertEqual(module.JUDGE_MODEL, 'glm-5.2')


if __name__ == '__main__':
    unittest.main()
