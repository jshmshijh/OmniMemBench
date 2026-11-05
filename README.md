# OmniMemBench: Towards Scalable Evaluation of Long-Term Omni-Modal Agent Memory

**NeurIPS 2026 · Evaluations and Datasets Track**

Junhan Shi, Qiyi Wang, Zhenning Shi, Xiongwei Wu, Ming Ma, Baoqi Pei, Yong Jiang, Qing Li, Weigao Sun, Yiran Zhong, Steven Hoi

[Dataset](https://huggingface.co/datasets/jshmsjh/OmniMemBench) · [Dataset documentation](https://huggingface.co/datasets/jshmsjh/OmniMemBench/blob/main/DATASHEET.md) · [Citation](#citation)

OmniMemBench evaluates long-term agent memory over multi-session dialogues containing text, images, audio, and video. It provides a common testbed for long-context models, retrieval-augmented generation, and structured memory systems.

- **Perceptually grounded evaluation:** an anti-leakage construction pipeline reduces textual shortcuts on questions requiring media evidence.
- **Retrieval diagnosis across memory representations:** Clue Coverage and Entry Relevance evaluate retrieved content, including paraphrased and consolidated memories.
- **Controlled context scaling:** content-isolated distractor sessions expand histories from 128K to 1M tokens while preserving target evidence and answers.

## Benchmark

| Context tier | Characters | QA instances |
|---|---:|---:|
| 128K | 103 | 2,719 |
| 256K | 51 | 1,312 |
| 512K | 51 | 1,312 |
| 1M | 51 | 1,312 |
| **Across all tiers** | **103 distinct characters** | **6,655** |

The 256K, 512K, and 1M tiers reuse the same 51-character subset and target questions, with additional distractor sessions. The total counts QA instances across tiers. The release contains 256 character-tier JSON files.

| Type | Task |
|---|---|
| SH | Single-Hop |
| MH | Multi-Hop |
| TR | Temporal Reasoning |
| CR | Causal Reasoning |
| ADV | Adversarial |
| REF | Answer Refusal |
| UPD | Knowledge Update |
| SUM | Summarization |

## Evaluation protocol

The paper uses **GLM-5.2 at temperature 0** for answer scoring and retrieval judgments.

| Metric | Definition |
|---|---|
| Non-SUM QA score | 0 (incorrect), 0.5 (partially correct), or 1 (correct) |
| SUM score | Fraction of the question's fixed reference nuggets covered by the answer |
| Overall | Mean of one score per valid question, including SUM |
| Clue Coverage (CC) | Fraction of evidence clues semantically covered by retrieved entries, averaged over valid questions |
| Entry Relevance (ER) | Fraction of retrieved entries relevant to the evidence clues, averaged over valid questions |

Missing responses and invalid judge outputs are excluded from the corresponding average. CC/ER use questions with nonempty clue and retrieved-entry sets. Context-scaling comparisons use the same 51 characters at every tier, including 128K; paired comparisons use their common valid questions.

The default pipeline setting in the paper uses Qwen3.5-35B-A3B for storage, Gemini-3-Flash for QA, retrieval K=10, and the 128K tier. Configure the corresponding model endpoints when reproducing that setting.

EM, F1, BLEU, and ID-based retrieval metrics are also available as auxiliary diagnostics. Judge prompts are in [`omnimembench/engine/evaluate/`](omnimembench/engine/evaluate/).

## Setup

### Install

```bash
git clone https://github.com/jshmshijh/OmniMemBench.git
cd OmniMemBench
pip install -e ".[all]"
```

Python 3.10 or newer is required. `all` installs every optional group except `video`:

| Group | Contents |
|---|---|
| `eval` | Lexical metrics and BERTScore |
| `rebuild` | Media reconstruction |
| `amem`, `hipporag`, `mem0`, `langmem` | External memory baselines; `baselines` installs all four |
| `video` | `decord` frame extraction for raw video embeddings; wheels are available for x86_64 |
| `dev` | Test and lint tools |

### Download the benchmark

```bash
python download_data.py
```

The script downloads benchmark JSONs, crosswalk files, the sample index, and dataset metadata from [`jshmsjh/OmniMemBench`](https://huggingface.co/datasets/jshmsjh/OmniMemBench) into the repository root. Use `--tiers 128k` to fetch a single tier, `--no_crosswalk` to skip the media mapping, and `--revision` to pin a dataset version.

To reconstruct the referenced images, audio, and video from their public source datasets:

```bash
python scripts/rebuild_multimodal/rebuild.py --runs 00001
```

Omit `--runs` to reconstruct all referenced media. See the [media guide](scripts/rebuild_multimodal/README.md) for source datasets and download options.

### Configure model and judge endpoints

```bash
cp .env.example .env
# Fill in the model, embedding, and judge settings in .env, then load them:
set -a
source .env
set +a
```

| Variable | Purpose |
|---|---|
| `LLM_MODEL`, `LLM_API_BASE`, `LLM_API_KEY` | Model answering the benchmark questions |
| `JUDGE_MODEL` | Paper judge; defaults to `glm-5.2` |
| `JUDGE_API_BASE`, `JUDGE_API_KEY` | GLM judge endpoint and key; blank values use the LLM endpoint and key for a shared gateway |
| `MODEL_PATH` | Tokenizer used for context truncation |
| `EMB_MODEL_PATH` | Local path or Hugging Face ID of the retrieval encoder, [`nvidia/omni-embed-nemotron-3b`](https://huggingface.co/nvidia/omni-embed-nemotron-3b), which has its own non-commercial license |
| `EMB_SERVER_URL` | OpenAI-compatible embedding server |
| `OMNIMEMBENCH_HF_REPO` | Dataset repository; defaults to `jshmsjh/OmniMemBench` |
| `HF_ENDPOINT` | Hugging Face endpoint for data and media downloads |

Set `JUDGE_MODEL` to your provider's GLM-5.2 model ID if its naming differs. The judge is configured independently of the model being evaluated.

## Run a memory pipeline

```bash
python benchmark/run_bench.py \
    --memory_name NaiveRAG \
    --data_name 00001 \
    --tier benchmark_128k \
    --data_root ./data \
    --llm_name "$LLM_MODEL" \
    --emb_mode caption \
    --retrieval_topk 10 \
    --save_results
```

For separate storage and QA backbones, add `--store_llm`, `--store_llm_base`, and `--store_llm_key` to inference, then add `--result_subdir store` to the scoring command below. These runs save results under `result_debug/{llm_name}/store/`; single-backbone runs use `result_debug/{llm_name}/`.

Score the saved answers and retrieved content with GLM-5.2:

```bash
python omnimembench/engine/evaluate/evaluation.py \
    --llm_name "$LLM_MODEL" \
    --memory_name NaiveRAG \
    --emb_mode caption \
    --data_root ./data \
    --tier benchmark_128k \
    --data_name 00001 \
    --enable_llm_judge \
    --enable_retrieval_judge \
    --judge_model "$JUDGE_MODEL" \
    --num_workers 4
```

Use `--all_datasets` in place of `--data_name 00001` to evaluate a complete tier. Keep the model, memory method, and embedding mode consistent between inference and scoring. Saved evaluations record the judge model; resume reuses scores from that same judge.

Available memory methods include SlidingWindow, NaiveRAG, GenerativeAgents, MemGPT, Reflexion, MultimodalRAG, NeuralGraph, AUGUSTUS, UniversalRAG, AMem, Mem0, LangMem, and HippoRAG. AMem, Mem0, LangMem, and HippoRAG need the `baselines` dependency group. With `--emb_mode raw`, video embeddings need the `video` group; without `decord`, the encoder falls back to the video caption.

## Run a long-context model

```bash
python benchmark/longctx/run_longctx.py \
    --input_dir ./data \
    --tiers 128k \
    --type omni \
    --model "$LLM_MODEL" \
    --api_key "$LLM_API_KEY" \
    --base_url "$LLM_API_BASE" \
    --judge_model "$JUDGE_MODEL" \
    --enable_judge
```

The runner reads `JUDGE_API_BASE` and `JUDGE_API_KEY` independently; `--judge_api_base` and `--judge_api_key` override them. Use `--judge_only` to score saved answers without rerunning inference.

Modes are `omni` (raw image, audio, and video), `vl` (raw image and video, captioned audio), and `text` (all media represented by captions). All modes use the same concise reference answers from the dataset's `answer` field and the same judge prompts; SUM uses fixed reference nuggets. Summaries report `overall_score`, including SUM, and `valid_judged`, its denominator; task-specific and auxiliary metrics are reported alongside it.

The Omni and VL tracks use the 57 characters listed in [`benchmark/longctx/track_run_ids.txt`](benchmark/longctx/track_run_ids.txt) (1,453 QAs at 128K). Add `--run_ids "$(cat benchmark/longctx/track_run_ids.txt)"` to the command above to run them; the `run_infer_*.sh` and `run_judge_*.sh` scripts in `benchmark/longctx/` already do.

Relative media paths resolve from the repository root, or from `--media_root /path/to/root`. Local images, audio, and video are sent inline; HTTP(S) media URLs are also supported.

Judgments use the paper's score domains: QA scores are 0, 0.5, or 1; SUM requires a binary label for every reference nugget; CC/ER requires one binary label per clue/entry. Failed judgments are retried up to the configured attempt limit and saved with their errors. Summaries exclude them from score denominators and report their counts. Valid zero scores remain in the average.

## Data Format

Each `data/{run_id}/benchmark_{tier}.json`:

```json
{
  "character_profile": {"persona_summary": "...", "conversation_style": "..."},
  "multi_session_dialogues": [
    {
      "session_id": "S1",
      "date": "2023-01-15",
      "dialogues": [
        {
          "round": "S1:1",
          "user": "...",
          "assistant": "...",
          "mm": [{"mm_id": "...", "type": "audio", "url": "multimodal/...", "caption": "..."}]
        }
      ]
    }
  ],
  "QAs": [
    {
      "type": "MH",
      "question": "...",
      "answer": "...",
      "evidence_chain": [],
      "clue": ["S1:3", "S4:7"]
    }
  ]
}
```

## Data generation

The construction scripts and stage overview are in [`generation/`](generation/).

## Acknowledgements

The memory framework and evaluation script in `omnimembench/engine/` are adapted from [Mem-Gallery](https://github.com/YuanchenBei/Mem-Gallery), which builds on [MemEngine](https://github.com/nuster1128/MemEngine). The HippoRAG and AMem baselines include modified copies of [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) and [A-MEM](https://github.com/agiresearch/A-mem). [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists their sources and MIT license terms. Benchmark media come from the public datasets listed in the [media guide](scripts/rebuild_multimodal/README.md).

## Citation

```bibtex
@inproceedings{shi2026omnimembench,
  title     = {OmniMemBench: Towards Scalable Evaluation of Long-Term Omni-Modal Agent Memory},
  author    = {Shi, Junhan and Wang, Qiyi and Shi, Zhenning and Wu, Xiongwei and Ma, Ming and Pei, Baoqi and Jiang, Yong and Li, Qing and Sun, Weigao and Zhong, Yiran and Hoi, Steven},
  booktitle = {NeurIPS 2026 Evaluations and Datasets Track},
  year      = {2026},
  url       = {https://github.com/jshmshijh/OmniMemBench}
}
```

## License and contact

Code: [Apache License 2.0](LICENSE); components adapted from Mem-Gallery, MemEngine, HippoRAG, and A-MEM remain under the MIT License ([notices](THIRD_PARTY_NOTICES.md)). Benchmark data: [CC BY-NC 4.0](https://huggingface.co/datasets/jshmsjh/OmniMemBench). Source media retain their original dataset licenses.

Contact: [shijh23@mails.tsinghua.edu.cn](mailto:shijh23@mails.tsinghua.edu.cn).
