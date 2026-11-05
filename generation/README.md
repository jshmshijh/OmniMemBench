# Data Generation Pipeline

This directory contains the pipeline used to construct OmniMemBench from public multimodal QA datasets.

## Pipeline steps

1. `preprocess/` — LLM client shared by all stages (`call_gemini.py`)
2. `filter/` — Scenario tagging, sampling, captioning, and the media registry
3. `framework/` — Task type assignment, persona, event graph, and session outlines
4. `rewrite/` — QA rewriting with evidence chains and candidate selection
5. `session/` — Dialogue planning and generation, summaries, distractor sessions, and tier assembly
6. `evaluate/` — Consistency, grounding, and uniqueness audits
7. `step6-final_data_filter/` — Media-removed leakage filter and token-distribution filter

Prompt templates sit next to the scripts that use them: `filter/prompts/`, `framework/prompt/`, `rewrite/prompt/`, and `session/prompt/`.

## Inputs and configuration

- Candidate pools: the dataset's `crosswalk/datasets/` contains the processed (`original/`) and scenario-tagged (`scenario_tagged/`) source pools used for construction. After `python download_data.py`, link them to the path the scripts read: `ln -s ../../crosswalk/datasets generation/preprocess/datasets`.
- Media: local `multimodal/...` paths, as rebuilt by `scripts/rebuild_multimodal/`, are read from the repository root (or `MEDIA_ROOT`) and sent inline; HTTP(S) URLs are passed through.
- Model endpoint: all stages call an OpenAI-compatible endpoint through `preprocess/call_gemini.py`, configured with `LLM_API_BASE`, `LLM_MODEL`, and `LLM_API_KEY` from the environment or `generation/.env`.

## Usage

```bash
python generation/run_pipeline_128k.py --start 1 --end 10
```

`--start` and `--end` select characters `00001` to `00010`. Stages write their results under their own directories, for example `framework/output/`.

The final filters run separately. `step6-final_data_filter/02_mm_leakage_filter.py` applies the media-removed leakage filter to the audited benchmarks in `evaluate/output/04_filtered/` (`--tier_dir`). `03_token_distribution_filter.py` then writes the 103 released characters to `step6-final_data_filter/output_03/`.
