# Multimodal Rebuild

Downloads media files referenced by benchmark data from public datasets.

## Usage

```bash
# Preview what will be downloaded
python scripts/rebuild_multimodal/rebuild.py --dry-run

# Download all media (~25 GB)
python scripts/rebuild_multimodal/rebuild.py

# Download for specific samples only
python scripts/rebuild_multimodal/rebuild.py --runs 00001,00003

# Use a HuggingFace mirror
HF_ENDPOINT=https://your-mirror.com python scripts/rebuild_multimodal/rebuild.py
```

## Source Datasets

Media is downloaded from these public HuggingFace/GitHub datasets:

| Source | Modality | HF Repo |
|--------|----------|---------|
| MMAU | Audio | lmms-lab/mmau |
| Flickr30k | Image | nlphuji/flickr30k |
| OmniBench | Audio+Image | m-a-p/OmniBench |
| ChartQA | Image | HuggingFaceM4/ChartQA |
| MMMU | Image | MMMU/MMMU |
| MMMU-Pro | Image | MMMU/MMMU_Pro |
| MVBench | Video | OpenGVLab/MVBench |
| LibriSpeech | Audio | openslr/librispeech_asr |
| MusicBench | Audio | amaai-lab/MusicBench |
| ESC-50 | Audio | GitHub/karolpiczak/ESC-50 |

## How It Works

1. Scans benchmark JSONs for `multimodal/...` paths
2. Maps each path to its source dataset via crosswalk files
3. Downloads source data and extracts referenced media
4. Places files at the expected relative paths

The process is idempotent — existing files are skipped.
