"""Global defaults driven by environment variables."""

import os


def _env(key, default=""):
    return os.environ.get(key, default)


# --- API Configuration ---
LLM_API_KEY = _env("LLM_API_KEY")
LLM_API_BASE = _env("LLM_API_BASE", "http://localhost:8000/v1")
LLM_MODEL = _env("LLM_MODEL")

EMB_SERVER_URL = _env("EMB_SERVER_URL", "http://localhost:8010/v1")
EMB_MODEL_PATH = _env("EMB_MODEL_PATH")

JUDGE_API_BASE = _env("JUDGE_API_BASE") or LLM_API_BASE
JUDGE_API_KEY = _env("JUDGE_API_KEY") or LLM_API_KEY
JUDGE_MODEL = _env("JUDGE_MODEL") or "glm-5.2"

# --- GPU Configuration ---
USABLE_GPU = _env("CUDA_VISIBLE_DEVICES", "0")

# --- Retrieval Defaults ---
RETRIEVAL_TOP_K = int(_env("RETRIEVAL_TOP_K", "10"))
CONTEXT_TOP_K = int(_env("CONTEXT_TOP_K", "10"))
GRAPH_MAX_NODES = 10
REFLECTION_TOP_K = 10

# --- Tokenizer / Model Paths ---
BACKBONE_PATH = _env("MODEL_PATH")
GME_QWEN2_VL_7B_PATH = _env("GME_7B_PATH")
GME_QWEN2_VL_2B_PATH = _env("GME_2B_PATH")
NEMOTRON_PATH = _env("EMB_MODEL_PATH")

# --- Encoder Presets ---
OMNI_ENCODER_CAPTION = {"method": "OmniEncoder", "path": NEMOTRON_PATH, "emb_mode": "caption"}
OMNI_ENCODER_RAW = {"method": "OmniEncoder", "path": NEMOTRON_PATH, "emb_mode": "raw"}
OMNI_ENCODER = OMNI_ENCODER_RAW

GLOBAL_CONFIG = {"usable_gpu": USABLE_GPU}
