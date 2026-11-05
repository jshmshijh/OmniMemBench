"""All memory method configurations in one registry.

Usage:
    from omnimembench.config import get_memory_config
    config = get_memory_config("NaiveRAG", overrides={"recall": {"topk": 5}})
"""

from __future__ import annotations

from omnimembench.config.base import MemoryConfig
from omnimembench.config import defaults as D

# =============================================================================
# Shared function configs
# =============================================================================

_LLM = {
    "method": "APILLM",
    "name": D.LLM_MODEL,
    "api_key": D.LLM_API_KEY,
    "base_url": D.LLM_API_BASE,
    "temperature": 0.0,
}

_TRUNCATION = {
    "method": "LMTruncation",
    "mode": "token",
    "path": D.BACKBONE_PATH,
    "number": 32768,
}

_MM_TRUNCATION = {
    "method": "MMLMTruncation",
    "mode": "token",
    "number": 32768,
    "path": D.BACKBONE_PATH,
    "tokens_per_image": 0,
    "tokens_per_audio": 0,
    "tokens_per_video": 0,
    "max_images": 999,
}

_TEXT_RETRIEVAL = {
    "method": "TextRetrieval",
    "encoder": D.OMNI_ENCODER,
    "mode": "cosine",
    "topk": D.RETRIEVAL_TOP_K,
}

_TIME_RETRIEVAL = {"method": "TimeRetrieval", "mode": "raw", "topk": D.RETRIEVAL_TOP_K}
_VALUE_RETRIEVAL = {"method": "ValueRetrieval", "mode": "identical", "topk": D.RETRIEVAL_TOP_K}

_MULTIMODAL_RETRIEVAL = {
    "method": "MultiModalRetrieval",
    "encoder": D.OMNI_ENCODER,
    "mode": "cosine",
    "topk": D.RETRIEVAL_TOP_K,
}

_CONCEPT_RETRIEVAL = {
    "method": "ConceptBasedRetrieval",
    "topk": D.RETRIEVAL_TOP_K,
    "min_concept_overlap": 1,
    "concept_weight": 1.0,
}

_UTILIZATION = {
    "method": "ConcateUtilization",
    "prefix": "[Memory Start]",
    "suffix": "[Memory End]",
    "list_config": {"index": True, "sep": "\n"},
    "dict_config": {"key_format": "(%s)", "key_value_sep": "\n", "item_sep": "\n"},
}

_MM_UTILIZATION = {
    "method": "MultiModalUtilization",
    "prefix": "[Memory Start]",
    "suffix": "[Memory End]",
    "list_config": {"index": True, "sep": "\n"},
}

_DISPLAY = {
    "method": "FileDisplay",
    "prefix": "----- Current Memory Start (%s) -----",
    "suffix": "----- End -----",
    "key_format": "(%s)",
    "key_value_sep": "\n",
    "item_sep": "\n",
    "output_path": "logs/sample.log",
}

_IMPORTANCE_JUDGE = {
    "method": "LLMJudge",
    "LLM_config": _LLM,
    "prompt": {
        "template": (
            "On the scale of 1 to 10, where 1 is purely unimportant and 10 is "
            "extremely important, rate the likely importance of the following piece of message.\n"
            "Message: {message}\n"
            "Your should just output the rating number between from 1 to 10, "
            "and do not output any other texts."
        ),
        "input_variables": ["message"],
    },
    "post_scale": 10,
}

_GA_REFLECTOR = {
    "method": "GAReflector",
    "LLM_config": _LLM,
    "question_prompt": {
        "template": (
            "Information: {information}\n"
            "Given only the information above, what are {question_number} most salient "
            "highlevel questions we can answer about the subjects in the statements?\n"
            "Please output each question in a single line, and do not output any other messages."
        ),
        "input_variables": ["information", "question_number"],
    },
    "insight_prompt": {
        "template": (
            "Statements: {statements}\n"
            "What {insight_number} high-level insights can you infer from the above statements?\n"
            "Please output each insight in a single line (without index), "
            "and do not output any other messages."
        ),
        "input_variables": ["statements", "insight_number"],
    },
    "question_number": 3,
    "insight_number": 3,
    "threshold": 0.3,
    "reflection_topk": D.REFLECTION_TOP_K,
}

_TRIAL_REFLECTOR = {
    "method": "TrialReflector",
    "LLM_config": _LLM,
    "prompt": {
        "template": (
            "Previous Insight: {previous_insight}\n"
            "New Trial: {new_trial}\n"
            "You are an advanced agent that can improve based on self-refection. "
            "You have been given a new trial in which you have interacted with the environment.\n"
            "Based on the previous insight and the new trial, please generate a new insight text.\n"
            "You may keep the information in the previous insight.\n"
            "You may diagnose a possible reason for failure and devise a new, concise, "
            "high level plan that aims to mitigate the same failure. Use complete sentences.\n"
            "Your output should be consice, and do not output any other messages except "
            "for the new insight text.\n"
            "Here is an example:\n[Example]\n{example}\n"
            "You should just output the text of your new insight, without any other messages."
        ),
        "input_variables": ["previous_insight", "new_trial", "example"],
    },
    "example": (
        "Previous Insight: Alice likes to watch movies.\n"
        "New Trial: (step 1) Alice: Could you recommend some movies to me? "
        "Assistant: Sure! What about Titanic and Before Sunrise? "
        "(Step 2) Alice: Great! I like this genre. Assistant: You are welcome!\n"
        "Output: Alice likes to watch romantic movies."
    ),
}

_SUMMARIZER = {
    "method": "LLMSummarizer",
    "LLM_config": _LLM,
    "prompt": {
        "template": (
            "Content: {content}\n"
            "Summarize the above content concisely, extracting the main themes and key information.\n"
            "Please output your summary directly in a single line, "
            "and do not output any other messages."
        ),
        "input_variables": ["content"],
    },
}

_MEMGPT_TRIGGER = {
    "LLM_config": _LLM,
    "func_list": [
        {
            "name": "memory_retrieval",
            "args": ["query: str"],
            "args_type": ["str"],
            "func_description": (
                "retrieve query-related information from (external) archival storage, "
                "and add the result into working memory."
            ),
            "args_description": [
                "query is a string to retrieve relevant information (e.g., 'Alice\\'s name)."
            ],
        },
        {
            "name": "memory_recall",
            "args": ["query: str"],
            "args_type": ["str"],
            "func_description": (
                "retrieve query-related information from (external) recall storage, "
                "and add the result into FIFO memory."
            ),
            "args_description": [
                "query is a string to retrieve relevant information (e.g., 'Alice\\'s name)."
            ],
        },
        {
            "name": "memory_archive",
            "args": ["memory_list: list"],
            "args_type": ["list"],
            "func_description": (
                "archive some memory from FIFO memory into (external) archival storage."
            ),
            "args_description": ["the index list of FIFO memory (e.g., [0, 2, 3])."],
        },
        {
            "name": "memory_transfer",
            "args": ["memory_list: list"],
            "args_type": ["list"],
            "func_description": (
                "transfer some memory from FIFO memory into working memory."
            ),
            "args_description": ["the index list of FIFO memory (e.g., [0, 2, 3])."],
        },
        {
            "name": "memory_save",
            "args": ["memory_list: list"],
            "args_type": ["list"],
            "func_description": (
                "archive some memory from working memory into (external) archival storage."
            ),
            "args_description": ["the index list of working memory (e.g., [0, 2, 3])."],
        },
    ],
    "prompt": {
        "template": (
            "Query: {text}\n{memory_prompt}\n{function_prompt}\n"
            "Please choose some of the following functions to execute, "
            "and you should output each executed function in a single line.\n"
            "{warning_content}{no_execute_prompt}"
            "Your output should follow the following format in the examples, "
            "and do not output other messages.\n{few_shot}"
        ),
        "input_variables": [
            "text", "memory_prompt", "function_prompt",
            "warning_content", "no_execute_prompt", "few_shot",
        ],
    },
    "no_execute": "If you do not execute any functions, just output [No Excuate].",
    "few_shot": (
        "EXAMPLE 1:\nmemory_retrieval('Alice\\'s name')\nmemory_save([0, 1])\n"
        "EXAMPLE 2:\nmemory_recall('Bob\\'s name')\nmemory_transfer([0])\n"
        "EXAMPLE 3:\nmemory_archive([0])\n"
        "EXAMPLE 4:\n[No Excuate]"
    ),
}

_LLM_CONCEPT_EXTRACTOR = {
    "method": "LLMConceptExtractor",
    "max_concepts": 10,
    "llm_method": "APILLM",
    "llm": {
        "method": "APILLM",
        "api_key": D.LLM_API_KEY,
        "base_url": D.LLM_API_BASE,
        "name": D.LLM_MODEL,
        "temperature": 0.0,
    },
    "extraction_prompt": None,
}

_UNIVERSALRAG_ROUTING = {
    "method": "UniversalRAGRouting",
    "llm": _LLM,
    "modality_mapping": {
        "no": "no",
        "document": "document",
        "image": "image",
        "audio": "audio",
        "video": "video",
        "paragraph": "document",
        "clip": "image",
        "error": "document",
        "text": "document",
        "visual": "image",
        "sound": "audio",
        "music": "audio",
    },
}

# =============================================================================
# Operation configs
# =============================================================================

_RECALLS = {
    "FUMemory": {
        "method": "FUMemoryRecall",
        "truncation": _TRUNCATION,
        "utilization": _UTILIZATION,
        "empty_memory": "None",
    },
    "SlidingWindow": {
        "method": "SlidingWindowRecall",
        "truncation": _TRUNCATION,
        "utilization": _UTILIZATION,
        "time_retrieval": _TIME_RETRIEVAL,
        "empty_memory": "None",
    },
    "NaiveRAG": {
        "method": "NaiveRAGRecall",
        "truncation": _TRUNCATION,
        "utilization": _UTILIZATION,
        "text_retrieval": _TEXT_RETRIEVAL,
        "empty_memory": "None",
    },
    "GenerativeAgents": {
        "method": "GenerativeAgentsRecall",
        "truncation": _TRUNCATION,
        "utilization": _UTILIZATION,
        "text_retrieval": _TEXT_RETRIEVAL,
        "time_retrieval": {"method": "TimeRetrieval", "mode": "exp", "coef": {"decay": 0.995}},
        "importance_retrieval": _VALUE_RETRIEVAL,
        "importance_judge": _IMPORTANCE_JUDGE,
        "topk": D.RETRIEVAL_TOP_K,
        "empty_memory": "None",
    },
    "MemGPT": {
        "method": "MemGPTRecall",
        "truncation": _TRUNCATION,
        "utilization": _UTILIZATION,
        "warning_threshold": 0.7,
        "warning_content": "We suggest to execute memory_archive or memory_transfer.",
        "trigger": _MEMGPT_TRIGGER,
        "recall_retrieval": _TEXT_RETRIEVAL,
        "archival_retrieval": _TEXT_RETRIEVAL,
        "empty_memory": "None",
    },
    "MultimodalRAG": {
        "method": "MultimodalRAGRecall",
        "truncation": _TRUNCATION,
        "utilization": _MM_UTILIZATION,
        "multimodal_retrieval": _MULTIMODAL_RETRIEVAL,
        "empty_memory": "None",
    },
    "MMFUMemory": {
        "method": "MMFUMemoryRecall",
        "truncation": _MM_TRUNCATION,
        "utilization": _MM_UTILIZATION,
        "empty_memory": [],
    },
    "NeuralGraph": {
        "method": "NeuralGraphRecall",
        "truncation": _TRUNCATION,
        "utilization": _MM_UTILIZATION,
        "multimodal_retrieval": _MULTIMODAL_RETRIEVAL,
        "max_depth": 3,
        "max_nodes": D.GRAPH_MAX_NODES,
        "traversal_threshold": 0.4,
        "empty_memory": "None",
    },
    "AUGUSTUS": {
        "method": "AUGUSTUSRecall",
        "truncation": _TRUNCATION,
        "utilization": _MM_UTILIZATION,
        "multimodal_retrieval": _MULTIMODAL_RETRIEVAL,
        "concept_retrieval": _CONCEPT_RETRIEVAL,
        "max_depth": 3,
        "max_nodes": D.GRAPH_MAX_NODES,
        "traversal_threshold": 0.5,
        "embedding_weight": 0.5,
        "concept_weight": 0.5,
        "empty_memory": "None",
    },
    "UniversalRAG": {
        "method": "UniversalRAGRecall",
        "truncation": _TRUNCATION,
        "utilization": _MM_UTILIZATION,
        "top_k": D.RETRIEVAL_TOP_K,
        "empty_memory": "None",
    },
}

_STORES = {
    "FUMemory": {"method": "FUMemoryStore"},
    "SlidingWindow": {"method": "NaiveRAGStore"},
    "NaiveRAG": {"method": "NaiveRAGStore"},
    "GenerativeAgents": {"method": "GenerativeAgentsStore"},
    "MemGPT": {
        "method": "MemGPTStore",
        "flush_checker": _TRUNCATION,
        "summarizer": {
            "method": "LLMSummarizer",
            "LLM_config": _LLM,
            "prompt": {
                "template": (
                    "Recursive Summary: {recursive_summary}\n"
                    "Recent Memory: {flush_context}\n"
                    "Please update the Recursive Summary based on Recursive Summary "
                    "and summarizing Recent Memory.\n"
                    "Just output the new Recursive Summary, without any other messages."
                ),
                "input_variables": ["recursive_summary", "flush_context"],
            },
        },
    },
    "Reflexion": {"method": "FUMemoryStore"},
    "MultimodalRAG": {"method": "MultimodalRAGStore"},
    "MMFUMemory": {"method": "MMFUMemoryStore"},
    "NeuralGraph": {
        "method": "NeuralGraphStore",
        "similarity_threshold": 0.6,
        "max_edges_per_node": 10,
    },
    "AUGUSTUS": {
        "method": "AUGUSTUSStore",
        "similarity_threshold": 0.7,
        "max_edges_per_node": 5,
        "min_shared_concepts": 1,
    },
    "UniversalRAG": {"method": "UniversalRAGStore"},
}

# =============================================================================
# Complete memory configurations
# =============================================================================

MEMORY_CONFIGS: dict[str, dict] = {
    "SlidingWindow": {
        "name": "SlidingWindow",
        "is_multimodal": False,
        "storage": {},
        "recall": _RECALLS["SlidingWindow"],
        "store": _STORES["SlidingWindow"],
        "display": _DISPLAY,
        "global_config": D.GLOBAL_CONFIG,
    },
    "NaiveRAG": {
        "name": "NaiveRAG",
        "is_multimodal": False,
        "storage": {},
        "recall": _RECALLS["NaiveRAG"],
        "store": _STORES["NaiveRAG"],
        "display": _DISPLAY,
        "global_config": D.GLOBAL_CONFIG,
    },
    "GenerativeAgents": {
        "name": "GenerativeAgents",
        "is_multimodal": False,
        "storage": {},
        "recall": _RECALLS["GenerativeAgents"],
        "store": _STORES["GenerativeAgents"],
        "reflect": {"method": "GenerativeAgentsReflect", "reflector": _GA_REFLECTOR},
        "display": _DISPLAY,
        "global_config": D.GLOBAL_CONFIG,
    },
    "MemGPT": {
        "name": "MemGPT",
        "is_multimodal": False,
        "storage": {},
        "recall": _RECALLS["MemGPT"],
        "store": _STORES["MemGPT"],
        "display": _DISPLAY,
        "global_config": D.GLOBAL_CONFIG,
    },
    "Reflexion": {
        "name": "Reflexion",
        "is_multimodal": False,
        "storage": {},
        "recall": _RECALLS["FUMemory"],
        "store": _STORES["Reflexion"],
        "optimize": {"method": "RFMemoryOptimize", "reflector": _TRIAL_REFLECTOR},
        "display": _DISPLAY,
        "global_config": D.GLOBAL_CONFIG,
    },
    "MultimodalRAG": {
        "name": "MultimodalRAG",
        "is_multimodal": True,
        "storage": {},
        "recall": _RECALLS["MultimodalRAG"],
        "store": _STORES["MultimodalRAG"],
        "multimodal_retrieval": _MULTIMODAL_RETRIEVAL,
        "display": _DISPLAY,
        "global_config": D.GLOBAL_CONFIG,
    },
    "NeuralGraph": {
        "name": "NeuralGraph",
        "is_multimodal": True,
        "storage": {},
        "recall": _RECALLS["NeuralGraph"],
        "store": _STORES["NeuralGraph"],
        "multimodal_retrieval": _MULTIMODAL_RETRIEVAL,
        "display": _DISPLAY,
        "global_config": D.GLOBAL_CONFIG,
    },
    "AUGUSTUS": {
        "name": "AUGUSTUS",
        "is_multimodal": True,
        "storage": {},
        "recall": _RECALLS["AUGUSTUS"],
        "store": _STORES["AUGUSTUS"],
        "multimodal_retrieval": _MULTIMODAL_RETRIEVAL,
        "concept_extractor": _LLM_CONCEPT_EXTRACTOR,
        "concept_retrieval": _CONCEPT_RETRIEVAL,
        "display": _DISPLAY,
        "global_config": D.GLOBAL_CONFIG,
    },
    "UniversalRAG": {
        "name": "UniversalRAG",
        "is_multimodal": True,
        "storage": {"method": "UniversalRAGStorage", "encoder": D.OMNI_ENCODER},
        "recall": _RECALLS["UniversalRAG"],
        "store": _STORES["UniversalRAG"],
        "routing": _UNIVERSALRAG_ROUTING,
        "retrieval": {"method": "UniversalRAGRetrieval", "top_k": D.RETRIEVAL_TOP_K},
        "display": _DISPLAY,
        "global_config": D.GLOBAL_CONFIG,
    },
}

# =============================================================================
# External baseline configurations (share a common template)
# =============================================================================

_EXTERNAL_BASELINE_TEMPLATE = {
    "user_id": "default_user",
    "top_k": D.RETRIEVAL_TOP_K,
    "llm_backend": "openai",
    "llm_model": D.LLM_MODEL,
    "api_key": D.LLM_API_KEY,
    "base_url": D.LLM_API_BASE,
    "retriever_model": D.NEMOTRON_PATH,
    "embedder_provider": "openai",
    "embedding_api_key": D.LLM_API_KEY,
    "embedding_base_url": D.EMB_SERVER_URL,
}

MEMORY_CONFIGS["AMem"] = {
    "name": "AMem",
    "is_multimodal": False,
    "amem": {**_EXTERNAL_BASELINE_TEMPLATE, "evo_threshold": 100},
    "empty_memory": "None",
    "global_config": D.GLOBAL_CONFIG,
}

MEMORY_CONFIGS["Mem0"] = {
    "name": "Mem0",
    "is_multimodal": False,
    "mem0": {**_EXTERNAL_BASELINE_TEMPLATE},
    "empty_memory": "None",
    "global_config": D.GLOBAL_CONFIG,
}

MEMORY_CONFIGS["LangMem"] = {
    "name": "LangMem",
    "is_multimodal": False,
    "langmem": {**_EXTERNAL_BASELINE_TEMPLATE},
    "empty_memory": "None",
    "global_config": D.GLOBAL_CONFIG,
}

MEMORY_CONFIGS["HippoRAG"] = {
    "name": "HippoRAG",
    "is_multimodal": False,
    "hipporag": {**_EXTERNAL_BASELINE_TEMPLATE},
    "empty_memory": "None",
    "global_config": D.GLOBAL_CONFIG,
}


def get_memory_config(name: str, overrides: dict | None = None) -> MemoryConfig:
    """Get a memory configuration by name with optional overrides.

    Args:
        name: Memory method name (e.g., "NaiveRAG", "AUGUSTUS", "AMem")
        overrides: Optional dict to override specific config values

    Returns:
        MemoryConfig object

    Raises:
        KeyError: If name is not a registered memory method
    """
    if name not in MEMORY_CONFIGS:
        available = ", ".join(sorted(MEMORY_CONFIGS.keys()))
        raise KeyError(f"Unknown memory method '{name}'. Available: {available}")
    return MemoryConfig(MEMORY_CONFIGS[name], overrides)


def list_memory_methods() -> list[str]:
    """Return all registered memory method names."""
    return sorted(MEMORY_CONFIGS.keys())
