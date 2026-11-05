import os
import re
import threading
from omnimembench.engine.memory.BaseMemory import ExplicitMemory

def _import_hipporag():
    from omnimembench.engine.baselines.hipporag import HippoRAG
    from omnimembench.engine.baselines.hipporag.utils.config_utils import BaseConfig as HippoRAGBaseConfig
    return HippoRAG, HippoRAGBaseConfig


class HippoRAG(ExplicitMemory):
    """
    Adapter that wraps HippoRAG2 (https://github.com/OSU-NLP-Group/HippoRAG) into
    OmniMemBench's ExplicitMemory interface.

    No external database needed — graph is built in-memory with networkx.

    Config fields (via DefaultHippoRAGMemoryConfig):
        hipporag.top_k                      : int
        hipporag.llm_name                   : str   — e.g. "gpt-4o-mini"
        hipporag.llm_base_url               : str | None
        hipporag.embedding_model_name       : str   — e.g. "text-embedding-3-small"
        hipporag.embedding_base_url         : str | None
        hipporag.embedding_batch_size       : int
        hipporag.openie_mode                : str   — "online" / "offline" / "Transformers-offline"
        hipporag.damping                    : float
        hipporag.linking_top_k              : int
        hipporag.passage_node_weight        : float
        hipporag.synonymy_edge_sim_threshold: float
        hipporag.is_directed_graph          : bool
        hipporag.save_openie                : bool
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._local = threading.local()
        self._cfg = self.config.args.hipporag
        self._pending_docs = []
        self._build_hipporag()

    def _build_hipporag(self):
        HippoRAGCls, HippoRAGBaseConfig = _import_hipporag()
        cfg = self._cfg
        # HippoRAG's CacheOpenAI/OpenAIEmbeddingModel read api_key from env
        api_key = getattr(cfg, "llm_api_key", "")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        config_dict = {
            "llm_name": getattr(cfg, "llm_name", "gpt-4o-mini"),
            "llm_base_url": getattr(cfg, "llm_base_url", None),
            "embedding_model_name": getattr(cfg, "embedding_model_name", "text-embedding-3-small"),
            "embedding_base_url": getattr(cfg, "embedding_base_url", None),
            "embedding_batch_size": int(getattr(cfg, "embedding_batch_size", 16)),
            "openie_mode": getattr(cfg, "openie_mode", "online"),
            "retrieval_top_k": int(getattr(cfg, "top_k", 10)),
            "linking_top_k": int(getattr(cfg, "linking_top_k", 5)),
            "damping": float(getattr(cfg, "damping", 0.5)),
            "passage_node_weight": float(getattr(cfg, "passage_node_weight", 0.05)),
            "is_directed_graph": getattr(cfg, "is_directed_graph", False),
            "synonymy_edge_sim_threshold": float(getattr(cfg, "synonymy_edge_sim_threshold", 0.8)),
            "save_openie": getattr(cfg, "save_openie", True),
        }
        # Allow dynamic save_dir injection (for parallel run isolation)
        save_dir = getattr(cfg, "save_dir", None)
        if save_dir:
            config_dict["save_dir"] = save_dir
        global_config = HippoRAGBaseConfig(**config_dict)
        self._memory = HippoRAGCls(global_config)

    # ── ExplicitMemory interface ─────────────────────────────────────────────

    def reset(self) -> None:
        self._pending_docs.clear()
        self._build_hipporag()
        self._local = threading.local()

    def store(self, observation) -> None:
        text = observation.get("text", "") if isinstance(observation, dict) else str(observation)
        timestamp = observation.get("timestamp", "") if isinstance(observation, dict) else ""
        if timestamp:
            text += f"\nTimestamp: {timestamp}"
        try:
            self._memory.index([text])
        except Exception as e:
            print(f"[HippoRAG] store error: {e}")

    def recall(self, query) -> str:
        query_text = query.get("text", query) if isinstance(query, dict) else str(query)
        top_k = int(getattr(self._cfg, "top_k", 10))

        try:
            solutions = self._memory.retrieve(queries=[query_text], num_to_retrieve=top_k)
        except Exception as e:
            print(f"[HippoRAG] recall error: {e}")
            return getattr(self.config.args, "empty_memory", "None")

        solution = solutions[0]
        self._local.last_retrieved_ids = []
        self._local.last_retrieved_texts = []

        if not solution.docs:
            return getattr(self.config.args, "empty_memory", "None")

        formatted_parts = []
        for i, doc in enumerate(solution.docs):
            self._local.last_retrieved_texts.append(doc)
            score = solution.doc_scores[i] if solution.doc_scores is not None and i < len(solution.doc_scores) else 0.0
            formatted_parts.append(f"[{i + 1}] (score: {score:.4f}) {doc}")

        return "[Memory Start]\n" + "\n".join(formatted_parts) + "\n[Memory End]"

    def display(self) -> None:
        count = len(self._memory.chunk_embedding_store.get_all_texts()) if hasattr(self._memory, 'chunk_embedding_store') else 0
        print(f"[HippoRAG] Total indexed docs: {count}")

    def manage(self, operation, **kwargs) -> None:
        pass

    def optimize(self, **kwargs) -> None:
        pass

    def _get_state(self):
        # HippoRAG persists all data (graph.pickle + parquet) under working_dir on disk.
        # Save the save_dir so _set_state can point _build_hipporag at the right location.
        return {
            'save_dir': getattr(self._cfg, 'save_dir', None),
            'working_dir': getattr(self._memory, 'working_dir', None),
        }

    def _set_state(self, state):
        # Restore save_dir into config; _build_hipporag will initialise graph from disk
        if state.get('save_dir'):
            self._cfg.save_dir = state['save_dir']
        self._build_hipporag()