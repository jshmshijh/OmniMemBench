import re
import threading
from omnimembench.engine.memory.BaseMemory import ExplicitMemory


class Mem0(ExplicitMemory):
    """
    Adapter that wraps mem0 (https://github.com/mem0ai/mem0) into OmniMemBench's
    ExplicitMemory interface.

    Config fields (via DefaultMem0MemoryConfig):
        mem0.user_id          : str   — mem0 user identifier
        mem0.top_k            : int   — retrieval top-k
        mem0.llm_provider     : str   — e.g. "openai", "litellm"
        mem0.llm_model        : str   — e.g. "gpt-4o-mini"
        mem0.llm_api_key      : str
        mem0.llm_base_url     : str
        mem0.embedding_provider: str  — e.g. "openai"
        mem0.embedding_model  : str   — e.g. "text-embedding-3-small"
        mem0.embedding_api_key: str
        mem0.embedding_base_url: str
        mem0.enable_graph     : bool  — whether to enable graph memory (Kuzu)
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._local = threading.local()
        self._mem0_cfg = self.config.args.mem0
        self._memory = self._build_mem0()

    def _build_mem0(self):
        """Construct a mem0.Memory instance from config."""
        from mem0 import Memory
        cfg = self._mem0_cfg
        user_id = getattr(cfg, "user_id", "default_user")
        mem0_config = {
            "llm": {
                "provider": cfg.llm_provider,
                "config": {
                    "model": cfg.llm_model,
                    "api_key": cfg.llm_api_key,
                    "openai_base_url": getattr(cfg, "llm_base_url", None),
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": cfg.embedding_provider,
                "config": {
                    "model": cfg.embedding_model,
                    "api_key": getattr(cfg, "embedding_api_key", cfg.llm_api_key),
                    "openai_base_url": getattr(cfg, "embedding_base_url", None),
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": getattr(cfg, "qdrant_path", f"/tmp/mem0_qdrant_{user_id}"),
                    "collection_name": f"mem0_{user_id}",
                    "embedding_model_dims": getattr(cfg, "embedding_dim", 2048),
                },
            },
            "version": "v1.1",
            "anonymous_telemetry": False,
        }
        if getattr(cfg, "enable_graph", False):
            mem0_config["graph_store"] = {
                "provider": "kuzu",
                "config": {"db_path": f"/tmp/mem0_graph_{cfg.user_id}"},
            }
        return Memory.from_config(mem0_config)

    def _close_client(self):
        """Release qdrant file lock held by the current client."""
        try:
            self._memory.vector_store.client.close()
        except Exception:
            pass
        try:
            self._memory._telemetry_vector_store.client.close()
        except Exception:
            pass

    # ── Text parsing ─────────────────────────────────────────────────────────

    @staticmethod
    def _parse_observation_to_messages(observation):
        """Split observation text like 'Alice: hi\nassistant: hello' into
        a list of (role, name, content) tuples."""
        text = observation.get("text", "") if isinstance(observation, dict) else str(observation)
        timestamp = observation.get("timestamp", "") if isinstance(observation, dict) else ""

        # Pattern: line starts with "SpeakerName: content"
        parts = re.split(r"\n(?=\S+?:\s)", text)
        messages = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            match = re.match(r"^(.+?):\s(.*)", part, re.DOTALL)
            if match:
                name = match.group(1).strip()
                content = match.group(2).strip()
                role = "assistant" if name.lower() == "assistant" else "user"
            else:
                name, content, role = "user", part, "user"
            messages.append({"role": role, "name": name, "content": content, "timestamp": timestamp})
        return messages or [{"role": "user", "name": "user", "content": text, "timestamp": timestamp}]

    # ── ExplicitMemory interface ─────────────────────────────────────────────

    def reset(self) -> None:
        try:
            self._memory.delete_all(user_id=self._mem0_cfg.user_id)
        except Exception:
            import shutil, os as _os
            self._close_client()
            tmp_path = getattr(self._mem0_cfg, "qdrant_path",
                               f"/tmp/mem0_qdrant_{self._mem0_cfg.user_id}")
            if _os.path.exists(tmp_path):
                shutil.rmtree(tmp_path, ignore_errors=True)
            self._memory = self._build_mem0()
        self._local = threading.local()

    def store(self, observation) -> None:
        messages = self._parse_observation_to_messages(observation)
        dialogue_id = observation.get("dialogue_id", "") if isinstance(observation, dict) else ""
        for msg in messages:
            text = (
                f"{msg['content']}\n"
                f"Speaker Name: {msg['name']}\n"
                f"Speaker Role: {msg['role']}"
            )
            try:
                self._memory.add(
                    messages={"content": text, "role": msg["role"], "name": msg["name"]},
                    user_id=self._mem0_cfg.user_id,
                    metadata={"timestamp": msg["timestamp"], "dialogue_id": dialogue_id},
                )
            except Exception as e:
                print(f"[Mem0] store error: {e}")

    def recall(self, query) -> str:
        query_text = query.get("text", query) if isinstance(query, dict) else str(query)
        top_k = int(getattr(self._mem0_cfg, "top_k", 10))

        result = self._memory.search(query=query_text, user_id=self._mem0_cfg.user_id, limit=top_k)
        memories = result.get("results", [])

        # Expose retrieved ids for retrieval evaluation
        self._local.last_retrieved_ids = []
        self._local.last_retrieved_texts = []

        formatted_parts = []
        for i, item in enumerate(memories):
            content = item.get("memory", "")
            metadata = item.get("metadata", {})
            ts = metadata.get("timestamp", "")
            did = metadata.get("dialogue_id", "")
            if did:
                self._local.last_retrieved_ids.append(did)
            self._local.last_retrieved_texts.append(content)

            parts = [f"[{i + 1}] {content}"]
            if ts:
                parts.append(f"Time: {ts}")
            formatted_parts.append("\n".join(parts))

        # Graph relations (if enabled)
        relations = result.get("relations")
        if relations:
            formatted_parts.append("Graph Relations:\n" + "\n".join(str(r) for r in relations))

        if not formatted_parts:
            return getattr(self.config.args, "empty_memory", "None")

        return "[Memory Start]\n" + "\n".join(formatted_parts) + "\n[Memory End]"

    def display(self) -> None:
        all_mems = self._memory.get_all(user_id=self._mem0_cfg.user_id)
        print(f"[Mem0] Total memories: {len(all_mems.get('results', []))}")

    def manage(self, operation, **kwargs) -> None:
        pass

    def optimize(self, **kwargs) -> None:
        pass

    def _get_state(self):
        state = {
            'qdrant_path': getattr(self._mem0_cfg, 'qdrant_path',
                                   f'/tmp/mem0_qdrant_{self._mem0_cfg.user_id}'),
            'user_id': self._mem0_cfg.user_id,
        }
        self._close_client()
        return state

    def _set_state(self, state):
        self._close_client()
        self._mem0_cfg.qdrant_path = state['qdrant_path']
        self._mem0_cfg.user_id = state['user_id']
        self._memory = self._build_mem0()