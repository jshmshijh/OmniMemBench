import re
import threading
from omnimembench.engine.memory.BaseMemory import ExplicitMemory

def _import_amem():
    from omnimembench.engine.baselines.amem.memory_system import AgenticMemorySystem
    return AgenticMemorySystem


class AMem(ExplicitMemory):
    """
    Adapter that wraps A-MEM (https://github.com/agiresearch/A-mem) into
    OmniMemBench's ExplicitMemory interface.

    Config fields (via DefaultAMemMemoryConfig):
        amem.user_id             : str
        amem.top_k               : int
        amem.llm_backend         : str   — "openai" or "ollama"
        amem.llm_model           : str
        amem.api_key             : str
        amem.base_url            : str
        amem.evo_threshold       : int   — consolidation trigger count
        amem.retriever_model     : str   — sentence-transformers model name
        amem.embedder_provider   : str   — "sentence-transformers" or "openai"
        amem.embedding_api_key   : str
        amem.embedding_base_url  : str
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._local = threading.local()
        self._cfg = self.config.args.amem
        self._memory = self._build_amem()

    def _build_amem(self):
        AgenticMemorySystem = _import_amem()
        cfg = self._cfg
        return AgenticMemorySystem(
            model_name=getattr(cfg, "retriever_model", "all-MiniLM-L6-v2"),
            llm_backend=getattr(cfg, "llm_backend", "openai"),
            llm_model=getattr(cfg, "llm_model", "gpt-4o-mini"),
            api_key=getattr(cfg, "api_key", ""),
            base_url=getattr(cfg, "base_url", None),
            evo_threshold=int(getattr(cfg, "evo_threshold", 100)),
            embedder_provider=getattr(cfg, "embedder_provider", "sentence-transformers"),
            embedding_api_key=getattr(cfg, "embedding_api_key", None),
            embedding_base_url=getattr(cfg, "embedding_base_url", None),
            user_id=getattr(cfg, "user_id", "default_user"),
        )

    @staticmethod
    def _parse_messages(observation):
        """Split 'Alice: hi\nassistant: hello' into (name, content) pairs."""
        text = observation.get("text", "") if isinstance(observation, dict) else str(observation)
        parts = re.split(r"\n(?=\S+?:\s)", text)
        messages = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            match = re.match(r"^(.+?):\s(.*)", part, re.DOTALL)
            if match:
                messages.append((match.group(1).strip(), match.group(2).strip()))
            else:
                messages.append(("user", part))
        return messages or [("user", text)]

    # ── ExplicitMemory interface ─────────────────────────────────────────────

    def reset(self) -> None:
        self._memory = self._build_amem()
        self._local = threading.local()

    def store(self, observation) -> None:
        timestamp = observation.get("timestamp", "") if isinstance(observation, dict) else ""
        dialogue_id = observation.get("dialogue_id", "") if isinstance(observation, dict) else ""
        messages = self._parse_messages(observation)
        for name, content in messages:
            text = f"Speaker {name} (role: {'assistant' if name.lower() == 'assistant' else 'user'}) says: {content}"
            try:
                self._memory.add_note(text, time=timestamp)
            except Exception as e:
                print(f"[AMem] store error: {e}")

    def recall(self, query) -> str:
        query_text = query.get("text", query) if isinstance(query, dict) else str(query)
        top_k = int(getattr(self._cfg, "top_k", 10))

        results = self._memory.search_agentic(query_text, k=top_k)

        self._local.last_retrieved_ids = []
        self._local.last_retrieved_texts = []

        if not results:
            return getattr(self.config.args, "empty_memory", "None")

        formatted_parts = []
        for i, mem in enumerate(results):
            content = mem.get("content", "")
            self._local.last_retrieved_texts.append(content)
            entry = (
                f"[{i + 1}] memory content: {content}\n"
                f"memory context: {mem.get('context', '')}\n"
                f"memory keywords: {mem.get('keywords', [])}\n"
                f"memory tags: {mem.get('tags', [])}\n"
                f"talk start time: {mem.get('timestamp', '')}"
            )
            formatted_parts.append(entry)

        return "[Memory Start]\n" + "\n".join(formatted_parts) + "\n[Memory End]"

    def display(self) -> None:
        print(f"[AMem] Total memories: {len(self._memory.memories)}")

    def manage(self, operation, **kwargs) -> None:
        if operation == "consolidate":
            self._memory.consolidate_memories()

    def optimize(self, **kwargs) -> None:
        pass

    def _get_state(self):
        # memories is a dict of {id: MemoryNote}, MemoryNote holds only plain data fields
        return {'memories': dict(self._memory.memories)}

    def _set_state(self, state):
        # Rebuild a fresh AMem instance, then replace memories dict directly
        self._memory = self._build_amem()
        self._memory.memories = state['memories']
        # Re-index into the retriever so search works correctly
        self._memory.consolidate_memories()