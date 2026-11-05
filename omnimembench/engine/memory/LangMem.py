import re
import uuid
import threading
from omnimembench.engine.memory.BaseMemory import ExplicitMemory


def _parse_qwen_xml_tool_calls(content):
    """Parse Qwen's XML-style tool calls into LangChain tool_calls format.

    Qwen format:
        <tool_call>
        <function=FuncName>
        <parameter=param>value</parameter>
        </function>
        </tool_call>
    """
    tool_calls = []
    for tc in re.finditer(
        r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>',
        content, re.DOTALL,
    ):
        name = tc.group(1)
        body = tc.group(2)
        args = {}
        for pm in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', body, re.DOTALL):
            args[pm.group(1)] = pm.group(2).strip()
        tool_calls.append({
            "name": name,
            "args": args,
            "id": f"call_{uuid.uuid4().hex[:8]}",
        })
    return tool_calls


def _patch_llm_for_qwen_tool_calls(llm):
    """Monkey-patch a ChatModel instance so Qwen XML tool calls are
    converted to standard tool_calls that LangMem/TrustCall can consume."""
    original_generate = llm._generate

    def _patched_generate(messages, stop=None, run_manager=None, **kwargs):
        result = original_generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for gen in result.generations:
            msg = gen.message
            if not msg.tool_calls and isinstance(msg.content, str) and '<tool_call>' in msg.content:
                parsed = _parse_qwen_xml_tool_calls(msg.content)
                if parsed:
                    msg.tool_calls = parsed
        return result

    llm._generate = _patched_generate


class LangMem(ExplicitMemory):
    """
    Adapter that wraps LangMem (https://github.com/langchain-ai/langmem) into
    OmniMemBench's ExplicitMemory interface.

    No external database needed — uses LangGraph's InMemoryStore.

    Config fields (via DefaultLangMemMemoryConfig):
        langmem.user_id                : str
        langmem.top_k                  : int
        langmem.llm_model              : str   — e.g. "openai:gpt-4o-mini"
        langmem.llm_api_key            : str
        langmem.llm_base_url           : str
        langmem.retriever_model        : str   — e.g. "huggingface:all-MiniLM-L6-v2"
        langmem.retriever_dim          : int
        langmem.query_model            : str | None
        langmem.enable_inserts         : bool
        langmem.enable_deletes         : bool
        langmem.query_limit            : int
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._local = threading.local()
        self._cfg = self.config.args.langmem
        self._memory_ids = {}
        self._init_langmem()

    def _init_langmem(self):
        # Python 3.10 compat: typing.NotRequired was added in 3.11;
        # patch it in before langmem imports so its module-level code doesn't crash.
        import typing
        if not hasattr(typing, "NotRequired"):
            from typing_extensions import NotRequired
            typing.NotRequired = NotRequired

        print("[LangMem] Step 1: Importing modules...")
        from langgraph.store.memory import InMemoryStore
        from langchain.chat_models import init_chat_model
        from langchain.embeddings import init_embeddings
        from langmem import create_memory_store_manager
        print("[LangMem] Step 1 done.")

        cfg = self._cfg
        llm_model = getattr(cfg, "llm_model", "openai:gpt-4o-mini")
        print(f"[LangMem] Step 2: Initializing LLM with model={llm_model}")
        llm_kwargs = {}
        if getattr(cfg, "llm_api_key", None):
            llm_kwargs["api_key"] = cfg.llm_api_key
        if getattr(cfg, "llm_base_url", None):
            llm_kwargs["base_url"] = cfg.llm_base_url

        self._llm = init_chat_model(llm_model, **llm_kwargs)
        _patch_llm_for_qwen_tool_calls(self._llm)
        print("[LangMem] Step 2 done.")

        retriever_model = getattr(cfg, "retriever_model", "huggingface:all-MiniLM-L6-v2")
        retriever_dim = int(getattr(cfg, "retriever_dim", 384))
        print(f"[LangMem] Step 3: Initializing embedding with model={retriever_model}, dim={retriever_dim}")
        print(f"[LangMem]   embedding_base_url={getattr(cfg, 'embedding_base_url', 'NOT SET')}")
        emb_kwargs = {}
        if getattr(cfg, "embedding_api_key", None):
            emb_kwargs["api_key"] = cfg.embedding_api_key
        if getattr(cfg, "embedding_base_url", None):
            emb_kwargs["base_url"] = cfg.embedding_base_url

        print("[LangMem] Step 4: Creating InMemoryStore...")
        self._store = InMemoryStore(
            index={
                "dims": retriever_dim,
                "embed": init_embeddings(retriever_model, **emb_kwargs),
                "fields": ["content"],
            },
        )
        print("[LangMem] Step 4 done.")

        query_model_name = getattr(cfg, "query_model", None)
        print(f"[LangMem] Step 5: Initializing query model={query_model_name}")
        self._query_model = (
            init_chat_model(query_model_name, **llm_kwargs)
            if query_model_name else None
        )
        print("[LangMem] Step 5 done.")

        user_id = getattr(cfg, "user_id", "default_user")
        print(f"[LangMem] Step 6: Creating memory store manager...")
        self._manager = create_memory_store_manager(
            self._llm,
            enable_inserts=getattr(cfg, "enable_inserts", True),
            enable_deletes=getattr(cfg, "enable_deletes", True),
            query_model=self._query_model,
            query_limit=int(getattr(cfg, "query_limit", 5)),
            namespace=("memories", user_id),
            store=self._store,
        )
        print("[LangMem] Step 6 done. Initialization complete.")

    @staticmethod
    def _parse_messages(observation):
        text = observation.get("text", "") if isinstance(observation, dict) else str(observation)
        timestamp = observation.get("timestamp", "") if isinstance(observation, dict) else ""
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
        self._memory_ids.clear()
        self._init_langmem()
        self._local = threading.local()

    def store(self, observation) -> None:
        messages = self._parse_messages(observation)
        msg_dicts = [
            {
                "role": m["role"],
                "name": m["name"],
                "content": f"{m['content']}\nTimestamp: {m['timestamp']}",
            }
            for m in messages
        ]
        try:
            final_puts = self._manager.invoke({"messages": msg_dicts})
            for put in final_puts:
                self._memory_ids[put["key"]] = put["value"]
        except Exception as e:
            print(f"[LangMem] store error: {e}")

    def recall(self, query) -> str:
        query_text = query.get("text", query) if isinstance(query, dict) else str(query)
        top_k = int(getattr(self._cfg, "top_k", 10))

        try:
            results = self._manager.search(query=query_text, limit=top_k)
        except Exception as e:
            print(f"[LangMem] recall error: {e}")
            return getattr(self.config.args, "empty_memory", "None")

        self._local.last_retrieved_ids = []
        self._local.last_retrieved_texts = []

        if not results:
            return getattr(self.config.args, "empty_memory", "None")

        formatted_parts = []
        for i, mem in enumerate(results):
            mem_dict = mem.dict() if hasattr(mem, "dict") else mem
            content = mem_dict.get("value", {}).get("content", str(mem_dict))
            self._local.last_retrieved_texts.append(content)
            formatted_parts.append(f"[{i + 1}] {content}")

        return "[Memory Start]\n" + "\n".join(formatted_parts) + "\n[Memory End]"

    def display(self) -> None:
        print(f"[LangMem] Total memories: {len(self._memory_ids)}")

    def manage(self, operation, **kwargs) -> None:
        pass

    def optimize(self, **kwargs) -> None:
        pass

    def _get_state(self):
        # _memory_ids maps key -> value (plain dicts), fully picklable
        return {'memory_ids': dict(self._memory_ids)}

    def _set_state(self, state):
        # Rebuild LangMem (new InMemoryStore), then restore items directly
        self._init_langmem()
        self._memory_ids = {}
        user_id = getattr(self._cfg, 'user_id', 'default_user')
        namespace = ('memories', user_id)
        for key, value in state['memory_ids'].items():
            try:
                self._store.put(namespace, key, value)
                self._memory_ids[key] = value
            except Exception as e:
                print(f'[LangMem] _set_state restore error key={key}: {e}')