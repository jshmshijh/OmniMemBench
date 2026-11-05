"""Shared LLM client with OpenAI-compatible API support.

Usage:
    from omnimembench.llm import create_llm, OpenAILLM, get_token_counter
    from omnimembench.llm.media import build_image_content, encode_image_base64

    llm = create_llm({"method": "APILLM", "name": "gpt-4o", "api_key": "...", "base_url": "..."})
    response = llm.run([{"role": "user", "content": "Hello"}])
    print(response.text)
"""

from omnimembench.llm.client import (
    BaseLLM,
    LLMResponse,
    LocalVLLM,
    OpenAILLM,
    TokenCounter,
    create_llm,
    get_token_counter,
)

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "LocalVLLM",
    "OpenAILLM",
    "TokenCounter",
    "create_llm",
    "get_token_counter",
]
