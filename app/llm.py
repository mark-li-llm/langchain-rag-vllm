"""LLM client configuration for vLLM integration.

This module provides ChatOpenAI client configured to connect to vLLM servers
via Nginx load balancer, enabling OpenAI-compatible API access to local models.
"""

import os
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

# ✅ 统一从 Settings 读取（优先级：入参 > settings > 环境变量）
from app.config import settings


def _coalesce(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None


def get_llm(
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = True,
    **kwargs
) -> ChatOpenAI:
    """Create ChatOpenAI client configured for vLLM via Nginx."""
    # --------- 统一取值：入参 > settings > env ----------
    model = _coalesce(model, getattr(settings, "llm_model", None), os.getenv("LLM_MODEL"))
    api_base = _coalesce(api_base, getattr(settings, "openai_api_base", None), os.getenv("OPENAI_API_BASE"))
    api_key  = _coalesce(api_key,  getattr(settings, "openai_api_key",  None), os.getenv("OPENAI_API_KEY"))

    if temperature is None:
        temperature = _coalesce(
            getattr(settings, "temperature", None),
            (lambda s=os.getenv("TEMPERATURE"): float(s) if s else None)()
        )
    if temperature is None:
        temperature = 0.2

    if max_tokens is None:
        max_tokens = _coalesce(
            getattr(settings, "max_output_tokens", None),
            (lambda s=os.getenv("MAX_OUTPUT_TOKENS"): int(s) if s else None)()
        )
    if max_tokens is None:
        max_tokens = 1000

    # --------- 校验 ----------
    if not model:
        raise ValueError("LLM model must be specified (settings.llm_model or LLM_MODEL).")
    if not api_base:
        raise ValueError("API base URL must be specified (settings.openai_api_base or OPENAI_API_BASE).")
    if not api_key:
        raise ValueError("API key must be specified (settings.openai_api_key or OPENAI_API_KEY).")

    print(f"Configuring ChatOpenAI for model '{model}' at '{api_base}'")
    
    # Log all parameters being passed to ChatOpenAI for debugging
    final_params = {
        "model": model,
        "openai_api_base": api_base,
        "openai_api_key": api_key,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "streaming": streaming,
        **kwargs
    }
    print(f"--> ChatOpenAI params: {final_params}")


    llm = ChatOpenAI(**final_params)
    return llm


def get_production_llm() -> ChatOpenAI:
    """Get LLM configured for production use with environment settings."""
    return get_llm(
        streaming=True,
        timeout=60.0,  # Use the correct 'timeout' parameter
    )


def validate_llm_connection(llm: BaseChatModel, test_prompt: str = "Hello") -> Dict[str, Any]:
    """Validate that LLM connection is working."""
    try:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=test_prompt)]
        response = llm.invoke(messages)
        has_content = bool(response.content)
        return {
            "status": "success",
            "response_has_content": has_content,
            "response_preview": str(response.content)[:100] if has_content else "",
            "response_type": type(response).__name__,
            "test_prompt": test_prompt
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__, "test_prompt": test_prompt}


def test_llm_streaming(llm: BaseChatModel, test_prompt: str = "Count to 5") -> Dict[str, Any]:
    """Test LLM streaming capability."""
    try:
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=test_prompt)]
        chunks = []
        for chunk in llm.stream(messages):
            if chunk.content:
                chunks.append(chunk.content)
        total_content = "".join(chunks)
        return {
            "status": "success",
            "chunks_received": len(chunks),
            "total_content_length": len(total_content),
            "content_preview": total_content[:100],
            "streaming_working": len(chunks) > 1,
            "test_prompt": test_prompt
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__, "test_prompt": test_prompt}


def get_llm_info(llm: ChatOpenAI) -> Dict[str, Any]:
    """Get information about LLM configuration."""
    return {
        "model_name": llm.model_name,
        "openai_api_base": llm.openai_api_base,
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "streaming": llm.streaming,
        "request_timeout": getattr(llm, 'request_timeout', None),
        "max_retries": getattr(llm, 'max_retries', None),
    }


def create_llm_with_fallback(
    primary_config: Dict[str, Any],
    fallback_config: Optional[Dict[str, Any]] = None
) -> BaseChatModel:
    """Create LLM with optional fallback configuration."""
    try:
        llm = get_llm(**primary_config)
        validation = validate_llm_connection(llm)
        if validation["status"] == "success":
            print("Primary LLM configuration successful")
            return llm
        else:
            print(f"Primary LLM validation failed: {validation.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"Primary LLM configuration failed: {e}")

    if fallback_config:
        try:
            print("Trying fallback LLM configuration...")
            llm = get_llm(**fallback_config)
            validation = validate_llm_connection(llm)
            if validation["status"] == "success":
                print("Fallback LLM configuration successful")
                return llm
            else:
                print(f"Fallback LLM validation failed: {validation.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Fallback LLM configuration failed: {e}")

    raise RuntimeError("Both primary and fallback LLM configurations failed")
