"""점수 기반 LLM 라우팅 유틸."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_openai import ChatOpenAI


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "y", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class RoutingConfig:
    temperature: float
    openai_diagnosis_model: str
    openai_rewrite_model: str
    opus_model: str
    opus_threshold: int
    self_improve_enabled: bool
    self_improve_max_iterations: int
    opus_max_calls: int
    opus_max_tokens: int


def read_routing_config() -> RoutingConfig:
    base_openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    temp = float(os.environ.get("OPENAI_TEMPERATURE", "0.2"))
    return RoutingConfig(
        temperature=temp,
        openai_diagnosis_model=os.environ.get(
            "OPENAI_DIAGNOSIS_MODEL", base_openai_model
        ),
        openai_rewrite_model=os.environ.get(
            "OPENAI_REWRITE_MODEL", base_openai_model
        ),
        opus_model=os.environ.get("ANTHROPIC_MODEL_OPUS", "claude-3-opus-20240229"),
        opus_threshold=max(0, min(100, _env_int("OPUS_SCORE_THRESHOLD", 70))),
        self_improve_enabled=_env_bool("SELF_IMPROVE_ENABLED", False),
        self_improve_max_iterations=max(1, _env_int("SELF_IMPROVE_MAX_ITERS", 3)),
        opus_max_calls=max(0, _env_int("OPUS_MAX_CALLS", 5)),
        opus_max_tokens=max(0, _env_int("OPUS_MAX_TOKENS", 0)),
    )


def make_openai_llm(model_name: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(model=model_name, temperature=temperature)


def build_openai_rewrite_llm(config: RoutingConfig) -> ChatOpenAI:
    return make_openai_llm(config.openai_rewrite_model, config.temperature)


def resolve_rewrite_model_key(
    score: int,
    *,
    has_opus: bool,
    threshold: int,
) -> str:
    if has_opus and score >= threshold:
        return "opus"
    return "openai"


def build_opus_llm(config: RoutingConfig) -> Any | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        from langchain_anthropic import ChatAnthropic
    except Exception:
        return None
    return ChatAnthropic(
        model=config.opus_model,
        temperature=config.temperature,
        anthropic_api_key=api_key,
    )


def model_key_to_label(model_key: str, config: RoutingConfig) -> str:
    if model_key == "opus":
        return f"Claude Opus ({config.opus_model})"
    return f"OpenAI ({config.openai_rewrite_model})"
