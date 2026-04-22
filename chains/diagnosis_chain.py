"""Chain 2: 프롬프트 진단 (Few-shot + CoT)."""

import json
import os
import random
from pathlib import Path
from typing import Any

from utils.vector_store import search_diagnosis

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from utils.notion import load_fewshot_examples_from_notion

_FEWSHOT_PATH = Path(__file__).parent.parent / "data" / "fewshot_examples.json"
_DEFAULT_FEWSHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "label": "낮은 점수 예시",
        "prompt": "이거 좀 해줘",
        "analysis": (
            "목표·형식·제약이 전무함 → 명확성 매우 낮음, "
            "출력 구조 없음, 맥락 없음."
        ),
        "scores": {
            "clarity": "5",
            "constraint": "3",
            "output_format": "4",
            "context": "2",
        },
        "total_hint": "낮음",
        "grade": "개선필요",
    },
    {
        "label": "높은 점수 예시",
        "prompt": (
            "당신은 시니어 데이터 분석가입니다. 아래 CSV 요약에서 "
            "(1) 월별 매출 추이 표, (2) 이상치 3건 이내 bullet, "
            "(3) 권장 액션 2개를 한국어로 작성하세요. 500자 이내."
        ),
        "analysis": "역할·입력·산출 형식·제약이 명시됨.",
        "scores": {
            "clarity": "22",
            "constraint": "20",
            "output_format": "23",
            "context": "21",
        },
        "total_hint": "높음",
        "grade": "우수",
    },
]


class ScoreBlock(BaseModel):
    score: int = Field(ge=0, le=25, description="0~25 점수")
    reason: str = Field(description="단계별 추론(Chain of Thought)을 포함한 문제 원인 설명")


class DiagnosisResult(BaseModel):
    clarity: ScoreBlock = Field(description="명확성")
    constraint: ScoreBlock = Field(description="제약조건")
    output_format: ScoreBlock = Field(description="출력형식")
    context: ScoreBlock = Field(description="맥락반영도")
    total_score: int = Field(ge=0, le=100, description="네 항목 합산(가중 전 기준)")
    grade: str = Field(description="우수/보통/개선필요 중 하나")


# F-25-2: 행위축별 진단 4항목 채점 가중치 테이블 (비율 합계 = 100)
# 키: domain_action 값 / 값: {criterion_key: weight_pct}
DOMAIN_ACTION_WEIGHTS: dict[str, dict[str, int]] = {
    "코드":   {"clarity": 20, "constraint": 40, "output_format": 20, "context": 20},
    "요약":   {"clarity": 20, "constraint": 20, "output_format": 40, "context": 20},
    "글쓰기": {"clarity": 25, "constraint": 20, "output_format": 25, "context": 30},
    "분석":   {"clarity": 30, "constraint": 20, "output_format": 20, "context": 30},
    "QA":    {"clarity": 40, "constraint": 20, "output_format": 20, "context": 20},
}
_DEFAULT_WEIGHTS: dict[str, int] = {
    "clarity": 25, "constraint": 25, "output_format": 25, "context": 25,
}
_CRITERION_KO: dict[str, str] = {
    "clarity": "명확성",
    "constraint": "제약조건",
    "output_format": "출력형식",
    "context": "맥락반영도",
}


def _build_domain_weights_hint(domain_action: str) -> str:
    """행위축별 채점 힌트 문자열 생성."""
    weights = DOMAIN_ACTION_WEIGHTS.get(domain_action, _DEFAULT_WEIGHTS)
    ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{_CRITERION_KO[k]}({v}%)" for k, v in ranked]
    return f"행위축 [{domain_action or '일반'}] 채점 중점 순서: {' > '.join(parts)}"


_EXPERT_SYSTEM_HEADER = """당신은 10년 경력의 프롬프트 엔지니어링 전문가입니다.
사용자의 프롬프트를 명확성/제약조건/출력형식/맥락반영도 기준으로 객관적으로 진단하고,
구체적인 개선안을 제시합니다.
항상 근거를 포함해 설명하며, 학습 효과를 높이는 방식으로 피드백합니다."""

_EXPERT_SYSTEM_FOOTER = """## Chain of Thought 지침
진단 시 내부적으로 단계별로 분석한 뒤, 각 항목의 reason 필드에 그 추론 과정을 요약해 담으세요.
JSON의 total_score는 네 항목 점수의 합과 일치해야 합니다.
grade는 total_score 기준: 80~100 우수, 50~79 보통, 0~49 개선필요."""

DIAGNOSIS_HUMAN = """## 맥락 프로필 (JSON)
{context_profile_json}

## 사용자가 기대하는 출력 형식 (사이드바)
{output_format}

## 개선 목적 (사용자 선택)
{improvement_goals_text}

## 행위축 채점 가중치 힌트
{domain_weights_hint}
{rag_context}
## 진단 대상 프롬프트
```
{user_prompt}
```

위 프롬프트만을 근거로 네 기준을 각각 0~25점으로 채점하고, reason에는 CoT 형태의 원인 분석을 작성하세요.
채점 시 위 행위축 가중치 힌트를 참고해 해당 행위에서 더 중요한 기준을 엄격하게 평가하세요.
{format_instructions}"""


def _get_fewshot_max() -> int:
    """FEWSHOT_MAX 환경변수로 few-shot 최대 개수 제어. 기본 4개."""
    raw = os.environ.get("FEWSHOT_MAX", "").strip()
    if not raw:
        return 4
    try:
        n = int(raw)
    except ValueError:
        return 4
    return max(1, n)


def _sample_fewshot(examples: list[dict[str, Any]], max_n: int) -> list[dict[str, Any]]:
    """등급(grade) 다양성이 섞이도록 예시를 샘플링한다.

    전체 예시 개수가 max_n 이하면 그대로 반환. 그렇지 않으면 등급별 버킷에서
    번갈아가며 최대 max_n개까지 뽑아 반환 순서를 유지한다.
    """
    if len(examples) <= max_n:
        return examples

    buckets: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        grade = str(ex.get("grade") or "기타").strip() or "기타"
        buckets.setdefault(grade, []).append(ex)

    rng = random.Random(42)
    for items in buckets.values():
        rng.shuffle(items)

    picked: list[dict[str, Any]] = []
    grade_keys = list(buckets.keys())
    rng.shuffle(grade_keys)
    idx = 0
    while len(picked) < max_n and any(buckets[g] for g in grade_keys):
        g = grade_keys[idx % len(grade_keys)]
        if buckets[g]:
            picked.append(buckets[g].pop())
        idx += 1
    return picked


def load_fewshot_examples() -> list[dict[str, Any]]:
    """data/fewshot_examples.json 에서 few-shot 예시를 로드한다.

    파일이 매우 크거나 개수가 많으면 TPM 한도를 쉽게 초과하므로, FEWSHOT_MAX
    (기본 4) 개로 등급 다양성 샘플링한 뒤 반환한다.
    """
    max_n = _get_fewshot_max()

    use_notion = os.environ.get("FEWSHOT_SOURCE_NOTION", "").strip().lower()
    if use_notion in {"1", "true", "y", "yes", "on"}:
        notion_examples = load_fewshot_examples_from_notion()
        if notion_examples:
            return _sample_fewshot(notion_examples, max_n)
    if not _FEWSHOT_PATH.exists():
        return _sample_fewshot(_DEFAULT_FEWSHOT_EXAMPLES, max_n)
    try:
        loaded = json.loads(_FEWSHOT_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _sample_fewshot(_DEFAULT_FEWSHOT_EXAMPLES, max_n)
    if not isinstance(loaded, list):
        return _sample_fewshot(_DEFAULT_FEWSHOT_EXAMPLES, max_n)
    normalized = [item for item in loaded if isinstance(item, dict)]
    if not normalized:
        return _sample_fewshot(_DEFAULT_FEWSHOT_EXAMPLES, max_n)
    return _sample_fewshot(normalized, max_n)


def format_fewshot_section(examples: list[dict[str, Any]]) -> str:
    """few-shot 예시 목록을 시스템 프롬프트 텍스트로 변환한다."""
    lines = ["## Few-shot 참고"]
    for ex in examples:
        score_text = ", ".join(f"{k} {v}" for k, v in ex["scores"].items())
        level_text = str(ex.get("level") or "").strip()
        source_text = str(ex.get("source") or "").strip()
        meta_parts: list[str] = []
        if level_text:
            meta_parts.append(f"레벨 {level_text}")
        if source_text:
            meta_parts.append(f"출처 {source_text}")
        meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""
        lines.extend([
            "",
            f"### {ex['label']}{meta_suffix}",
            f"프롬프트: \"{ex['prompt']}\"",
            f"분석: {ex['analysis']}",
            f"점수 예: {score_text}, 합계 {ex['total_hint']} → {ex['grade']}.",
        ])
    return "\n".join(lines)


def build_diagnosis_chain(llm: ChatOpenAI):
    fewshot_section = format_fewshot_section(load_fewshot_examples())
    expert_system = "\n\n".join([_EXPERT_SYSTEM_HEADER, fewshot_section, _EXPERT_SYSTEM_FOOTER])
    # few-shot 예시 본문에 포함된 '{', '}' 가 f-string 템플릿 변수로 오인되지 않도록 이스케이프.
    expert_system_safe = expert_system.replace("{", "{{").replace("}", "}}")

    parser = JsonOutputParser(pydantic_object=DiagnosisResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", expert_system_safe),
            ("human", DIAGNOSIS_HUMAN),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


def _format_rag_diag(results: list[dict[str, Any]]) -> str:
    """RAG 검색 결과를 진단 프롬프트 블록으로 포맷팅. 결과 없으면 ""."""
    if not results:
        return ""
    lines = ["## RAG 참고 사례"]
    for i, item in enumerate(results, 1):
        meta = item.get("metadata") or {}
        domain = str(meta.get("domain_action") or "")
        source = str(meta.get("source") or "")
        tag = " / ".join(part for part in [
            f"domain_action: {domain}" if domain else "",
            f"source: {source}" if source else "",
        ] if part)
        text = str(item.get("text") or "").strip()
        header = f"{i}. ({tag})" if tag else f"{i}."
        lines.append(f"{header}\n{text}")
    return "\n".join(lines) + "\n"


def prep_diagnosis_input(inputs: dict[str, Any]) -> dict[str, Any]:
    profile = inputs.get("context_profile") or {}
    goals = inputs.get("improvement_goals") or []
    domain_action = str(profile.get("domain_action") or "")
    user_prompt = str(inputs.get("user_prompt") or "")
    rag_results = search_diagnosis(user_prompt, domain_action, k=3)
    return {
        "context_profile_json": json.dumps(profile, ensure_ascii=False, indent=2),
        "output_format": inputs.get("output_format") or "",
        "improvement_goals_text": ", ".join(goals) if goals else "(선택 없음)",
        "domain_weights_hint": _build_domain_weights_hint(domain_action),
        "user_prompt": user_prompt,
        "rag_context": _format_rag_diag(rag_results),
    }
