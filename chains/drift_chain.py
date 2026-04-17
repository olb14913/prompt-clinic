"""Chain: F-23-2 의도 드리프트 측정 (원본 vs 개선안 3축 보존도 평가)."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class DriftScore(BaseModel):
    goal_preservation: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "목표 보존도: 개선안이 원본의 목적·페르소나·대상을 얼마나 유지하는가 "
            "(0=완전 이탈, 1=완전 보존)"
        ),
    )
    constraint_preservation: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "제약 보존도: 원본에 있던 글자 수·말투·제외어 등 제약이 개선안에서도 유지되는가 "
            "(0=완전 이탈, 1=완전 보존)"
        ),
    )
    structure_similarity: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "구조 유사도: 문장 구조·순서·형식이 원본과 얼마나 유사한가 "
            "(0=완전 다름, 1=동일)"
        ),
    )


_DRIFT_SYSTEM = """당신은 프롬프트 개선 품질을 검수하는 전문가입니다.
원본 프롬프트와 개선된 프롬프트를 비교해 3개 축의 보존도(0.0~1.0)를 평가하세요.

점수 기준:
- 0.0 : 해당 차원이 원본에서 완전히 이탈됨
- 0.5 : 부분적으로 보존됨
- 1.0 : 해당 차원이 원본과 완전히 일치하거나 더 명확하게 보존됨

평가 기준:
1. goal_preservation (50%) : 원본의 목적·페르소나·대상이 개선안에서도 유지되는가
2. constraint_preservation (30%) : 원본에 있던 제약(길이/말투/금지어 등)이 개선안에서도 유지되는가
3. structure_similarity (20%) : 문장 구조·흐름이 원본과 유사한가

원본에 해당 차원이 아예 없었다면(예: 원본에 제약이 없었다면) 보존도는 1.0으로 평가하세요."""

_DRIFT_HUMAN = """다음 두 프롬프트를 비교해 보존도를 JSON으로만 출력하세요.

## 원본 프롬프트
```
{original_prompt}
```

## 개선된 프롬프트
```
{improved_prompt}
```

{format_instructions}"""


def build_drift_chain(llm: ChatOpenAI):
    parser = JsonOutputParser(pydantic_object=DriftScore)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _DRIFT_SYSTEM), ("human", _DRIFT_HUMAN)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def compute_drift_score(result: dict[str, Any]) -> float:
    """가중 드리프트 산출: 1 - (목표50% + 제약30% + 구조20%)."""
    g = float(result.get("goal_preservation", 1.0))
    c = float(result.get("constraint_preservation", 1.0))
    s = float(result.get("structure_similarity", 1.0))
    preservation = 0.5 * g + 0.3 * c + 0.2 * s
    return round(1.0 - preservation, 4)


def prep_drift_input(original_prompt: str, improved_prompt: str) -> dict[str, str]:
    return {
        "original_prompt": original_prompt,
        "improved_prompt": improved_prompt,
    }
