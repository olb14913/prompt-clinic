"""Chain: F-20-1 맥락 모호성 게이팅 + F-20-3 소크라테스 보완 질문 생성."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GateScore(BaseModel):
    goal_ambiguity: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "목표 모호도: 페르소나/대상/결과물 형식이 불명확한 정도 "
            "(0=매우 명확, 1=완전 불명확)"
        ),
    )
    constraint_ambiguity: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "제약 모호도: 글자 수/말투/제외어 등 제약이 없거나 불명확한 정도 "
            "(0=매우 명확, 1=완전 불명확)"
        ),
    )
    success_ambiguity: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "성공기준 모호도: 기대 결과물의 기준이 없거나 불명확한 정도 "
            "(0=매우 명확, 1=완전 불명확)"
        ),
    )
    weak_axes: list[str] = Field(
        description=(
            "모호도 0.6 이상인 축 이름 목록 "
            "(예: ['목표', '제약']). 없으면 빈 리스트."
        ),
    )


class GateQuestions(BaseModel):
    questions: list[str] = Field(
        description=(
            "사용자가 '사용목적' 또는 '개선 포인트'를 보완할 수 있도록 돕는 "
            "질문 1~3개"
        ),
    )


# ──────────────────────────────────────────────────────
# F-20-1: Gate analysis chain
# ──────────────────────────────────────────────────────

_GATE_SYSTEM = """당신은 LLM 프롬프트의 맥락 충분성을 평가하는 전문가입니다.
사용자가 제공한 정보(사용목적, 개선포인트, 진단 프롬프트)를 바탕으로
3개 축의 모호도(0.0~1.0)를 정밀하게 평가하세요.

점수 기준:
- 0.0 : 해당 차원이 매우 명확하게 명시됨
- 0.5 : 부분적으로 명시되어 있으나 보완이 권장됨
- 1.0 : 해당 차원에 대한 정보가 전혀 없거나 완전히 불명확함

평가 기준:
1. 목표 (goal_ambiguity)     : 페르소나, 대상 독자, 최종 결과물의 형식이 구체적으로 명시되었는가
2. 제약 (constraint_ambiguity): 글자 수, 말투, 제외어 등 AI가 지켜야 할 범위와 금지사항이 포함되었는가
3. 성공기준 (success_ambiguity): 정량적 지표나 우수 사례 등 결과물의 기대치가 명확히 표현되었는가

weak_axes: 모호도가 0.6 이상인 항목 이름(목표, 제약, 성공기준 중)만 포함하세요."""

_GATE_HUMAN = """다음 입력의 맥락 모호도를 JSON으로만 출력하세요.

## 사용목적
{purpose}

## 개선포인트
{improvement_goals_text}

## 진단 대상 프롬프트
{user_prompt}

{format_instructions}"""


# ──────────────────────────────────────────────────────
# F-20-3: Question generation chain
# ──────────────────────────────────────────────────────

_QUESTION_SYSTEM = """당신은 프롬프트 작성자가 맥락을 더 명확하게 표현할 수 있도록 돕는 코치입니다.
부족한 맥락 축(목표/제약/성공기준)에 대해 '사용목적' 입력 또는 '개선 포인트' 선택으로
보완할 수 있는 구체적이고 간결한 질문 1~3개를 생성하세요.

질문 원칙:
- 짧고 명확하게 (30자 이내 권장)
- 예/아니오가 아닌 구체적 답변을 유도하는 질문
- '사용목적' 텍스트 보완 또는 '개선 포인트' 항목 선택으로 해결 가능한 것만"""

_QUESTION_HUMAN = """다음 조건에 맞는 보완 질문을 JSON으로만 출력하세요.

## 부족한 맥락 축
{weak_axes_text}

## 현재 사용목적
{purpose}

## 진단 대상 프롬프트 (요약)
{user_prompt_brief}

{format_instructions}"""


def build_gate_chain(llm: ChatOpenAI):
    parser = JsonOutputParser(pydantic_object=GateScore)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _GATE_SYSTEM), ("human", _GATE_HUMAN)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def build_question_chain(llm: ChatOpenAI):
    parser = JsonOutputParser(pydantic_object=GateQuestions)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _QUESTION_SYSTEM), ("human", _QUESTION_HUMAN)]
    ).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser


def compute_gate_total_score(gate_score: dict[str, Any]) -> float:
    """3축 가중 합산: 목표 40% + 제약 30% + 성공기준 30%."""
    g = float(gate_score.get("goal_ambiguity", 0.0))
    c = float(gate_score.get("constraint_ambiguity", 0.0))
    s = float(gate_score.get("success_ambiguity", 0.0))
    return round(0.4 * g + 0.3 * c + 0.3 * s, 4)


def prep_gate_input(
    purpose: str,
    user_prompt: str,
    improvement_goals: list[str],
) -> dict[str, str]:
    goals_text = (
        "\n".join(f"- {g}" for g in improvement_goals)
        if improvement_goals
        else "(선택 없음)"
    )
    return {
        "purpose": purpose or "(없음)",
        "improvement_goals_text": goals_text,
        "user_prompt": user_prompt or "",
    }


def prep_question_input(
    purpose: str,
    user_prompt: str,
    weak_axes: list[str],
) -> dict[str, str]:
    axes_text = ", ".join(weak_axes) if weak_axes else "전반적 맥락"
    brief = (user_prompt[:150] + "...") if len(user_prompt) > 150 else user_prompt
    return {
        "weak_axes_text": axes_text,
        "purpose": purpose or "(없음)",
        "user_prompt_brief": brief,
    }
