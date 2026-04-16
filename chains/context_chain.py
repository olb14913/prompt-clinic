"""Chain 1: 맥락 프로필 생성 (Context Discovery)."""

from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ContextProfile(BaseModel):
    """맥락 프로필 JSON 스키마."""

    purpose: str = Field(description="정제된 목적 요약")
    output_format: str = Field(description="출력형식")
    improvement_goals: list[str] = Field(description="선택된 개선 목적 리스트")
    context_summary: str = Field(description="진단에 활용할 맥락 한 줄 요약")
    # F-25-1: 도메인 2축 분류 결과
    domain_action: str = Field(
        description="행위축 분류 결과: 코드 / 요약 / 글쓰기 / 분석 / QA 중 하나"
    )
    domain_knowledge: str = Field(
        description="학문축 분류 결과: 의학 / 법률 / 코딩 / 디자인 / 마케팅 / 과학 / 일반 중 하나"
    )
    confidence_action: float = Field(
        ge=0.0, le=1.0, description="행위축 분류 신뢰도 (0.0~1.0)"
    )
    confidence_knowledge: float = Field(
        ge=0.0, le=1.0, description="학문축 분류 신뢰도 (0.0~1.0)"
    )


CONTEXT_SYSTEM = """당신은 사용자의 입력을 구조화해 프롬프트 진단에 쓰일 맥락 프로필을 만듭니다.
목적을 한 문장으로 정제하고, 출력 형식과 개선 목표를 반영한 요약을 제공합니다.

또한 진단 대상 프롬프트를 아래 2가지 축으로 동시에 분류하세요.

행위축 (5종 중 1개 선택): 코드 / 요약 / 글쓰기 / 분석 / QA
학문축 (7종 중 1개 선택): 의학 / 법률 / 코딩 / 디자인 / 마케팅 / 과학 / 일반

각 축의 분류 결과와 신뢰도(0.0~1.0)를 반환하세요.
신뢰도는 분류 결과에 대한 확신 정도를 나타냅니다 (1.0에 가까울수록 확실)."""

CONTEXT_HUMAN = """다음 입력을 바탕으로 맥락 프로필을 JSON으로만 출력하세요.

## 목적 (자유 텍스트)
{purpose}

## 출력 형식
{output_format}

## 개선 목적 (선택됨)
{improvement_goals_text}

## 진단 대상 프롬프트
{user_prompt}

{format_instructions}"""


def build_context_chain(llm: ChatOpenAI):
    parser = JsonOutputParser(pydantic_object=ContextProfile)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXT_SYSTEM),
            ("human", CONTEXT_HUMAN),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


def format_improvement_goals(goals: list[str]) -> str:
    if not goals:
        return "(선택 없음)"
    return "\n".join(f"- {g}" for g in goals)


def prep_context_input(inputs: dict[str, Any]) -> dict[str, Any]:
    goals = inputs.get("improvement_goals") or []
    return {
        "purpose": inputs.get("purpose") or "",
        "output_format": inputs.get("output_format") or "",
        "improvement_goals_text": format_improvement_goals(goals),
        "user_prompt": inputs.get("user_prompt") or "",
    }
