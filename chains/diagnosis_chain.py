"""Chain 2: 프롬프트 진단 (Few-shot + CoT)."""

import json
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


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


EXPERT_SYSTEM = """당신은 10년 경력의 프롬프트 엔지니어링 전문가입니다.
사용자의 프롬프트를 명확성/제약조건/출력형식/맥락반영도 기준으로 객관적으로 진단하고,
구체적인 개선안을 제시합니다.
항상 근거를 포함해 설명하며, 학습 효과를 높이는 방식으로 피드백합니다.

## Few-shot 참고

### 낮은 점수 예시
프롬프트: "이거 좀 해줘"
분석: 목표·형식·제약이 전무함 → 명확성 매우 낮음, 출력 구조 없음, 맥락 없음.
점수 예: clarity 5, constraint 3, output_format 4, context 2, 합계 낮음 → 개선필요.

### 높은 점수 예시
프롬프트: "당신은 시니어 데이터 분석가입니다. 아래 CSV 요약에서 (1) 월별 매출 추이 표, (2) 이상치 3건 이내 bullet, (3) 권장 액션 2개를 한국어로 작성하세요. 500자 이내."
분석: 역할·입력·산출 형식·제약이 명시됨.
점수 예: clarity 22+, constraint 20+, output_format 23+, context 21+, 합계 높음 → 우수.

## Chain of Thought 지침
진단 시 내부적으로 단계별로 분석한 뒤, 각 항목의 reason 필드에 그 추론 과정을 요약해 담으세요.
JSON의 total_score는 네 항목 점수의 합과 일치해야 합니다.
grade는 total_score 기준: 80~100 우수, 50~79 보통, 0~49 개선필요."""

DIAGNOSIS_HUMAN = """## 맥락 프로필 (JSON)
{context_profile_json}

## 사용자가 기대하는 출력 형식 (사이드바)
{output_format}

## 개선 목적 (사용자 선택)
{improvement_goals_text}

## 진단 대상 프롬프트
```
{user_prompt}
```

위 프롬프트만을 근거로 네 기준을 각각 0~25점으로 채점하고, reason에는 CoT 형태의 원인 분석을 작성하세요.
{format_instructions}"""


def build_diagnosis_chain(llm: ChatOpenAI):
    parser = JsonOutputParser(pydantic_object=DiagnosisResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", EXPERT_SYSTEM),
            ("human", DIAGNOSIS_HUMAN),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


def prep_diagnosis_input(inputs: dict[str, Any]) -> dict[str, Any]:
    profile = inputs.get("context_profile") or {}
    goals = inputs.get("improvement_goals") or []
    return {
        "context_profile_json": json.dumps(profile, ensure_ascii=False, indent=2),
        "output_format": inputs.get("output_format") or "",
        "improvement_goals_text": ", ".join(goals) if goals else "(선택 없음)",
        "user_prompt": inputs.get("user_prompt") or "",
    }
