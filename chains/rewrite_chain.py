"""Chain 3: 프롬프트 개선 (Rewrite + CoT)."""

import json
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ChangeItem(BaseModel):
    criterion: str = Field(description="기준명 (명확성/제약조건/출력형식/맥락반영도 등)")
    before: str = Field(description="변경 전 표현")
    after: str = Field(description="변경 후 표현")
    reason: str = Field(description="변경 이유 (근거 포함)")


class RewriteResult(BaseModel):
    improved_prompt: str = Field(description="개선된 프롬프트 전문")
    changes: list[ChangeItem] = Field(description="항목별 변경 사항")


REWRITE_SYSTEM = """당신은 10년 경력의 프롬프트 엔지니어링 전문가입니다.
사용자의 프롬프트를 명확성/제약조건/출력형식/맥락반영도 기준으로 객관적으로 진단하고,
구체적인 개선안을 제시합니다.
항상 근거를 포함해 설명하며, 학습 효과를 높이는 방식으로 피드백합니다.

재작성 시 Chain of Thought로 약점을 짚은 뒤 improved_prompt에 반영하세요.
changes에는 기준별로 before/after/reason을 구체적으로 적으세요."""

REWRITE_HUMAN = """## 맥락 프로필
{context_profile_json}

## 진단 결과 (JSON)
{diagnosis_json}

## 개선 목적 (반드시 반영)
{improvement_goals_text}

## 원본 프롬프트
```
{user_prompt}
```

진단과 맥락·개선 목적을 모두 반영해 프롬프트를 재작성하고, 변경 이유를 changes에 정리하세요.
{format_instructions}"""


def build_rewrite_chain(llm: ChatOpenAI):
    parser = JsonOutputParser(pydantic_object=RewriteResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REWRITE_SYSTEM),
            ("human", REWRITE_HUMAN),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


def prep_rewrite_input(inputs: dict[str, Any]) -> dict[str, Any]:
    profile = inputs.get("context_profile") or {}
    diagnosis = inputs.get("diagnosis") or {}
    goals = inputs.get("improvement_goals") or []
    return {
        "context_profile_json": json.dumps(profile, ensure_ascii=False, indent=2),
        "diagnosis_json": json.dumps(diagnosis, ensure_ascii=False, indent=2),
        "improvement_goals_text": ", ".join(goals) if goals else "(선택 없음)",
        "user_prompt": inputs.get("user_prompt") or "",
    }
