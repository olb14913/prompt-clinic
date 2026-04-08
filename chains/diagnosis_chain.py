"""Chain 2: 프롬프트 진단 (Few-shot + CoT)."""

import json
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

_FEWSHOT_PATH = Path(__file__).parent.parent / "data" / "fewshot_examples.json"


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

## 진단 대상 프롬프트
```
{user_prompt}
```

위 프롬프트만을 근거로 네 기준을 각각 0~25점으로 채점하고, reason에는 CoT 형태의 원인 분석을 작성하세요.
{format_instructions}"""


def load_fewshot_examples() -> list[dict[str, Any]]:
    """data/fewshot_examples.json 에서 few-shot 예시를 로드한다."""
    with _FEWSHOT_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def format_fewshot_section(examples: list[dict[str, Any]]) -> str:
    """few-shot 예시 목록을 시스템 프롬프트 텍스트로 변환한다."""
    lines = ["## Few-shot 참고"]
    for ex in examples:
        score_text = ", ".join(f"{k} {v}" for k, v in ex["scores"].items())
        lines.extend([
            "",
            f"### {ex['label']}",
            f"프롬프트: \"{ex['prompt']}\"",
            f"분석: {ex['analysis']}",
            f"점수 예: {score_text}, 합계 {ex['total_hint']} → {ex['grade']}.",
        ])
    return "\n".join(lines)


def build_diagnosis_chain(llm: ChatOpenAI):
    fewshot_section = format_fewshot_section(load_fewshot_examples())
    expert_system = "\n\n".join([_EXPERT_SYSTEM_HEADER, fewshot_section, _EXPERT_SYSTEM_FOOTER])

    parser = JsonOutputParser(pydantic_object=DiagnosisResult)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", expert_system),
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
