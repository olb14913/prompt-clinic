"""행위축/학문축 키워드 기반 분류 (수집 스크립트 공용 모듈).

사용자 요청 스펙 그대로의 키워드 테이블을 유지한다.
- 행위축: 코드/요약/글쓰기/분석/QA (무일치 시 "")
- 학문축: 의학/법률/코딩/디자인/마케팅/과학/일반 (무일치 시 "일반")
우선순위는 테이블 순서를 따른다 (상위 먼저 매칭).
"""

from __future__ import annotations

from typing import Iterable

# 행위축 키워드 테이블 (순서 = 우선순위)
ACTION_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("코드", ("code", "coding", "python", "javascript", "program", "script", "debug")),
    ("요약", ("summarize", "summary", "tldr", "brief", "shorten")),
    ("글쓰기", ("write", "blog", "essay", "article", "story", "creative", "email")),
    ("분석", ("analyze", "analysis", "review", "evaluate", "compare", "assess")),
    ("QA", ("answer", "question", "faq", "explain", "help", "assist")),
]

# 학문축 키워드 테이블 (순서 = 우선순위)
KNOWLEDGE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("의학", ("medical", "health", "doctor", "disease", "symptom", "treatment")),
    ("법률", ("legal", "law", "contract", "attorney", "regulation", "policy")),
    ("코딩", ("code", "programming", "developer", "software", "algorithm")),
    ("디자인", ("design", "ui", "ux", "visual", "graphic", "layout")),
    ("마케팅", ("marketing", "sales", "brand", "campaign", "customer", "product")),
    ("과학", ("science", "research", "physics", "chemistry", "biology", "data")),
]

DEFAULT_ACTION = ""
DEFAULT_KNOWLEDGE = "일반"


def _match_first(text_lower: str, table: Iterable[tuple[str, tuple[str, ...]]]) -> str:
    for label, keywords in table:
        for kw in keywords:
            if kw in text_lower:
                return label
    return ""


def classify_prompt(text: str) -> tuple[str, str]:
    """프롬프트 문자열을 (domain_action, domain_knowledge)로 분류."""
    if not text:
        return DEFAULT_ACTION, DEFAULT_KNOWLEDGE
    lowered = text.lower()
    action = _match_first(lowered, ACTION_KEYWORDS) or DEFAULT_ACTION
    knowledge = _match_first(lowered, KNOWLEDGE_KEYWORDS) or DEFAULT_KNOWLEDGE
    return action, knowledge


def build_collected_record(prompt: str) -> dict[str, str]:
    """수집 JSONL 한 줄 레코드 포맷. quality_tag는 고정 'good'."""
    action, knowledge = classify_prompt(prompt)
    return {
        "prompt": prompt,
        "domain_action": action,
        "domain_knowledge": knowledge,
        "quality_tag": "good",
    }
