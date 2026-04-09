"""Notion API 연동."""

from __future__ import annotations

import os
from typing import Any

import requests

NOTION_API_VERSION = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"

_CRITERION_LABELS: dict[str, str] = {
    "clarity": "명확성",
    "constraint": "제약조건",
    "output_format": "출력형식",
    "context": "맥락반영도",
}


def _headers() -> dict[str, str]:
    api_key = os.environ.get("NOTION_API_KEY", "")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Notion-Version": NOTION_API_VERSION,
    }


def _rich_text(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": {"content": content[:2000]}}]


def _heading(level: int, text: str) -> dict[str, Any]:
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": _rich_text(text)}}


def _paragraph(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": _rich_text(text)},
    }


def _code_block(text: str) -> dict[str, Any]:
    return {
        "object": "block",
        "type": "code",
        "code": {"rich_text": _rich_text(text[:2000]), "language": "plain text"},
    }


def _build_blocks(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_name = snapshot.get("prompt_name") or ""
    purpose = snapshot.get("purpose") or ""
    output_format = snapshot.get("output_format") or ""
    goals: list[str] = snapshot.get("improvement_goals") or []
    user_prompt = snapshot.get("user_prompt") or ""
    weighted: dict[str, Any] = snapshot.get("weighted") or {}
    rewrite: dict[str, Any] = snapshot.get("rewrite") or {}
    improved = str(rewrite.get("improved_prompt") or "")
    changes: list[dict[str, Any]] = list(rewrite.get("changes") or [])

    blocks: list[dict[str, Any]] = []

    # 맥락
    blocks.append(_heading(2, "맥락"))
    ctx_lines = [
        f"프롬프트 명: {prompt_name or '(없음)'}",
        f"목적: {purpose or '(없음)'}",
        f"출력 형식: {output_format}",
        f"개선 목적: {', '.join(goals) if goals else '(없음)'}",
    ]
    blocks.append(_paragraph("\n".join(ctx_lines)))

    # 진단 점수
    blocks.append(_heading(2, "진단 결과"))
    scores = weighted.get("weighted_scores") or {}
    score_lines = [
        f"{label}: {scores.get(key, 0)}/25"
        for key, label in _CRITERION_LABELS.items()
    ]
    score_lines.append(
        f"종합: {weighted.get('total_score', 0)}/100"
        f" {weighted.get('grade_badge', '')} {weighted.get('grade', '')}"
    )
    blocks.append(_paragraph("\n".join(score_lines)))

    # Before / After
    blocks.append(_heading(2, "Before"))
    blocks.append(_code_block(user_prompt))
    blocks.append(_heading(2, "After"))
    blocks.append(_code_block(improved))

    # 변경 이유
    if changes:
        blocks.append(_heading(2, "변경 이유"))
        change_lines = [
            (
                f"[{ch.get('criterion', '')}]"
                f" {ch.get('before', '')} → {ch.get('after', '')}"
                f" : {ch.get('reason', '')}"
            )
            for ch in changes
        ]
        blocks.append(_paragraph("\n".join(change_lines)))

    return blocks


def _build_payload(snapshot: dict[str, Any], db_id: str) -> dict[str, Any]:
    prompt_name = snapshot.get("prompt_name") or ""
    purpose = snapshot.get("purpose") or ""
    user_prompt = snapshot.get("user_prompt") or ""
    goals: list[str] = snapshot.get("improvement_goals") or []
    weighted: dict[str, Any] = snapshot.get("weighted") or {}
    rewrite: dict[str, Any] = snapshot.get("rewrite") or {}
    improved = str(rewrite.get("improved_prompt") or "")

    total_score: int = int(weighted.get("total_score") or 0)
    grade: str = str(weighted.get("grade") or "")

    return {
        "parent": {"database_id": db_id},
        "properties": {
            "목적": {"title": _rich_text((prompt_name or purpose)[:2000])},
            "종합점수": {"number": total_score},
            "등급": {"select": {"name": grade}} if grade else {"select": {}},
            "Before": {"rich_text": _rich_text(user_prompt)},
            "After": {"rich_text": _rich_text(improved)},
            "개선목적": {
                "multi_select": [{"name": g} for g in goals if g]
            },
        },
        "children": _build_blocks(snapshot),
    }


def save_diagnosis_page(snapshot: dict[str, Any]) -> str:
    """진단 결과를 Notion 데이터베이스 페이지로 저장합니다.

    Args:
        snapshot: session_state.last_snapshot 과 동일한 구조의 딕셔너리.

    Returns:
        생성된 Notion 페이지 URL.

    Raises:
        RuntimeError: NOTION_API_KEY 또는 NOTION_DB_ID 미설정 시.
        requests.HTTPError: Notion API 호출 실패 시.
    """
    api_key = os.environ.get("NOTION_API_KEY", "")
    db_id = os.environ.get("NOTION_DB_ID", "")
    if not api_key or not db_id:
        raise RuntimeError("NOTION_API_KEY 또는 NOTION_DB_ID가 설정되지 않았습니다.")

    payload = _build_payload(snapshot, db_id)
    resp = requests.post(
        f"{NOTION_BASE_URL}/pages",
        headers=_headers(),
        json=payload,
        timeout=10,
    )
    resp.raise_for_status()
    return str(resp.json().get("url") or "")
