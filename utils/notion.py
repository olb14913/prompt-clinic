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


def _find_property_name(
    db_props: dict[str, Any],
    candidates: list[str],
    allowed_types: set[str],
    *,
    allow_any_fallback: bool = False,
) -> tuple[str, str] | None:
    for candidate in candidates:
        prop = db_props.get(candidate)
        if isinstance(prop, dict):
            prop_type = str(prop.get("type") or "")
            if prop_type in allowed_types:
                return candidate, prop_type
    if allow_any_fallback:
        for name, prop in db_props.items():
            if isinstance(prop, dict):
                prop_type = str(prop.get("type") or "")
                if prop_type in allowed_types:
                    return name, prop_type
    return None


def _set_text_prop(
    payload_props: dict[str, Any],
    prop_name: str,
    prop_type: str,
    text: str,
) -> None:
    if prop_type == "title":
        payload_props[prop_name] = {"title": _rich_text(text)}
        return
    payload_props[prop_name] = {"rich_text": _rich_text(text)}


def _extract_option_names(prop_meta: dict[str, Any], key: str) -> set[str]:
    opt_meta = prop_meta.get(key)
    if not isinstance(opt_meta, dict):
        return set()
    options = opt_meta.get("options")
    if not isinstance(options, list):
        return set()
    names: set[str] = set()
    for opt in options:
        if isinstance(opt, dict):
            name = str(opt.get("name") or "").strip()
            if name:
                names.add(name)
    return names


def _build_legacy_properties(snapshot: dict[str, Any]) -> dict[str, Any]:
    prompt_name = str(snapshot.get("prompt_name") or "")
    purpose = str(snapshot.get("purpose") or "")
    user_prompt = str(snapshot.get("user_prompt") or "")
    goals: list[str] = snapshot.get("improvement_goals") or []
    weighted: dict[str, Any] = snapshot.get("weighted") or {}
    rewrite: dict[str, Any] = snapshot.get("rewrite") or {}
    improved = str(rewrite.get("improved_prompt") or "")
    total_score = int(weighted.get("total_score") or 0)
    grade = str(weighted.get("grade") or "")
    return {
        "목적": {"title": _rich_text((prompt_name or purpose)[:2000])},
        "종합점수": {"number": total_score},
        "등급": {"select": {"name": grade}} if grade else {"select": {}},
        "Before": {"rich_text": _rich_text(user_prompt)},
        "After": {"rich_text": _rich_text(improved)},
        "개선목적": {"multi_select": [{"name": g} for g in goals if g]},
    }


def _build_properties_by_schema(
    snapshot: dict[str, Any],
    db_props: dict[str, Any],
) -> dict[str, Any]:
    prompt_name = snapshot.get("prompt_name") or ""
    purpose = snapshot.get("purpose") or ""
    output_format = snapshot.get("output_format") or ""
    user_prompt = snapshot.get("user_prompt") or ""
    goals: list[str] = snapshot.get("improvement_goals") or []
    weighted: dict[str, Any] = snapshot.get("weighted") or {}
    rewrite: dict[str, Any] = snapshot.get("rewrite") or {}
    improved = str(rewrite.get("improved_prompt") or "")

    total_score: int = int(weighted.get("total_score") or 0)
    grade: str = str(weighted.get("grade") or "")
    title_text = str(prompt_name or purpose or "prompt_clinic")
    usage_purpose = str(purpose or "")
    payload_props: dict[str, Any] = {}

    title_match = _find_property_name(
        db_props,
        candidates=["프롬프트 명", "목적", "Name", "제목"],
        allowed_types={"title"},
        allow_any_fallback=True,
    )
    if title_match is None:
        raise RuntimeError("Notion DB에 title 타입 컬럼이 없습니다.")
    title_name, title_type = title_match
    _set_text_prop(payload_props, title_name, title_type, title_text[:2000])

    usage_match = _find_property_name(
        db_props,
        candidates=["프롬프트 사용목적", "사용목적", "purpose"],
        allowed_types={"rich_text", "title"},
    )
    if usage_match is not None and usage_purpose:
        usage_name, usage_type = usage_match
        if usage_name != title_name:
            _set_text_prop(payload_props, usage_name, usage_type, usage_purpose[:2000])

    score_match = _find_property_name(
        db_props,
        candidates=["종합점수", "총점", "total_score"],
        allowed_types={"number"},
    )
    if score_match is not None:
        payload_props[score_match[0]] = {"number": total_score}

    grade_match = _find_property_name(
        db_props,
        candidates=["등급", "grade"],
        allowed_types={"select"},
    )
    if grade_match is not None:
        grade_name = grade_match[0]
        grade_meta = db_props.get(grade_name)
        if isinstance(grade_meta, dict):
            option_names = _extract_option_names(grade_meta, "select")
            if not grade:
                payload_props[grade_name] = {"select": {}}
            elif not option_names or grade in option_names:
                payload_props[grade_name] = {"select": {"name": grade}}

    before_match = _find_property_name(
        db_props,
        candidates=["Before", "원본 프롬프트", "원본"],
        allowed_types={"rich_text"},
    )
    if before_match is not None and user_prompt:
        payload_props[before_match[0]] = {"rich_text": _rich_text(user_prompt)}

    after_match = _find_property_name(
        db_props,
        candidates=["After", "개선 프롬프트", "개선"],
        allowed_types={"rich_text"},
    )
    if after_match is not None and improved:
        payload_props[after_match[0]] = {"rich_text": _rich_text(improved)}

    goals_match = _find_property_name(
        db_props,
        candidates=["개선목적", "개선 목적", "improvement_goals"],
        allowed_types={"multi_select"},
    )
    if goals_match is not None and goals:
        goals_name = goals_match[0]
        goals_meta = db_props.get(goals_name)
        if isinstance(goals_meta, dict):
            option_names = _extract_option_names(goals_meta, "multi_select")
            if option_names:
                valid_goals = [g for g in goals if g and g in option_names]
            else:
                valid_goals = [g for g in goals if g]
            if valid_goals:
                payload_props[goals_name] = {
                    "multi_select": [{"name": g} for g in valid_goals]
                }

    output_match = _find_property_name(
        db_props,
        candidates=["출력형식", "출력 형식", "output_format"],
        allowed_types={"rich_text", "select"},
    )
    if output_match is not None and output_format:
        output_name, output_type = output_match
        if output_type == "select":
            output_meta = db_props.get(output_name)
            if isinstance(output_meta, dict):
                option_names = _extract_option_names(output_meta, "select")
                if not option_names or output_format in option_names:
                    payload_props[output_name] = {"select": {"name": output_format}}
        else:
            payload_props[output_name] = {"rich_text": _rich_text(output_format)}

    return payload_props


def _fetch_database_properties(db_id: str) -> dict[str, Any]:
    resp = requests.get(
        f"{NOTION_BASE_URL}/databases/{db_id}",
        headers=_headers(),
        timeout=10,
    )
    resp.raise_for_status()
    body = resp.json()
    props = body.get("properties")
    if isinstance(props, dict):
        return props
    return {}


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

    try:
        db_props = _fetch_database_properties(db_id)
        mapped_props = _build_properties_by_schema(snapshot, db_props)
    except Exception:
        # DB 스키마 조회 실패 시 기존 매핑으로 fallback
        mapped_props = _build_legacy_properties(snapshot)

    payload = {
        "parent": {"database_id": db_id},
        "properties": mapped_props,
        "children": _build_blocks(snapshot),
    }
    resp = requests.post(
        f"{NOTION_BASE_URL}/pages",
        headers=_headers(),
        json=payload,
        timeout=10,
    )
    resp.raise_for_status()
    return str(resp.json().get("url") or "")
