"""Notion API 연동."""

from __future__ import annotations

import os
from datetime import datetime, timezone
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

# `main.IMPROVEMENT_OPTIONS`와 동일 문자열 — Notion에 동일 이름의 속성(체크박스/셀렉트 등)이 있으면 매핑
_IMPROVEMENT_GOAL_LABELS: tuple[str, ...] = (
    "출력 품질 높이기",
    "맥락 보완",
    "구조화",
    "일관성 높이기",
    "토큰 줄이기",
)


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


def _norm_prop_key(name: str) -> str:
    return " ".join(str(name).strip().split())


def _resolve_db_prop_name(db_props: dict[str, Any], label: str) -> str | None:
    target = _norm_prop_key(label)
    for key in db_props:
        if _norm_prop_key(key) == target:
            return key
    return None


def _select_active_option_name(prop_meta: dict[str, Any]) -> str:
    """select 속성에서 '선택됨'에 해당할 옵션 이름 (예/Yes 등 우선)."""
    option_names = sorted(_extract_option_names(prop_meta, "select"))
    if not option_names:
        return ""
    positive = {"예", "Yes", "YES", "yes", "Y", "O", "해당", "적용", "선택", "☑"}
    for n in option_names:
        if n in positive:
            return n
    return option_names[0]


def _apply_structured_improvement_goals(
    snapshot: dict[str, Any],
    db_props: dict[str, Any],
    payload_props: dict[str, Any],
) -> bool:
    """Notion DB에 '구조화' 등 목적별 컬럼이 있으면 checkbox/select로 반영. True면 최소 1개 속성에 기록."""
    goals_raw = snapshot.get("improvement_goals") or []
    selected = {_norm_prop_key(str(g)) for g in goals_raw if str(g).strip()}
    any_written = False
    for label in _IMPROVEMENT_GOAL_LABELS:
        key = _resolve_db_prop_name(db_props, label)
        if not key:
            continue
        meta = db_props.get(key)
        if not isinstance(meta, dict):
            continue
        ptype = str(meta.get("type") or "")
        on = _norm_prop_key(label) in selected
        if ptype == "checkbox":
            payload_props[key] = {"checkbox": on}
            any_written = True
        elif ptype == "select":
            if on:
                opt = _select_active_option_name(meta)
                option_names = _extract_option_names(meta, "select")
                if opt and (not option_names or opt in option_names):
                    payload_props[key] = {"select": {"name": opt}}
                else:
                    payload_props[key] = {"select": {}}
            else:
                payload_props[key] = {"select": {}}
            any_written = True
        elif ptype == "multi_select":
            onames = _extract_option_names(meta, "multi_select")
            if on and onames:
                pick = label if label in onames else next(
                    (n for n in onames if label in n or n in label), ""
                )
                if not pick:
                    pick = sorted(onames)[0]
                payload_props[key] = {"multi_select": [{"name": pick}]}
            else:
                payload_props[key] = {"multi_select": []}
            any_written = True
    return any_written


def _changes_summary_only(snapshot: dict[str, Any]) -> str:
    changes: list[dict[str, Any]] = list(
        (snapshot.get("rewrite") or {}).get("changes") or []
    )
    return "\n".join(
        f"[{ch.get('criterion', '')}] {ch.get('before', '')} → {ch.get('after', '')} : {ch.get('reason', '')}"
        for ch in changes
    ).strip()


def _selected_improvement_goals(snapshot: dict[str, Any]) -> list[str]:
    goals_raw = snapshot.get("improvement_goals") or []
    selected = [str(g).strip() for g in goals_raw if str(g).strip()]
    uniq: list[str] = []
    seen: set[str] = set()
    for item in selected:
        if item in seen:
            continue
        seen.add(item)
        uniq.append(item)
    return uniq


def _initial_total_score(snapshot: dict[str, Any]) -> int:
    """최초 진단(개선 전) 종합점수. `main.last_snapshot`의 diagnosis_weighted 사용."""
    diag_w = snapshot.get("diagnosis_weighted")
    if isinstance(diag_w, dict) and diag_w.get("total_score") is not None:
        try:
            return int(diag_w.get("total_score") or 0)
        except (TypeError, ValueError):
            pass
    return 0


def _build_improvement_points_body(snapshot: dict[str, Any]) -> str:
    """Notion 단일 '개선포인트' 텍스트: UI 개선 목적 + 재작성 changes (구조화 속성 없을 때)."""
    goals = [
        str(g).strip()
        for g in (snapshot.get("improvement_goals") or [])
        if str(g).strip()
    ]
    changes_summary = _changes_summary_only(snapshot)
    blocks: list[str] = []
    if goals:
        blocks.append("[선택한 개선 목적]\n" + "\n".join(f"- {g}" for g in goals))
    if changes_summary:
        blocks.append("[재작성 변경 요약]\n" + changes_summary)
    return "\n\n".join(blocks).strip()


def _build_legacy_properties(snapshot: dict[str, Any]) -> dict[str, Any]:
    prompt_name = str(snapshot.get("prompt_name") or "")
    purpose = str(snapshot.get("purpose") or "")
    user_prompt = str(snapshot.get("user_prompt") or "")
    weighted: dict[str, Any] = snapshot.get("weighted") or {}
    rewrite: dict[str, Any] = snapshot.get("rewrite") or {}
    improved = str(rewrite.get("improved_prompt") or "")
    changes: list[dict[str, Any]] = list(rewrite.get("changes") or [])
    total_score = int(weighted.get("total_score") or 0)
    grade = str(weighted.get("grade") or "")
    changes_summary = "\n".join(
        f"[{ch.get('criterion', '')}] {ch.get('before', '')} → {ch.get('after', '')} : {ch.get('reason', '')}"
        for ch in changes
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    props: dict[str, Any] = {
        "프롬프트 명": {"title": _rich_text(prompt_name[:40])},
        "종합점수": {"number": total_score},
        "등급": {"select": {"name": grade}} if grade else {"select": {}},
        "Before": {"rich_text": _rich_text(user_prompt)},
        "After": {"rich_text": _rich_text(improved)},
        "날짜": {"date": {"start": now_iso}},
    }
    if purpose:
        props["프롬프트 사용목적"] = {"rich_text": _rich_text(purpose)}
    improve_body = _build_improvement_points_body(snapshot)
    if improve_body:
        props["개선포인트"] = {"rich_text": _rich_text(improve_body)}
    initial_score = _initial_total_score(snapshot)
    if initial_score > 0:
        props["초기점수"] = {"number": initial_score}
    return props


def _build_properties_by_schema(
    snapshot: dict[str, Any],
    db_props: dict[str, Any],
) -> dict[str, Any]:
    prompt_name = str(snapshot.get("prompt_name") or "")
    purpose = str(snapshot.get("purpose") or "")
    user_prompt = str(snapshot.get("user_prompt") or "")
    weighted: dict[str, Any] = snapshot.get("weighted") or {}
    rewrite: dict[str, Any] = snapshot.get("rewrite") or {}
    improved = str(rewrite.get("improved_prompt") or "")

    total_score: int = int(weighted.get("total_score") or 0)
    grade: str = str(weighted.get("grade") or "")
    title_text = prompt_name[:40]
    now_iso = datetime.now(timezone.utc).isoformat()
    payload_props: dict[str, Any] = {}

    title_match = _find_property_name(
        db_props,
        candidates=["프롬프트 명", "Name", "제목"],
        allowed_types={"title"},
        allow_any_fallback=True,
    )
    if title_match is None:
        raise RuntimeError("Notion DB에 title 타입 컬럼이 없습니다.")
    title_name, title_type = title_match
    _set_text_prop(payload_props, title_name, title_type, title_text)

    usage_match = _find_property_name(
        db_props,
        candidates=["프롬프트 사용목적", "사용목적", "purpose"],
        allowed_types={"rich_text", "title"},
    )
    if usage_match is not None and purpose:
        usage_name, usage_type = usage_match
        if usage_name != title_name:
            _set_text_prop(payload_props, usage_name, usage_type, purpose[:2000])

    score_match = _find_property_name(
        db_props,
        candidates=["종합점수", "총점", "total_score"],
        allowed_types={"number"},
    )
    if score_match is not None:
        payload_props[score_match[0]] = {"number": total_score}

    initial_total = _initial_total_score(snapshot)
    initial_match = _find_property_name(
        db_props,
        candidates=[
            "초기점수",
            "# 초기점수",
            "Initial score",
            "initial_score",
            "initial total score",
        ],
        allowed_types={"number"},
    )
    if initial_match is not None:
        payload_props[initial_match[0]] = {"number": initial_total}

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

    structured_goals_written = _apply_structured_improvement_goals(
        snapshot, db_props, payload_props
    )

    improve_match = _find_property_name(
        db_props,
        candidates=[
            "개선포인트",
            "개선 포인트",
            "# 개선포인트",
            "changes",
            "improvement_points",
        ],
        allowed_types={"rich_text", "title", "multi_select"},
    )
    if improve_match is not None:
        imp_name, imp_type = improve_match
        if imp_type == "multi_select":
            selected_goals = _selected_improvement_goals(snapshot)
            prop_meta = db_props.get(imp_name)
            option_names = (
                _extract_option_names(prop_meta, "multi_select")
                if isinstance(prop_meta, dict)
                else set()
            )
            selected_tags = [
                {"name": goal}
                for goal in selected_goals
                if not option_names or goal in option_names
            ]
            payload_props[imp_name] = {"multi_select": selected_tags}
        else:
            if structured_goals_written:
                improve_body = _changes_summary_only(snapshot)
            else:
                improve_body = _build_improvement_points_body(snapshot)
            if improve_body:
                _set_text_prop(payload_props, imp_name, imp_type, improve_body[:2000])

    date_match = _find_property_name(
        db_props,
        candidates=["날짜", "date", "생성일"],
        allowed_types={"date"},
    )
    if date_match is not None:
        payload_props[date_match[0]] = {"date": {"start": now_iso}}

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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "y", "yes", "on"}


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _score_to_grade(score: int) -> str:
    if score >= 80:
        return "우수"
    if score >= 50:
        return "보통"
    return "개선필요"


def _score_to_hint(score: int) -> str:
    if score >= 80:
        return "높음"
    if score >= 50:
        return "중간"
    return "낮음"


def _join_rich_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    chunks: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        plain = str(item.get("plain_text") or "").strip()
        if plain:
            chunks.append(plain)
            continue
        text_obj = item.get("text")
        if isinstance(text_obj, dict):
            content = str(text_obj.get("content") or "").strip()
            if content:
                chunks.append(content)
    return "".join(chunks).strip()


def _property_text(prop_value: Any) -> str:
    if not isinstance(prop_value, dict):
        return ""
    prop_type = str(prop_value.get("type") or "")
    if prop_type in {"title", "rich_text"}:
        return _join_rich_text(prop_value.get(prop_type))
    if prop_type == "select":
        select_obj = prop_value.get("select")
        if isinstance(select_obj, dict):
            return str(select_obj.get("name") or "").strip()
        return ""
    if prop_type == "multi_select":
        items = prop_value.get("multi_select")
        if not isinstance(items, list):
            return ""
        names = []
        for item in items:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                if name:
                    names.append(name)
        return ", ".join(names)
    if prop_type == "number":
        number_val = prop_value.get("number")
        if number_val is None:
            return ""
        return str(number_val)
    return ""


def _property_number(prop_value: Any) -> int | None:
    if not isinstance(prop_value, dict):
        return None
    prop_type = str(prop_value.get("type") or "")
    if prop_type == "number":
        return _safe_int(prop_value.get("number"))
    if prop_type in {"title", "rich_text", "select"}:
        return _safe_int(_property_text(prop_value))
    return None


def _query_database_pages(db_id: str, page_size: int) -> list[dict[str, Any]]:
    query = {
        "page_size": max(1, min(page_size, 100)),
        "sorts": [{"timestamp": "last_edited_time", "direction": "descending"}],
    }
    resp = requests.post(
        f"{NOTION_BASE_URL}/databases/{db_id}/query",
        headers=_headers(),
        json=query,
        timeout=10,
    )
    resp.raise_for_status()
    body = resp.json()
    results = body.get("results")
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, dict)]


def _pick_property_name(
    db_props: dict[str, Any],
    candidates: list[str],
    allowed_types: set[str],
) -> str | None:
    matched = _find_property_name(
        db_props,
        candidates=candidates,
        allowed_types=allowed_types,
    )
    if matched is None:
        return None
    return matched[0]


def _score_candidates() -> dict[str, list[str]]:
    return {
        "clarity": ["명확성", "clarity"],
        "constraint": ["제약조건", "constraint"],
        "output_format": ["출력형식", "output_format"],
        "context": ["맥락반영도", "context"],
    }


def _infer_level_from_score(score: int) -> int:
    if score >= 80:
        return 4
    if score >= 60:
        return 3
    if score >= 40:
        return 2
    return 1


def _normalize_level(level_text: str, total_score: int) -> int:
    cleaned = level_text.strip().lower()
    if cleaned:
        if cleaned.isdigit():
            parsed = _safe_int(cleaned)
            if parsed is not None:
                return max(1, min(4, parsed))
        if "초급" in cleaned:
            return 1
        if "중급" in cleaned:
            return 2
        if "고급" in cleaned:
            return 3
        if "전문" in cleaned:
            return 4
    return _infer_level_from_score(total_score)


def _select_balanced_examples(
    candidates: list[dict[str, Any]],
    limit: int,
    per_level_limit: int,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: [], 4: []}
    for item in candidates:
        level = _safe_int(item.get("level"))
        if level is None:
            continue
        level = max(1, min(4, level))
        grouped[level].append(item)

    for level in [1, 2, 3, 4]:
        grouped[level].sort(
            key=lambda row: _safe_int(row.get("total_score")) or 0,
            reverse=True,
        )

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for level in [1, 2, 3, 4]:
        picked = 0
        for item in grouped[level]:
            item_key = f"{item.get('page_id')}::{item.get('prompt')}"
            if item_key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(item_key)
            picked += 1
            if picked >= per_level_limit or len(selected) >= limit:
                break
        if len(selected) >= limit:
            break

    if len(selected) >= limit:
        return selected[:limit]

    remaining = sorted(
        candidates,
        key=lambda row: _safe_int(row.get("total_score")) or 0,
        reverse=True,
    )
    for item in remaining:
        item_key = f"{item.get('page_id')}::{item.get('prompt')}"
        if item_key in selected_keys:
            continue
        selected.append(item)
        selected_keys.add(item_key)
        if len(selected) >= limit:
            break
    return selected


def load_fewshot_examples_from_notion(limit: int = 8) -> list[dict[str, Any]]:
    """Notion DB에서 few-shot 예시를 조회해 diagnosis 입력 포맷으로 변환."""
    if not _env_bool("NOTION_FEWSHOT_ENABLED", False):
        return []
    api_key = os.environ.get("NOTION_API_KEY", "")
    db_id = os.environ.get("NOTION_FEWSHOT_DB_ID") or os.environ.get("NOTION_DB_ID", "")
    if not api_key or not db_id:
        return []

    db_props = _fetch_database_properties(db_id)
    pages = _query_database_pages(db_id, page_size=max(10, limit * 2))
    if not pages:
        return []

    prompt_prop = _pick_property_name(
        db_props,
        candidates=["After", "개선 프롬프트", "프롬프트", "Before", "원본 프롬프트", "원본"],
        allowed_types={"rich_text", "title"},
    )
    analysis_prop = _pick_property_name(
        db_props,
        candidates=["분석", "analysis", "요약", "진단요약"],
        allowed_types={"rich_text", "title"},
    )
    grade_prop = _pick_property_name(
        db_props,
        candidates=["등급", "grade"],
        allowed_types={"select", "rich_text", "title"},
    )
    total_prop = _pick_property_name(
        db_props,
        candidates=["종합점수", "총점", "total_score"],
        allowed_types={"number", "rich_text", "title"},
    )
    level_prop = _pick_property_name(
        db_props,
        candidates=["레벨", "prompt_level", "level"],
        allowed_types={"number", "select", "rich_text", "title"},
    )
    score_props: dict[str, str | None] = {}
    for key, score_name_candidates in _score_candidates().items():
        score_props[key] = _pick_property_name(
            db_props,
            candidates=score_name_candidates,
            allowed_types={"number", "rich_text", "title"},
        )

    candidates: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()
    for page in pages:
        props = page.get("properties")
        if not isinstance(props, dict):
            continue
        prompt = _property_text(props.get(prompt_prop)) if prompt_prop else ""
        prompt = prompt.strip()
        if not prompt or prompt in seen_prompts:
            continue

        score_map: dict[str, int] = {}
        for key in ["clarity", "constraint", "output_format", "context"]:
            prop_name = score_props.get(key)
            parsed = _property_number(props.get(prop_name)) if prop_name else None
            score_map[key] = parsed if parsed is not None else 0

        total_score = _property_number(props.get(total_prop)) if total_prop else None
        if total_score is None:
            total_score = sum(score_map.values())
        grade = _property_text(props.get(grade_prop)) if grade_prop else ""
        grade = grade.strip() or _score_to_grade(total_score)

        analysis = _property_text(props.get(analysis_prop)) if analysis_prop else ""
        analysis = analysis.strip()
        if not analysis:
            analysis = f"Notion 수집 사례: 총점 {total_score}/100, 등급 {grade}."

        level_text = _property_text(props.get(level_prop)) if level_prop else ""
        level_num = _normalize_level(level_text, total_score)

        candidates.append({
            "label": f"Notion 자동 수집 예시 (레벨 {level_num})",
            "prompt": prompt,
            "analysis": analysis,
            "scores": {k: str(v) for k, v in score_map.items()},
            "total_hint": _score_to_hint(total_score),
            "grade": grade,
            "level": level_num,
            "total_score": total_score,
            "source": "notion",
            "page_id": str(page.get("id") or ""),
        })
        seen_prompts.add(prompt)
        if len(candidates) >= max(limit * 3, 12):
            break

    if not candidates:
        return []
    per_level_limit = _safe_int(os.environ.get("NOTION_FEWSHOT_PER_LEVEL"))
    if per_level_limit is None:
        per_level_limit = 2
    selected = _select_balanced_examples(
        candidates,
        limit=max(1, limit),
        per_level_limit=max(1, per_level_limit),
    )
    examples = []
    for item in selected:
        examples.append({
            "label": item.get("label", "Notion 자동 수집 예시"),
            "prompt": item.get("prompt", ""),
            "analysis": item.get("analysis", ""),
            "scores": item.get("scores", {}),
            "total_hint": item.get("total_hint", "중간"),
            "grade": item.get("grade", "보통"),
            "level": item.get("level", 0),
            "source": item.get("source", "notion"),
            "page_id": item.get("page_id", ""),
            "total_score": item.get("total_score", 0),
        })
    return examples


def _build_fewshot_properties(
    record: dict[str, Any],
    db_props: dict[str, Any],
) -> dict[str, Any]:
    """F-15-2: build_run_record() 결과를 Notion few-shot DB 프로퍼티로 변환."""
    prompt_name = str(record.get("prompt_name") or "")
    user_prompt = str(record.get("user_prompt") or "")
    improved = str(record.get("improved_prompt") or "")
    total_score = int(record.get("total_score") or 0)
    grade = str(record.get("grade") or "")
    analysis = str(record.get("analysis_summary") or "")
    quality_tag = str(record.get("quality_tag") or "")
    domain_action = str(record.get("domain_action") or "")
    domain_knowledge = str(record.get("domain_knowledge") or "")
    title_text = (prompt_name or user_prompt)[:40]

    payload: dict[str, Any] = {}

    title_match = _find_property_name(
        db_props,
        candidates=["프롬프트 명", "Name", "제목"],
        allowed_types={"title"},
        allow_any_fallback=True,
    )
    if title_match is None:
        return {"title": {"title": _rich_text(title_text)}}
    title_name, title_type = title_match
    _set_text_prop(payload, title_name, title_type, title_text)

    tag_match = _find_property_name(
        db_props,
        candidates=["quality_tag", "태그", "tag"],
        allowed_types={"select"},
    )
    if tag_match and quality_tag:
        payload[tag_match[0]] = {"select": {"name": quality_tag}}

    score_match = _find_property_name(
        db_props,
        candidates=["종합점수", "총점", "total_score"],
        allowed_types={"number"},
    )
    if score_match:
        payload[score_match[0]] = {"number": total_score}

    grade_match = _find_property_name(
        db_props,
        candidates=["등급", "grade"],
        allowed_types={"select"},
    )
    if grade_match and grade:
        grade_name = grade_match[0]
        grade_meta = db_props.get(grade_name)
        option_names = (
            _extract_option_names(grade_meta, "select")
            if isinstance(grade_meta, dict)
            else set()
        )
        if not option_names or grade in option_names:
            payload[grade_name] = {"select": {"name": grade}}

    before_match = _find_property_name(
        db_props,
        candidates=["Before", "원본", "원본 프롬프트"],
        allowed_types={"rich_text"},
    )
    if before_match and user_prompt:
        payload[before_match[0]] = {"rich_text": _rich_text(user_prompt)}

    after_match = _find_property_name(
        db_props,
        candidates=["After", "개선 프롬프트"],
        allowed_types={"rich_text"},
    )
    if after_match and improved:
        payload[after_match[0]] = {"rich_text": _rich_text(improved)}

    analysis_match = _find_property_name(
        db_props,
        candidates=["분석요약", "analysis", "진단요약"],
        allowed_types={"rich_text"},
    )
    if analysis_match and analysis:
        payload[analysis_match[0]] = {"rich_text": _rich_text(analysis)}

    action_match = _find_property_name(
        db_props,
        candidates=["행위축", "domain_action"],
        allowed_types={"rich_text", "select"},
    )
    if action_match and domain_action:
        prop_type = action_match[1]
        if prop_type == "select":
            payload[action_match[0]] = {"select": {"name": domain_action}}
        else:
            payload[action_match[0]] = {"rich_text": _rich_text(domain_action)}

    knowledge_match = _find_property_name(
        db_props,
        candidates=["학문축", "domain_knowledge"],
        allowed_types={"rich_text", "select"},
    )
    if knowledge_match and domain_knowledge:
        prop_type = knowledge_match[1]
        if prop_type == "select":
            payload[knowledge_match[0]] = {"select": {"name": domain_knowledge}}
        else:
            payload[knowledge_match[0]] = {"rich_text": _rich_text(domain_knowledge)}

    return payload


def push_fewshot_record(record: dict[str, Any]) -> bool:
    """F-15-2: quality_tag good/bad 레코드를 Notion few-shot DB에 push.

    NOTION_FEWSHOT_DB_ID 미설정 시 False 반환 (에러 아님).
    API 실패도 False 반환 — 호출측 진단 결과에 영향 없음.
    """
    api_key = os.environ.get("NOTION_API_KEY", "")
    db_id = os.environ.get("NOTION_FEWSHOT_DB_ID", "")
    if not api_key or not db_id:
        return False
    try:
        db_props = _fetch_database_properties(db_id)
        props = _build_fewshot_properties(record, db_props)
        payload = {
            "parent": {"database_id": db_id},
            "properties": props,
        }
        resp = requests.post(
            f"{NOTION_BASE_URL}/pages",
            headers=_headers(),
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception:
        return False


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
