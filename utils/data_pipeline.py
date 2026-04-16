"""학습 데이터 적재 및 few-shot 자동 갱신 유틸."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROMPT_RUNS_PATH = DATA_DIR / "prompt_runs.jsonl"
FEWSHOT_PATH = DATA_DIR / "fewshot_examples.json"

CRITERION_KEYS = ["clarity", "constraint", "output_format", "context"]
CRITERION_LABELS = {
    "clarity": "명확성",
    "constraint": "제약조건",
    "output_format": "출력형식",
    "context": "맥락반영도",
}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


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


def infer_prompt_level(total_score: int) -> dict[str, Any]:
    if total_score >= 80:
        return {"level": 4, "label": "전문가"}
    if total_score >= 60:
        return {"level": 3, "label": "고급"}
    if total_score >= 40:
        return {"level": 2, "label": "중급"}
    return {"level": 1, "label": "초급"}


def _analysis_summary(weighted: dict[str, Any]) -> str:
    scores = weighted.get("weighted_scores") or {}
    reasons = weighted.get("reasons") or {}
    weakest = sorted(CRITERION_KEYS, key=lambda key: _safe_int(scores.get(key)))
    chunks: list[str] = []
    for key in weakest[:2]:
        reason = str(reasons.get(key) or "").strip()
        if not reason:
            continue
        label = CRITERION_LABELS[key]
        chunks.append(f"{label}: {reason[:100]}")
    if not chunks:
        return "자동 수집된 진단 사례입니다."
    return " / ".join(chunks)


def build_run_record(snapshot: dict[str, Any]) -> dict[str, Any]:
    weighted: dict[str, Any] = snapshot.get("weighted") or {}
    score_map = weighted.get("weighted_scores") or {}
    total_score = _safe_int(weighted.get("total_score"))
    score_grade = str(weighted.get("grade") or _score_to_grade(total_score))
    level_info = infer_prompt_level(total_score)
    ts_val = snapshot.get("ts")
    if isinstance(ts_val, datetime):
        ts_text = ts_val.isoformat(timespec="seconds")
    else:
        ts_text = datetime.now().isoformat(timespec="seconds")
    # F-25-3: 행위축·학문축 분류 결과 저장
    domain_result: dict[str, Any] = snapshot.get("domain_result") or {}
    return {
        "ts": ts_text,
        "prompt_name": str(snapshot.get("prompt_name") or ""),
        "purpose": str(snapshot.get("purpose") or ""),
        "output_format": str(snapshot.get("output_format") or ""),
        "improvement_goals": list(snapshot.get("improvement_goals") or []),
        "user_prompt": str(snapshot.get("user_prompt") or ""),
        "improved_prompt": str((snapshot.get("rewrite") or {}).get("improved_prompt") or ""),
        "total_score": total_score,
        "grade": score_grade,
        "scores": {key: str(_safe_int(score_map.get(key))) for key in CRITERION_KEYS},
        "prompt_level": level_info,
        "analysis_summary": _analysis_summary(weighted),
        "domain_action": str(domain_result.get("domain_action") or ""),
        "domain_knowledge": str(domain_result.get("domain_knowledge") or ""),
    }


def append_run_record(record: dict[str, Any], path: Path = PROMPT_RUNS_PATH) -> None:
    _ensure_data_dir()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_ts(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.min


def load_run_records(path: Path = PROMPT_RUNS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                records.append(parsed)
    records.sort(
        key=lambda item: _parse_ts(str(item.get("ts") or "")),
        reverse=True,
    )
    return records


def _level_balanced_pick(
    records: list[dict[str, Any]],
    per_level_limit: int,
    max_examples: int,
    seen_prompts: set[str] | None = None,
) -> list[dict[str, Any]]:
    """레벨 균형 선별 내부 헬퍼."""
    if seen_prompts is None:
        seen_prompts = set()
    grouped: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: [], 4: []}
    for run in records:
        level = _safe_int((run.get("prompt_level") or {}).get("level"))
        if level in grouped:
            grouped[level].append(run)

    selected: list[dict[str, Any]] = []
    for level in [1, 2, 3, 4]:
        picked = 0
        for run in grouped[level]:
            prompt = str(run.get("user_prompt") or "").strip()
            if not prompt or prompt in seen_prompts:
                continue
            selected.append(run)
            seen_prompts.add(prompt)
            picked += 1
            if picked >= per_level_limit:
                break
    return selected[:max_examples]


def _select_records_for_fewshot(
    records: list[dict[str, Any]],
    per_level_limit: int = 2,
    max_examples: int = 8,
    domain_action: str = "",
) -> list[dict[str, Any]]:
    """F-25-3: 동일 행위축 사례 우선 선별 → 부족 시 전체 레벨 균형 fallback."""
    seen_prompts: set[str] = set()

    # 1단계: 동일 행위축 필터링 후 레벨 균형 선별
    if domain_action:
        same_axis = [r for r in records if r.get("domain_action") == domain_action]
        if len(same_axis) >= 3:
            selected = _level_balanced_pick(
                same_axis, per_level_limit, max_examples, seen_prompts
            )
            if len(selected) >= 3:
                return selected[:max_examples]
            # 부족분 보충: fallback 추가 (아래로 계속)
        else:
            selected = _level_balanced_pick(
                same_axis, per_level_limit, max_examples, seen_prompts
            )
    else:
        selected = []

    # 2단계: 전체 레코드에서 fallback (이미 선택된 프롬프트 제외)
    fallback = _level_balanced_pick(
        records, per_level_limit, max_examples - len(selected), seen_prompts
    )
    selected = selected + fallback

    # 3단계: 최소 3개 보장
    if len(selected) < 3:
        for run in records:
            prompt = str(run.get("user_prompt") or "").strip()
            if not prompt or prompt in seen_prompts:
                continue
            selected.append(run)
            seen_prompts.add(prompt)
            if len(selected) >= 3:
                break

    return selected[:max_examples]


def _record_to_fewshot_example(run: dict[str, Any]) -> dict[str, Any]:
    level_info = run.get("prompt_level") or {}
    level = _safe_int(level_info.get("level"))
    total_score = _safe_int(run.get("total_score"))
    grade = str(run.get("grade") or _score_to_grade(total_score))
    prompt = str(run.get("user_prompt") or "")
    analysis_base = str(run.get("analysis_summary") or "").strip()
    analysis = f"총점 {total_score}/100 ({grade}). {analysis_base}".strip()
    scores = run.get("scores") or {}
    return {
        "label": f"레벨 {level} 자동 수집 예시",
        "prompt": prompt,
        "analysis": analysis,
        "scores": {key: str(scores.get(key) or "0") for key in CRITERION_KEYS},
        "total_hint": _score_to_hint(total_score),
        "grade": grade,
        "level": level,
        "source": "auto",
        "collected_at": str(run.get("ts") or ""),
        "domain_action": str(run.get("domain_action") or ""),
    }


def _load_existing_fewshot(path: Path = FEWSHOT_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def refresh_fewshot_examples_from_runs(
    runs_path: Path = PROMPT_RUNS_PATH,
    fewshot_path: Path = FEWSHOT_PATH,
    domain_action: str = "",
) -> bool:
    records = load_run_records(runs_path)
    if not records:
        return False
    selected = _select_records_for_fewshot(records, domain_action=domain_action)
    examples = [_record_to_fewshot_example(run) for run in selected]
    if len(examples) < 3:
        existing = _load_existing_fewshot(fewshot_path)
        seen_prompts = {str(item.get("prompt") or "") for item in examples}
        for item in existing:
            prompt = str(item.get("prompt") or "")
            if not prompt or prompt in seen_prompts:
                continue
            examples.append(item)
            seen_prompts.add(prompt)
            if len(examples) >= 3:
                break
    if not examples:
        return False
    _ensure_data_dir()
    fewshot_path.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def sync_learning_data(snapshot: dict[str, Any]) -> None:
    if not isinstance(snapshot, dict):
        return
    try:
        record = build_run_record(snapshot)
        append_run_record(record)
        # F-25-3: 행위축 기반 few-shot 필터링
        domain_action = str((snapshot.get("domain_result") or {}).get("domain_action") or "")
        refresh_fewshot_examples_from_runs(domain_action=domain_action)
    except Exception:
        # 학습 데이터 적재 실패는 사용자 진단 결과를 막지 않는다.
        return
