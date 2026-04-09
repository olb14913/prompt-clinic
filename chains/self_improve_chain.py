"""F-09: 자가개선 루프(autoresearch 패턴)."""

from __future__ import annotations

from typing import Any, Callable

from chains.model_router import RoutingConfig, model_key_to_label, resolve_rewrite_model_key

CRITERION_KEYS = ["clarity", "constraint", "output_format", "context"]

GOAL_TO_CRITERIA: dict[str, list[str]] = {
    "토큰 줄이기": ["constraint", "clarity"],
    "일관성 높이기": ["constraint", "clarity"],
    "출력 품질 높이기": ["output_format", "clarity"],
    "구조화": ["output_format", "clarity"],
    "맥락 보완": ["context"],
}


def apply_goal_weights(diagnosis: dict[str, Any], goals: list[str]) -> dict[str, Any]:
    base_scores: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for key in CRITERION_KEYS:
        block = diagnosis.get(key) or {}
        base_scores[key] = int(block.get("score", 0))
        reasons[key] = str(block.get("reason", ""))

    bonus = {k: 0 for k in CRITERION_KEYS}
    for goal in goals:
        for crit in GOAL_TO_CRITERIA.get(goal, []):
            bonus[crit] += 5

    weighted_scores = {
        key: min(25, base_scores[key] + bonus[key])
        for key in CRITERION_KEYS
    }
    total = sum(weighted_scores.values())
    if total >= 80:
        grade, badge = "우수", "🟢"
    elif total >= 50:
        grade, badge = "보통", "🟡"
    else:
        grade, badge = "개선필요", "🔴"
    return {
        "weighted_scores": weighted_scores,
        "base_scores": base_scores,
        "bonus": bonus,
        "total_score": total,
        "grade": grade,
        "grade_badge": badge,
        "reasons": reasons,
    }


def run_self_improve_loop(
    *,
    base_input: dict[str, Any],
    context_profile: dict[str, Any],
    diagnosis_r: Any,
    rewrite_r_openai: Any,
    rewrite_r_opus: Any | None,
    routing: RoutingConfig,
    max_iters: int,
    invoke_with_retry_fn: Callable[..., Any],
    on_iteration: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """개선→재진단 반복 후 최고점 결과를 반환한다."""
    current_prompt = str(base_input.get("user_prompt") or "")
    best_payload: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    prev_score = -1

    for idx in range(max_iters):
        iter_no = idx + 1
        merged_for_diag = {
            **base_input,
            "user_prompt": current_prompt,
            "context_profile": context_profile,
        }
        if on_iteration is not None:
            on_iteration(iter_no, max_iters, "진단")
        diagnosis = invoke_with_retry_fn(diagnosis_r.invoke, merged_for_diag)
        weighted = apply_goal_weights(diagnosis, list(base_input.get("improvement_goals") or []))
        score = int(weighted.get("total_score") or 0)
        model_key = resolve_rewrite_model_key(
            score,
            has_opus=rewrite_r_opus is not None,
            threshold=routing.opus_threshold,
        )
        rewrite_r = rewrite_r_opus if model_key == "opus" else rewrite_r_openai
        if rewrite_r is None:
            rewrite_r = rewrite_r_openai
        if on_iteration is not None:
            model_label = model_key_to_label(model_key, routing)
            on_iteration(iter_no, max_iters, f"개선안 생성 ({model_label})")
        rewrite = invoke_with_retry_fn(
            rewrite_r.invoke,
            {**merged_for_diag, "diagnosis": diagnosis},
        )
        improved = str(rewrite.get("improved_prompt") or "").strip()
        record = {
            "iteration": iter_no,
            "input_prompt": current_prompt,
            "improved_prompt": improved,
            "diagnosis_raw": diagnosis,
            "weighted": weighted,
            "rewrite": rewrite,
            "rewrite_model_key": model_key,
        }
        history.append(record)
        if best_payload is None:
            best_payload = record
        else:
            best_score = int((best_payload.get("weighted") or {}).get("total_score") or 0)
            if score > best_score:
                best_payload = record

        # 점수 향상이 없거나 개선 결과가 비어 있으면 조기 종료
        if score <= prev_score or not improved or improved == current_prompt:
            break
        prev_score = score
        current_prompt = improved

    return {
        "best": best_payload or {},
        "history": history,
    }
