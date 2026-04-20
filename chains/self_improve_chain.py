"""F-09: 자가개선 루프(autoresearch 패턴)."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from chains.model_router import RoutingConfig, model_key_to_label

CRITERION_KEYS = ["clarity", "constraint", "output_format", "context"]

# ── Phase 루프 상수 ──────────────────────────────────────────────────────────
_PHASE1_MAX_ITERS = 3    # Phase 1 OpenAI 최대 반복 횟수
_PHASE2_EXTRA_ITERS = 2  # Phase 2 검증 후 추가 루프 (총 Opus = 1 + 2 = 3)

# ── F-22-1: 정체 패턴 상수 ──────────────────────────────────────────────────
STAGNATION_SPINNING = "SPINNING"           # 동일 출력 반복
STAGNATION_OSCILLATION = "OSCILLATION"    # A→B→A 교번
STAGNATION_NO_DRIFT = "NO_DRIFT"          # 점수 변화 미미
STAGNATION_DIMINISHING = "DIMINISHING_RETURNS"  # 개선율 지속 감소

# ── F-22-2: 페르소나 상수 ───────────────────────────────────────────────────
PERSONA_HACKER = "HACKER"
PERSONA_RESEARCHER = "RESEARCHER"
PERSONA_SIMPLIFIER = "SIMPLIFIER"
PERSONA_ARCHITECT = "ARCHITECT"
PERSONA_CONTRARIAN = "CONTRARIAN"

_PATTERN_PERSONA_MAP: dict[str, str] = {
    STAGNATION_SPINNING: PERSONA_HACKER,
    STAGNATION_NO_DRIFT: PERSONA_RESEARCHER,
    STAGNATION_OSCILLATION: PERSONA_ARCHITECT,
    STAGNATION_DIMINISHING: PERSONA_SIMPLIFIER,
}

# ── F-22-3: 페르소나별 재작성 지시문 ────────────────────────────────────────
_PERSONA_INSTRUCTIONS: dict[str, str] = {
    PERSONA_HACKER: (
        "기존 접근 방식을 완전히 무시하고 우회로를 찾으세요. "
        "관습적인 표현 대신 새로운 구조와 표현을 시도하세요."
    ),
    PERSONA_RESEARCHER: (
        "추가 맥락과 더 깊은 의미를 탐색하세요. "
        "배경 정보, 전제 조건, 구체적인 예시를 적극적으로 포함시키세요."
    ),
    PERSONA_SIMPLIFIER: (
        "불필요한 모든 요소를 제거하고 핵심만 남기세요. "
        "가장 간결하고 명확한 표현으로 재구성하세요."
    ),
    PERSONA_ARCHITECT: (
        "전체 구조를 재설계하세요. "
        "현재의 논리적 흐름을 버리고 더 체계적인 구성으로 재건하세요."
    ),
    PERSONA_CONTRARIAN: (
        "모든 가정을 뒤집고 반대 방향으로 접근하세요. "
        "현재 방식의 약점을 역이용해 완전히 다른 관점으로 재작성하세요."
    ),
}

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


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_stagnation_pattern(history: list[dict[str, Any]]) -> str | None:
    """F-22-1: 자가개선 이력에서 정체 패턴을 감지한다.

    반환값: 패턴 상수 문자열 또는 None (정체 없음)
    """
    if len(history) < 2:
        return None

    prompts = [str(r.get("improved_prompt") or "") for r in history]
    scores = [
        int((r.get("weighted") or {}).get("total_score") or 0) for r in history
    ]

    # SPINNING: 최근 2개 이상의 출력 프롬프트 해시 동일
    if _prompt_hash(prompts[-1]) == _prompt_hash(prompts[-2]):
        return STAGNATION_SPINNING

    # OSCILLATION: 현재 프롬프트가 이전 히스토리에 이미 등장
    current_hash = _prompt_hash(prompts[-1])
    if any(_prompt_hash(p) == current_hash for p in prompts[:-1]):
        return STAGNATION_OSCILLATION

    # NO_DRIFT: 최근 2회 점수 변화가 epsilon(1점) 미만
    if abs(scores[-1] - scores[-2]) < 1:
        return STAGNATION_NO_DRIFT

    # DIMINISHING_RETURNS: 최근 3회 개선량이 연속 감소
    if len(scores) >= 3:
        delta_prev = scores[-2] - scores[-3]
        delta_curr = scores[-1] - scores[-2]
        if delta_curr < delta_prev and delta_curr <= 0:
            return STAGNATION_DIMINISHING

    return None


def select_persona_for_pattern(pattern: str) -> str:
    """F-22-2: 정체 패턴에 대응하는 페르소나를 반환한다."""
    return _PATTERN_PERSONA_MAP.get(pattern, PERSONA_CONTRARIAN)


def get_persona_instruction(persona: str) -> str:
    """F-22-3: 페르소나 이름으로 재작성 지시문을 반환한다."""
    return _PERSONA_INSTRUCTIONS.get(persona, "")


def _select_best_iteration(history: list[dict[str, Any]]) -> dict[str, Any]:
    """F-13-1: 이력에서 최적 이터레이션을 선택한다.

    선택 기준:
    1. total_score 내림차순 (높을수록 우선)
    2. 동점 시 iteration 내림차순 (나중 이터레이션 = 더 정제된 결과)
    """
    if not history:
        return {}
    return max(
        history,
        key=lambda r: (
            int((r.get("weighted") or {}).get("total_score") or 0),
            int(r.get("iteration") or 0),
        ),
    )


def _run_phase1_openai_loop(
    *,
    base_input: dict[str, Any],
    context_profile: dict[str, Any],
    diagnosis_r: Any,
    rewrite_r_openai: Any,
    routing: RoutingConfig,
    invoke_with_retry_fn: Callable[..., Any],
    on_iteration: Callable[[int, int, str, int], None] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Phase 1: OpenAI 전용 루프. 반환: (history, phase1_best_score)"""
    current_prompt = str(base_input.get("user_prompt") or "")
    history: list[dict[str, Any]] = []
    prev_score = -1
    phase1_best_score = 0
    label_openai = model_key_to_label("openai", routing)

    for idx in range(_PHASE1_MAX_ITERS):
        iter_no = idx + 1
        merged = {
            **base_input,
            "user_prompt": current_prompt,
            "context_profile": context_profile,
        }
        if on_iteration is not None:
            on_iteration(iter_no, _PHASE1_MAX_ITERS, "진단", 0)
        diagnosis = invoke_with_retry_fn(diagnosis_r.invoke, merged)
        weighted = apply_goal_weights(
            diagnosis, list(base_input.get("improvement_goals") or [])
        )
        score = int(weighted.get("total_score") or 0)

        if on_iteration is not None:
            on_iteration(iter_no, _PHASE1_MAX_ITERS, f"개선안 생성 ({label_openai})", score)
        rewrite = invoke_with_retry_fn(
            rewrite_r_openai.invoke,
            {**merged, "diagnosis": diagnosis},
        )
        improved = str(rewrite.get("improved_prompt") or "").strip()
        history.append({
            "iteration": iter_no,
            "input_prompt": current_prompt,
            "improved_prompt": improved,
            "diagnosis_raw": diagnosis,
            "weighted": weighted,
            "rewrite": rewrite,
            "rewrite_model_key": "openai",
            "strategy": "phase1",
            "persona": "",
        })
        phase1_best_score = max(phase1_best_score, score)

        if score == 100:
            break
        stagnant = score <= prev_score or not improved or improved == current_prompt
        if stagnant:
            break
        prev_score = score
        current_prompt = improved

    return history, phase1_best_score


def _run_phase2_opus_loop(
    *,
    history: list[dict[str, Any]],
    phase1_best_score: int,
    base_input: dict[str, Any],
    context_profile: dict[str, Any],
    diagnosis_r: Any,
    rewrite_r_opus: Any,
    routing: RoutingConfig,
    invoke_with_retry_fn: Callable[..., Any],
    on_iteration: Callable[[int, int, str, int], None] | None,
) -> list[dict[str, Any]]:
    """Phase 2: Opus 교차 검증 1회 + 조건부 추가 루프. 반환: 업데이트된 history."""
    result = list(history)
    label_opus = model_key_to_label("opus", routing)
    total_p2 = _PHASE1_MAX_ITERS + 1 + _PHASE2_EXTRA_ITERS

    # Phase 1 best 결과에서 입력 프롬프트·진단 추출
    best_p1 = _select_best_iteration(result)
    p1_prompt = str(best_p1.get("improved_prompt") or "").strip()
    if not p1_prompt:
        p1_prompt = str(base_input.get("user_prompt") or "")
    p1_diag = best_p1.get("diagnosis_raw") or {}

    # ── 2a: Opus 교차 검증 (1 call) ─────────────────────────────────────────
    iter_val = len(result) + 1
    merged_p1 = {
        **base_input,
        "user_prompt": p1_prompt,
        "context_profile": context_profile,
    }
    if on_iteration is not None:
        on_iteration(iter_val, total_p2, f"Opus 교차 검증 ({label_opus})", 0)
    try:
        rewrite_val = invoke_with_retry_fn(
            rewrite_r_opus.invoke,
            {**merged_p1, "diagnosis": p1_diag},
        )
        improved_val = str(rewrite_val.get("improved_prompt") or "").strip()
    except Exception:
        return result

    if not improved_val:
        return result

    merged_val = {
        **base_input,
        "user_prompt": improved_val,
        "context_profile": context_profile,
    }
    diagnosis_val = invoke_with_retry_fn(diagnosis_r.invoke, merged_val)
    weighted_val = apply_goal_weights(
        diagnosis_val, list(base_input.get("improvement_goals") or [])
    )
    opus_score = int(weighted_val.get("total_score") or 0)

    result.append({
        "iteration": iter_val,
        "input_prompt": p1_prompt,
        "improved_prompt": improved_val,
        "diagnosis_raw": diagnosis_val,
        "weighted": weighted_val,
        "rewrite": rewrite_val,
        "rewrite_model_key": "opus",
        "strategy": "phase2_validation",
        "persona": "",
    })

    # ── 2b: 추가 루프 (최대 _PHASE2_EXTRA_ITERS = 2회) ──────────────────────
    persona_mode = opus_score >= phase1_best_score
    current_p2 = improved_val

    for extra_idx in range(_PHASE2_EXTRA_ITERS):
        iter_extra = len(result) + 1
        merged_extra = {
            **base_input,
            "user_prompt": current_p2,
            "context_profile": context_profile,
        }

        # 페르소나 결정
        persona_name = ""
        persona_hint = ""
        strategy = "phase2_persona" if persona_mode else "phase2_fix"

        if persona_mode and extra_idx == 0:
            persona_name = PERSONA_CONTRARIAN
            persona_hint = get_persona_instruction(PERSONA_CONTRARIAN)
        else:
            pattern = detect_stagnation_pattern(result)
            if pattern is not None:
                persona_name = select_persona_for_pattern(pattern)
                persona_hint = get_persona_instruction(persona_name)
                strategy = pattern

        if on_iteration is not None:
            if persona_name:
                label = f"전략 전환 ({persona_name})"
            else:
                label = f"개선안 생성 ({label_opus})"
            on_iteration(iter_extra, total_p2, label, 0)

        diag_extra = invoke_with_retry_fn(diagnosis_r.invoke, merged_extra)
        w_extra = apply_goal_weights(
            diag_extra, list(base_input.get("improvement_goals") or [])
        )
        rw_extra = invoke_with_retry_fn(
            rewrite_r_opus.invoke,
            {
                **merged_extra,
                "diagnosis": diag_extra,
                "persona_instruction": persona_hint,
            },
        )
        imp_extra = str(rw_extra.get("improved_prompt") or "").strip()

        result.append({
            "iteration": iter_extra,
            "input_prompt": current_p2,
            "improved_prompt": imp_extra,
            "diagnosis_raw": diag_extra,
            "weighted": w_extra,
            "rewrite": rw_extra,
            "rewrite_model_key": "opus",
            "strategy": strategy,
            "persona": persona_name,
        })

        if not imp_extra or imp_extra == current_p2:
            break
        current_p2 = imp_extra

    return result


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
    on_iteration: Callable[[int, int, str, int], None] | None = None,
) -> dict[str, Any]:
    """Phase 1 (OpenAI 최대 3회) → Phase 2 (Opus 최대 3회) 후 최고점 결과 반환."""
    history, phase1_best_score = _run_phase1_openai_loop(
        base_input=base_input,
        context_profile=context_profile,
        diagnosis_r=diagnosis_r,
        rewrite_r_openai=rewrite_r_openai,
        routing=routing,
        invoke_with_retry_fn=invoke_with_retry_fn,
        on_iteration=on_iteration,
    )

    if rewrite_r_opus is not None:
        history = _run_phase2_opus_loop(
            history=history,
            phase1_best_score=phase1_best_score,
            base_input=base_input,
            context_profile=context_profile,
            diagnosis_r=diagnosis_r,
            rewrite_r_opus=rewrite_r_opus,
            routing=routing,
            invoke_with_retry_fn=invoke_with_retry_fn,
            on_iteration=on_iteration,
        )

    best = _select_best_iteration(history)
    best_iter_no = int(best.get("iteration") or 0) if best else None
    return {
        "best": best,
        "history": history,
        "best_iteration_no": best_iter_no,
    }
