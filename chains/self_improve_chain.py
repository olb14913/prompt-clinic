"""F-09: 자가개선 루프(autoresearch 패턴)."""

from __future__ import annotations

import hashlib
from typing import Any, Callable

from chains.model_router import RoutingConfig, model_key_to_label, resolve_rewrite_model_key

CRITERION_KEYS = ["clarity", "constraint", "output_format", "context"]

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


def _count_tokens_approx(texts: list[str]) -> int:
    """F-13-4: 텍스트 리스트의 토큰 수 추정. tiktoken 없으면 len//4 fallback."""
    combined = " ".join(texts)
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(combined))
    except Exception:
        return len(combined) // 4


def _opus_budget_exceeded(
    calls_used: int,
    tokens_used: int,
    estimated_tokens: int,
    routing: RoutingConfig,
) -> bool:
    """F-13-4: Opus 예산 초과 여부. 0은 무제한."""
    if routing.opus_max_calls > 0 and calls_used >= routing.opus_max_calls:
        return True
    if routing.opus_max_tokens > 0 and (
        tokens_used + estimated_tokens
    ) > routing.opus_max_tokens:
        return True
    return False


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
    history: list[dict[str, Any]] = []
    prev_score = -1
    opus_calls_used = 0
    opus_tokens_used = 0

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
        # F-13-4: Opus 예산 초과 시 OpenAI fallback
        if model_key == "opus":
            estimated = _count_tokens_approx([current_prompt])
            if _opus_budget_exceeded(
                opus_calls_used, opus_tokens_used, estimated, routing
            ):
                model_key = "openai"
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
        # F-13-4: Opus 사용량 업데이트
        if model_key == "opus":
            opus_calls_used += 1
            opus_tokens_used += _count_tokens_approx([current_prompt, str(rewrite)])
        improved = str(rewrite.get("improved_prompt") or "").strip()
        record = {
            "iteration": iter_no,
            "input_prompt": current_prompt,
            "improved_prompt": improved,
            "diagnosis_raw": diagnosis,
            "weighted": weighted,
            "rewrite": rewrite,
            "rewrite_model_key": model_key,
            "strategy": "default",
            "persona": "",
        }
        history.append(record)

        # 점수 향상이 없거나 개선 결과가 비어 있으면 조기 종료
        stagnant = score <= prev_score or not improved or improved == current_prompt
        if stagnant:
            # F-13-2: 정체 감지 → 페르소나 전략으로 1회 재시도
            pattern = detect_stagnation_pattern(history)
            if pattern is not None:
                persona = select_persona_for_pattern(pattern)
                persona_hint = get_persona_instruction(persona)
                if on_iteration is not None:
                    on_iteration(iter_no, max_iters, f"전략 전환 ({persona})")
                # F-13-4: 페르소나 재시도 Opus 예산 체크
                retry_model_key = model_key
                retry_rewrite_r = rewrite_r
                if retry_model_key == "opus":
                    estimated_retry = _count_tokens_approx([current_prompt])
                    if _opus_budget_exceeded(
                        opus_calls_used, opus_tokens_used, estimated_retry, routing
                    ):
                        retry_model_key = "openai"
                        retry_rewrite_r = rewrite_r_openai
                try:
                    rewrite_retry = invoke_with_retry_fn(
                        retry_rewrite_r.invoke,
                        {
                            **merged_for_diag,
                            "diagnosis": diagnosis,
                            "persona_instruction": persona_hint,
                        },
                    )
                    # F-13-4: 페르소나 재시도 Opus 사용량 업데이트
                    if retry_model_key == "opus":
                        opus_calls_used += 1
                        opus_tokens_used += _count_tokens_approx(
                            [current_prompt, str(rewrite_retry)]
                        )
                    improved_retry = str(
                        rewrite_retry.get("improved_prompt") or ""
                    ).strip()
                    if improved_retry and improved_retry != current_prompt:
                        retry_record = {
                            "iteration": iter_no,
                            "input_prompt": current_prompt,
                            "improved_prompt": improved_retry,
                            "diagnosis_raw": diagnosis,
                            "weighted": weighted,
                            "rewrite": rewrite_retry,
                            "rewrite_model_key": retry_model_key,
                            "strategy": pattern,
                            "persona": persona,
                        }
                        history.append(retry_record)
                        current_prompt = improved_retry
                        prev_score = score
                        continue
                except Exception:
                    pass
            break
        prev_score = score
        current_prompt = improved

    # F-13-1: 루프 종료 후 최적 이터레이션 선택
    best = _select_best_iteration(history)
    best_iter_no = int(best.get("iteration") or 0)
    return {
        "best": best,
        "history": history,
        "best_iteration_no": best_iter_no,
    }
