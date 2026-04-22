"""F-16-4 보고용 일회성 평가 스크립트.

기존 ``data/prompt_runs.jsonl``에서 F-13-5 이후(= ``before_token_count > 0``)
레코드를 뽑아, **동일 조건**에서 ``RAG_ENABLED=false``로 파이프라인을 재실행한다.

결과는 ``data/rag_off_eval.jsonl``에 append된다. 운영 로그
(``data/prompt_runs.jsonl``)는 건드리지 않는다.

핵심 설계 원칙 (main.py와 동일 흐름 재현):
- RAG_ENABLED만 false로 강제. 나머지 env(특히 SELF_IMPROVE_ENABLED)는
  ``.env`` 값을 그대로 따른다 → 원본 18건과 루프 유무가 맞음.
- SELF_IMPROVE_ENABLED=true: run_self_improve_loop 사용, best 선택.
- SELF_IMPROVE_ENABLED=false: rewrite 후 **개선결과를 재진단**해서 해당 점수를
  total_score로 저장 (main.py 1687~1708과 동일).
- 토큰 수는 원본(user_prompt) / 개선(improved_prompt) 기준 tiktoken.
- latency는 전 체인 합계(self-improve 포함).

사용법::

    python -m scripts.eval_rag_off                 # 전체 18건 재실행
    python -m scripts.eval_rag_off --limit 3       # 앞 3건만 스모크 테스트
    python -m scripts.eval_rag_off --dedupe        # user_prompt 기준 중복 제거
    python -m scripts.eval_rag_off --output data/custom.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except Exception:
    pass

# RAG만 강제로 끈다. SELF_IMPROVE_ENABLED는 .env 값을 그대로 유지해야
# 원본 18건(대부분 자가개선 루프를 돈 상태)과 조건이 맞는다.
os.environ["RAG_ENABLED"] = "false"

from chains.model_router import (  # noqa: E402
    build_openai_rewrite_llm,
    build_opus_llm,
    make_openai_llm,
    read_routing_config,
)
from chains.pipeline import build_chain_segments  # noqa: E402
from chains.self_improve_chain import apply_goal_weights, run_self_improve_loop  # noqa: E402
from utils.data_pipeline import _count_tokens, infer_prompt_level  # noqa: E402

PROMPT_RUNS = ROOT / "data" / "prompt_runs.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "rag_off_eval.jsonl"


def load_baseline_records(path: Path = PROMPT_RUNS) -> list[dict[str, Any]]:
    """F-13-5 이후(before_token_count > 0) 레코드만 반환."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            if int(rec.get("before_token_count") or 0) <= 0:
                continue
            records.append(rec)
    return records


def dedupe_by_prompt(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(user_prompt, improvement_goals 정렬 튜플) 단위로 중복 제거."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    out: list[dict[str, Any]] = []
    for rec in records:
        key = (
            str(rec.get("user_prompt") or "").strip(),
            tuple(sorted(rec.get("improvement_goals") or [])),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _invoke_passthrough(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """invoke_with_retry 없이 그대로 호출하는 patch-through.
    run_self_improve_loop가 invoke_with_retry_fn 인자를 요구하기 때문에 필요."""
    return fn(*args, **kwargs)


def run_single_off(
    record: dict[str, Any],
    routing: Any,
    context_r: Any,
    diagnosis_r: Any,
    rewrite_r: Any,
    rewrite_r_openai: Any,
    rewrite_r_opus: Any,
) -> dict[str, Any]:
    """단일 레코드를 RAG OFF 조건으로 재실행.
    main.py의 SELF_IMPROVE 분기 흐름(1630~1705)을 동일하게 흉내낸다."""
    base_input: dict[str, Any] = {
        "purpose": str(record.get("purpose") or ""),
        "output_format": str(record.get("output_format") or ""),
        "improvement_goals": list(record.get("improvement_goals") or []),
        "user_prompt": str(record.get("user_prompt") or ""),
    }

    t0 = time.perf_counter()

    # Step 1: context
    t_ctx0 = time.perf_counter()
    context_profile = context_r.invoke(base_input)
    t_ctx = time.perf_counter() - t_ctx0
    merged = {**base_input, "context_profile": context_profile}

    # Step 2: diagnosis (원본 프롬프트 진단)
    t_diag0 = time.perf_counter()
    diagnosis = diagnosis_r.invoke(merged)
    t_diag = time.perf_counter() - t_diag0
    diagnosis_weighted = apply_goal_weights(diagnosis, base_input["improvement_goals"])

    # Step 3: rewrite (self-improve on/off 분기)
    loop_history: list[dict[str, Any]] = []
    best_iteration_no: int | None = None
    rewrite: dict[str, Any]
    improved_weighted: dict[str, Any]

    if routing.self_improve_enabled:
        t_rw0 = time.perf_counter()
        loop_result = run_self_improve_loop(
            base_input=base_input,
            context_profile=context_profile,
            diagnosis_r=diagnosis_r,
            rewrite_r_openai=rewrite_r_openai,
            rewrite_r_opus=rewrite_r_opus,
            routing=routing,
            max_iters=routing.self_improve_max_iterations,
            invoke_with_retry_fn=_invoke_passthrough,
            on_iteration=None,
        )
        t_rw = time.perf_counter() - t_rw0
        loop_history = list(loop_result.get("history") or [])
        best_iteration_no = loop_result.get("best_iteration_no")
        best = loop_result.get("best") or {}
        rewrite = best.get("rewrite") or {}
        improved_weighted = best.get("weighted") or diagnosis_weighted
        t_rediag = 0.0
    else:
        merged_for_rewrite = {**merged, "diagnosis": diagnosis}
        t_rw0 = time.perf_counter()
        rewrite = rewrite_r.invoke(merged_for_rewrite)
        t_rw = time.perf_counter() - t_rw0

        # 개선결과 재진단 (main.py 1694~1705 동일)
        improved_prompt = str((rewrite or {}).get("improved_prompt") or "")
        improved_input = {
            **base_input,
            "user_prompt": improved_prompt,
            "context_profile": context_profile,
        }
        t_rediag0 = time.perf_counter()
        improved_diagnosis = diagnosis_r.invoke(improved_input)
        t_rediag = time.perf_counter() - t_rediag0
        improved_weighted = apply_goal_weights(
            improved_diagnosis, base_input["improvement_goals"]
        )

    improved_prompt = str((rewrite or {}).get("improved_prompt") or "")
    total_score = int(improved_weighted.get("total_score") or 0)
    level_info = infer_prompt_level(total_score)
    t_total = time.perf_counter() - t0

    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "paired_ts": record.get("ts"),
        "rag_enabled": False,
        "self_improve_enabled": bool(routing.self_improve_enabled),
        "prompt_name": record.get("prompt_name"),
        "purpose": base_input["purpose"],
        "output_format": base_input["output_format"],
        "improvement_goals": base_input["improvement_goals"],
        "user_prompt": base_input["user_prompt"],
        "improved_prompt": improved_prompt,
        "total_score": total_score,
        "grade": str(improved_weighted.get("grade") or ""),
        "scores": {
            k: str(v) for k, v in (improved_weighted.get("weighted_scores") or {}).items()
        },
        "original_scores": {
            k: str(v) for k, v in (diagnosis_weighted.get("weighted_scores") or {}).items()
        },
        "original_total_score": int(diagnosis_weighted.get("total_score") or 0),
        "prompt_level": level_info,
        "domain_action": str(context_profile.get("domain_action") or ""),
        "domain_knowledge": str(context_profile.get("domain_knowledge") or ""),
        "before_token_count": _count_tokens(base_input["user_prompt"]),
        "after_token_count": _count_tokens(improved_prompt),
        "loop_history": loop_history,
        "best_iteration_no": best_iteration_no,
        "latency_context_ms": round(t_ctx * 1000, 1),
        "latency_diagnosis_ms": round(t_diag * 1000, 1),
        "latency_rewrite_ms": round(t_rw * 1000, 1),
        "latency_rediag_ms": round(t_rediag * 1000, 1),
        "latency_total_ms": round(t_total * 1000, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG OFF 재실행 평가 스크립트")
    parser.add_argument("--limit", type=int, default=0, help="재실행 건수 제한 (0=전체)")
    parser.add_argument(
        "--dedupe", action="store_true", help="user_prompt 중복 제거 후 실행"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT), help="출력 JSONL 경로"
    )
    parser.add_argument(
        "--clean", action="store_true", help="실행 전 출력 파일 비움 (append 기본)"
    )
    args = parser.parse_args()

    baseline = load_baseline_records()
    if args.dedupe:
        baseline = dedupe_by_prompt(baseline)
    if args.limit > 0:
        baseline = baseline[: args.limit]

    if not baseline:
        print("[eval_rag_off] 재실행할 베이스라인 레코드가 없습니다.")
        return 1

    routing = read_routing_config()
    print(f"[eval_rag_off] RAG_ENABLED={os.environ.get('RAG_ENABLED')}")
    print(
        f"[eval_rag_off] SELF_IMPROVE_ENABLED={routing.self_improve_enabled} "
        f"(max_iters={routing.self_improve_max_iterations})"
    )
    print(f"[eval_rag_off] 재실행 대상: {len(baseline)}건")

    llm = make_openai_llm(routing.openai_diagnosis_model, routing.temperature)
    context_r, diagnosis_r, rewrite_r = build_chain_segments(llm)

    rewrite_r_openai = None
    rewrite_r_opus = None
    if routing.self_improve_enabled:
        rewrite_openai_llm = build_openai_rewrite_llm(routing)
        _, _, rewrite_r_openai = build_chain_segments(rewrite_openai_llm)
        rewrite_opus_llm = build_opus_llm(routing)
        if rewrite_opus_llm is not None:
            _, _, rewrite_r_opus = build_chain_segments(rewrite_opus_llm)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.clean and output_path.exists():
        output_path.unlink()

    success = 0
    with output_path.open("a", encoding="utf-8") as out:
        for i, rec in enumerate(baseline, 1):
            prompt_preview = str(rec.get("user_prompt") or "")[:40].replace("\n", " ")
            try:
                result = run_single_off(
                    rec,
                    routing,
                    context_r,
                    diagnosis_r,
                    rewrite_r,
                    rewrite_r_openai,
                    rewrite_r_opus,
                )
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                success += 1
                iters = len(result.get("loop_history") or [])
                print(
                    f"  [{i}/{len(baseline)}] OK score={result['total_score']} "
                    f"(orig={result['original_total_score']}) "
                    f"tokens {result['before_token_count']}→{result['after_token_count']} "
                    f"iters={iters} latency={result['latency_total_ms']}ms | {prompt_preview!r}"
                )
            except Exception as exc:
                print(f"  [{i}/{len(baseline)}] FAIL {exc} | {prompt_preview!r}")

    print(f"[eval_rag_off] 완료: {success}/{len(baseline)}건 → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
