"""F-16-4 보고용 집계 스크립트.

``data/prompt_runs.jsonl``(= RAG ON 베이스라인, F-13-5 이후 레코드)과
``data/rag_off_eval.jsonl``(= eval_rag_off.py 결과)을 비교해 콘솔 + Markdown
형태로 요약을 출력한다.

사용법::

    python -m scripts.summarize_rag_eval
    python -m scripts.summarize_rag_eval --markdown docs/rag_evaluation_stats.md

주의: ``--markdown`` 경로는 **덮어쓰기 대상 전용**이다. 사람이 작성한
``docs/rag_evaluation_report.md`` 보고서와는 경로를 반드시 분리할 것.

집계 항목:
- 전체 표본 대비 ON vs OFF 평균/중앙값 (total_score, 4항목, before/after 토큰, 토큰 절감율)
- "토큰 줄이기" 개선목표 서브셋 동일 비교
- OFF 레코드에 한해 latency 통계
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows cp949 기본 콘솔에서도 한글/유니코드가 깨지지 않게 UTF-8 강제.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

RAG_ON_PATH = ROOT / "data" / "prompt_runs.jsonl"
RAG_OFF_PATH = ROOT / "data" / "rag_off_eval.jsonl"

CRITERIA = ["clarity", "constraint", "output_format", "context"]
TOKEN_REDUCE_GOAL = "토큰 줄이기"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _has_token_counts(rec: dict[str, Any]) -> bool:
    return int(rec.get("before_token_count") or 0) > 0


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _reduction_rate(before: int, after: int) -> float:
    """(before-after)/before * 100. before=0이면 0.0 반환."""
    if before <= 0:
        return 0.0
    return (before - after) / before * 100.0


def _avg(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


@dataclass
class Summary:
    label: str
    n: int
    total_score_avg: float
    total_score_med: float
    criteria_avg: dict[str, float]
    before_tok_avg: float
    after_tok_avg: float
    reduction_avg: float
    reduction_med: float
    latency_total_ms_avg: float  # OFF 전용, ON은 0


def summarize(records: list[dict[str, Any]], label: str) -> Summary:
    n = len(records)
    if n == 0:
        return Summary(label, 0, 0, 0, {k: 0.0 for k in CRITERIA}, 0, 0, 0, 0, 0)

    totals = [_safe_int(r.get("total_score")) for r in records]
    crit_avg: dict[str, float] = {}
    for key in CRITERIA:
        vals = [_safe_int((r.get("scores") or {}).get(key)) for r in records]
        crit_avg[key] = _avg(vals)

    befores = [_safe_int(r.get("before_token_count")) for r in records]
    afters = [_safe_int(r.get("after_token_count")) for r in records]
    reductions = [_reduction_rate(b, a) for b, a in zip(befores, afters)]

    latencies = [float(r.get("latency_total_ms") or 0) for r in records if r.get("latency_total_ms")]
    return Summary(
        label=label,
        n=n,
        total_score_avg=_avg(totals),
        total_score_med=_median(totals),
        criteria_avg=crit_avg,
        before_tok_avg=_avg(befores),
        after_tok_avg=_avg(afters),
        reduction_avg=_avg(reductions),
        reduction_med=_median(reductions),
        latency_total_ms_avg=_avg(latencies),
    )


def _fmt_row(s: Summary) -> str:
    crits = " / ".join(f"{k}={s.criteria_avg[k]:.1f}" for k in CRITERIA)
    latency = f" | latency≈{s.latency_total_ms_avg:.0f}ms" if s.latency_total_ms_avg else ""
    return (
        f"[{s.label}] n={s.n}  score avg={s.total_score_avg:.1f} "
        f"(med={s.total_score_med:.1f})  {crits}  "
        f"tok {s.before_tok_avg:.1f}→{s.after_tok_avg:.1f}  "
        f"reduce avg={s.reduction_avg:.1f}% (med={s.reduction_med:.1f}%){latency}"
    )


def _markdown_table(summaries: list[Summary], title: str) -> str:
    lines = [f"### {title}", ""]
    header = (
        "| 조건 | n | total_score 평균 | total_score 중앙 | "
        "clarity | constraint | output_format | context | "
        "before tok | after tok | 토큰 절감율 평균(%) | 평균 latency(ms) |"
    )
    sep = "|" + "|".join(["---"] * 12) + "|"
    lines.append(header)
    lines.append(sep)
    for s in summaries:
        c = s.criteria_avg
        lines.append(
            "| {label} | {n} | {ts:.1f} | {tm:.1f} | "
            "{cl:.1f} | {co:.1f} | {of:.1f} | {cx:.1f} | "
            "{bt:.1f} | {at:.1f} | {red:.1f} | {lat:.0f} |".format(
                label=s.label,
                n=s.n,
                ts=s.total_score_avg,
                tm=s.total_score_med,
                cl=c["clarity"],
                co=c["constraint"],
                of=c["output_format"],
                cx=c["context"],
                bt=s.before_tok_avg,
                at=s.after_tok_avg,
                red=s.reduction_avg,
                lat=s.latency_total_ms_avg,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG ON/OFF 비교 집계")
    parser.add_argument(
        "--markdown",
        type=str,
        default="",
        help=(
            "자동 생성 표를 내보낼 Markdown 경로 (예: docs/rag_evaluation_stats.md). "
            "사람이 작성한 보고서(docs/rag_evaluation_report.md)를 가리키면 덮어쓰므로 "
            "반드시 다른 파일을 지정할 것."
        ),
    )
    args = parser.parse_args()

    # 안전장치: 사람이 쓴 보고서 경로로 덮어쓰려는 시도 차단
    protected = {"docs/rag_evaluation_report.md", "docs\\rag_evaluation_report.md"}
    if args.markdown and args.markdown.replace("\\", "/").lower() in {
        p.replace("\\", "/").lower() for p in protected
    }:
        print(
            "[summarize] 오류: --markdown 인자가 수기 보고서 경로를 가리킵니다. "
            "다른 경로(예: docs/rag_evaluation_stats.md)를 지정하세요."
        )
        return 2

    rag_on_all = [r for r in _load_jsonl(RAG_ON_PATH) if _has_token_counts(r)]
    rag_off_all = [r for r in _load_jsonl(RAG_OFF_PATH) if _has_token_counts(r)]

    rag_on_tr = [r for r in rag_on_all if TOKEN_REDUCE_GOAL in (r.get("improvement_goals") or [])]
    rag_off_tr = [r for r in rag_off_all if TOKEN_REDUCE_GOAL in (r.get("improvement_goals") or [])]

    summaries_full = [
        summarize(rag_on_all, "RAG ON (운영 로그)"),
        summarize(rag_off_all, "RAG OFF (재실행)"),
    ]
    summaries_tr = [
        summarize(rag_on_tr, "RAG ON / 토큰 줄이기"),
        summarize(rag_off_tr, "RAG OFF / 토큰 줄이기"),
    ]

    print("=" * 100)
    print("[RAG 전후 비교 - 전체 표본]")
    for s in summaries_full:
        print(" ", _fmt_row(s))

    print()
    print("[토큰 줄이기 서브셋]")
    for s in summaries_tr:
        print(" ", _fmt_row(s))
    print("=" * 100)

    if args.markdown:
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        parts = [
            "# RAG 정량 평가 집계 (자동 생성)",
            "",
            "- RAG ON 데이터: `data/prompt_runs.jsonl` 중 `before_token_count > 0` 필터",
            "- RAG OFF 데이터: `scripts/eval_rag_off.py` 재실행 결과 (`data/rag_off_eval.jsonl`)",
            "- 집계 스크립트: `python -m scripts.summarize_rag_eval --markdown <path>`",
            "",
            _markdown_table(summaries_full, "1. 전체 표본 (RAG ON 18건 vs 동일 조건 RAG OFF 재실행)"),
            _markdown_table(summaries_tr, "2. 토큰 줄이기 서브셋"),
        ]
        md_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
        print(f"[markdown] → {md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
