"""수집된 JSONL 두 개를 합쳐 fewshot_examples.json에 병합.

- data/collected_huggingface.jsonl + data/collected_github.jsonl 로드
- 프롬프트 텍스트(공백 정규화) 기준 중복 제거
- data/fewshot_examples.json의 기존 레코드와 병합
  - 기존 항목 우선(수동 큐레이션 보호)
  - 외부 수집 항목은 앱 스키마(label/prompt/analysis/scores/total_hint/grade/...)로 확장
- 도메인별 집계 통계를 표준출력에 인쇄
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

DATA_DIR = _PROJECT_ROOT / "data"
HF_PATH = DATA_DIR / "collected_huggingface.jsonl"
GH_PATH = DATA_DIR / "collected_github.jsonl"
FEWSHOT_PATH = DATA_DIR / "fewshot_examples.json"

# 외부 수집 레코드를 앱 fewshot 포맷으로 확장할 때 사용하는 고정 점수 (good 품질 가정)
DEFAULT_GOOD_SCORES = {
    "clarity": "22",
    "constraint": "20",
    "output_format": "20",
    "context": "22",
}
DEFAULT_GOOD_GRADE = "우수"
DEFAULT_GOOD_TOTAL_HINT = "높음"
DEFAULT_GOOD_LEVEL = 4

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_prompt(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip()).lower()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[merge] skip (not found): {path.relative_to(_PROJECT_ROOT)}")
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    print(
        f"[merge] loaded {len(records):>5} records from "
        f"{path.relative_to(_PROJECT_ROOT)}"
    )
    return records


def _load_existing_fewshot(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _dedupe_collected(
    records: Iterable[dict[str, Any]],
    skip_keys: set[str],
) -> tuple[list[dict[str, Any]], int]:
    """프롬프트 정규화 기준 중복 제거. skip_keys에 이미 있는 프롬프트는 버린다."""
    seen: set[str] = set(skip_keys)
    unique: list[dict[str, Any]] = []
    dropped = 0
    for rec in records:
        prompt = str(rec.get("prompt") or "")
        key = _normalize_prompt(prompt)
        if not key or key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(rec)
    return unique, dropped


def _expand_to_fewshot(
    rec: dict[str, Any],
    source_tag: str,
    collected_at: str,
) -> dict[str, Any]:
    prompt = str(rec.get("prompt") or "").strip()
    action = str(rec.get("domain_action") or "")
    knowledge = str(rec.get("domain_knowledge") or "")
    analysis = (
        "외부 프롬프트 카탈로그에서 수집한 good-품질 예시. "
        f"키워드 기반 도메인: 행위={action or '(무)'}, 학문={knowledge or '(무)'}."
    )
    return {
        "label": f"외부 수집 예시 ({source_tag})",
        "prompt": prompt,
        "analysis": analysis,
        "scores": dict(DEFAULT_GOOD_SCORES),
        "total_hint": DEFAULT_GOOD_TOTAL_HINT,
        "grade": DEFAULT_GOOD_GRADE,
        "level": DEFAULT_GOOD_LEVEL,
        "source": source_tag,
        "collected_at": collected_at,
        "domain_action": action,
        "domain_knowledge": knowledge,
        "quality_tag": str(rec.get("quality_tag") or "good"),
    }


def _print_stats(label: str, examples: list[dict[str, Any]]) -> None:
    if not examples:
        print(f"\n[{label}] (empty)")
        return
    action_counter = Counter(str(ex.get("domain_action") or "") for ex in examples)
    knowledge_counter = Counter(
        str(ex.get("domain_knowledge") or "") for ex in examples
    )
    print(f"\n[{label}] total={len(examples)}")
    print("  · 행위축:")
    for name, cnt in sorted(
        action_counter.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        display = name or "(무)"
        print(f"      {display:<8} {cnt}")
    print("  · 학문축:")
    for name, cnt in sorted(
        knowledge_counter.items(), key=lambda kv: (-kv[1], kv[0])
    ):
        display = name or "(무)"
        print(f"      {display:<8} {cnt}")


def merge() -> None:
    hf_records = _load_jsonl(HF_PATH)
    gh_records = _load_jsonl(GH_PATH)

    existing = _load_existing_fewshot(FEWSHOT_PATH)
    existing_keys = {_normalize_prompt(str(it.get("prompt") or "")) for it in existing}

    hf_unique, hf_dropped = _dedupe_collected(hf_records, existing_keys)
    # huggingface 결과를 기준으로 한 뒤, github에서 추가
    seen_keys = existing_keys | {_normalize_prompt(str(r.get("prompt") or "")) for r in hf_unique}
    gh_unique, gh_dropped = _dedupe_collected(gh_records, seen_keys)

    collected_at = datetime.now().isoformat(timespec="seconds")
    expanded_hf = [_expand_to_fewshot(r, "huggingface", collected_at) for r in hf_unique]
    expanded_gh = [_expand_to_fewshot(r, "github", collected_at) for r in gh_unique]

    merged = [*existing, *expanded_hf, *expanded_gh]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEWSHOT_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n[merge] dedupe summary")
    print(f"  huggingface dropped (dup/empty): {hf_dropped}")
    print(f"  github      dropped (dup/empty): {gh_dropped}")
    print(
        f"  existing={len(existing)} + hf_new={len(expanded_hf)}"
        f" + gh_new={len(expanded_gh)} → total={len(merged)}"
    )
    print(
        f"  written → {FEWSHOT_PATH.relative_to(_PROJECT_ROOT)}"
    )

    _print_stats("existing", existing)
    _print_stats("huggingface (new)", expanded_hf)
    _print_stats("github (new)", expanded_gh)
    _print_stats("merged (total)", merged)


if __name__ == "__main__":
    merge()
