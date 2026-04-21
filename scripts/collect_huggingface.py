"""HuggingFace `fka/awesome-chatgpt-prompts` 로드 → 분류 → JSONL 저장.

- datasets 라이브러리로 로드
- 각 프롬프트에 대해 행위축/학문축 키워드 분류
- data/collected_huggingface.jsonl 로 저장 (UTF-8, line-per-record, 전체 재작성)

실행:
    python -m scripts.collect_huggingface
    또는
    python scripts/collect_huggingface.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# `python scripts/collect_huggingface.py` 실행 대응
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._classify import build_collected_record  # noqa: E402

OUTPUT_PATH = _PROJECT_ROOT / "data" / "collected_huggingface.jsonl"
DATASET_ID = "fka/awesome-chatgpt-prompts"
PROMPT_COLUMN_CANDIDATES = ("prompt", "Prompt", "text", "content")


def _extract_prompt(row: dict) -> str:
    for key in PROMPT_COLUMN_CANDIDATES:
        if key in row and isinstance(row[key], str):
            text = row[key].strip()
            if text:
                return text
    return ""


def collect() -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - 런타임 가이드
        print(
            "[collect_huggingface] `datasets` 라이브러리가 필요합니다. "
            "`pip install datasets` 로 설치하세요.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(f"[collect_huggingface] loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID)
    # 보통 'train' 스플릿 하나지만 방어적으로 처리
    split_name = "train" if "train" in ds else next(iter(ds.keys()))
    rows = ds[split_name]
    print(f"[collect_huggingface] split={split_name}, rows={len(rows)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        for row in rows:
            prompt_text = _extract_prompt(row)
            if not prompt_text:
                skipped += 1
                continue
            record = build_collected_record(prompt_text)
            fp.write(json.dumps(record, ensure_ascii=False))
            fp.write("\n")
            written += 1

    print(
        f"[collect_huggingface] wrote {written} records "
        f"(skipped {skipped}) → {OUTPUT_PATH.relative_to(_PROJECT_ROOT)}"
    )
    return written


if __name__ == "__main__":
    collect()
