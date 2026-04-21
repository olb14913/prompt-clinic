"""GitHub `f/awesome-chatgpt-prompts` CSV 수집 → 분류 → JSONL 저장.

- https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv
- CSV 파싱 후 행위축/학문축 분류 적용
- data/collected_github.jsonl 로 저장 (UTF-8, line-per-record, 전체 재작성)
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import requests

# awesome-chatgpt-prompts CSV에는 길이가 긴 필드가 있어 기본 한도(131072)를 초과함.
# Windows 환경에서 sys.maxsize가 OverflowError를 내는 경우에 대비해 단계적으로 낮춤.
_csv_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_limit)
        break
    except OverflowError:
        _csv_limit = int(_csv_limit // 10)
        if _csv_limit < 10**6:
            csv.field_size_limit(10**7)
            break

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._classify import build_collected_record  # noqa: E402

CSV_URL = (
    "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv"
)
OUTPUT_PATH = _PROJECT_ROOT / "data" / "collected_github.jsonl"
PROMPT_COLUMN_CANDIDATES = ("prompt", "Prompt", "text")


def _fetch_csv(url: str) -> str:
    print(f"[collect_github] fetching {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _pick_prompt(row: dict[str, str]) -> str:
    for key in PROMPT_COLUMN_CANDIDATES:
        if key in row and isinstance(row[key], str):
            text = row[key].strip()
            if text:
                return text
    return ""


def collect() -> int:
    csv_text = _fetch_csv(CSV_URL)
    reader = csv.DictReader(io.StringIO(csv_text))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as fp:
        for row in reader:
            prompt_text = _pick_prompt(row)
            if not prompt_text:
                skipped += 1
                continue
            record = build_collected_record(prompt_text)
            fp.write(json.dumps(record, ensure_ascii=False))
            fp.write("\n")
            written += 1

    print(
        f"[collect_github] wrote {written} records "
        f"(skipped {skipped}) → {OUTPUT_PATH.relative_to(_PROJECT_ROOT)}"
    )
    return written


if __name__ == "__main__":
    collect()
