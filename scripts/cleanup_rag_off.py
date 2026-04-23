"""data/rag_off_eval.jsonl에서 구버전 레코드만 제거.

구버전(수정 전 eval_rag_off.py)은 ``self_improve_enabled`` 필드를 기록하지
않았고, ``total_score`` 의미도 원본 프롬프트 점수라 평가에 쓸 수 없다.

이 스크립트는 해당 필드가 없는 레코드를 걸러내고, 유효 레코드만 원 파일에
다시 쓴다. 제거 전 백업은 ``rag_off_eval.backup.jsonl``로 저장한다.

사용법::

    python -m scripts.cleanup_rag_off
    python -m scripts.cleanup_rag_off --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TARGET = ROOT / "data" / "rag_off_eval.jsonl"
BACKUP = ROOT / "data" / "rag_off_eval.backup.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="rag_off_eval.jsonl 정리")
    parser.add_argument("--dry-run", action="store_true", help="실제로 쓰지 않고 집계만")
    args = parser.parse_args()

    if not TARGET.exists():
        print(f"[cleanup] 대상 파일 없음: {TARGET}")
        return 1

    kept: list[str] = []
    dropped = 0
    total = 0
    with TARGET.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            if not isinstance(rec, dict) or "self_improve_enabled" not in rec:
                dropped += 1
                continue
            kept.append(line)

    print(f"[cleanup] 전체 {total}건 → 유지 {len(kept)}건 / 제거 {dropped}건")

    if args.dry_run:
        print("[cleanup] --dry-run 모드: 실제 파일은 수정하지 않음")
        return 0

    if dropped == 0:
        print("[cleanup] 제거할 레코드 없음. 파일 유지.")
        return 0

    shutil.copy2(TARGET, BACKUP)
    print(f"[cleanup] 백업: {BACKUP}")

    with TARGET.open("w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")
    print(f"[cleanup] 완료: {TARGET} ({len(kept)}건)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
