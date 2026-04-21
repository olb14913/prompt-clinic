"""학문축별 도메인 지식 수집 — 한국어 Wikipedia (옵션 A).

각 학문축 폴더(`data/wiki/{domain}/`)에 해당 분야 시드 키워드로 Wikipedia 검색 →
상위 문서를 플레인 텍스트로 받아 frontmatter 포함 markdown으로 저장한다.

- 의존성 없음 (requests + stdlib만 사용)
- `RAG_ENABLED=true`로 전환 후 `utils.vector_store.build_index()`를 재실행하면
  재작성 체인(Chain 3)의 `search_rewrite()`가 학문축 필터로 이 문서들을 참조한다.

실행:
    python scripts/collect_wiki.py                       # 모든 도메인, 도메인당 15건
    python scripts/collect_wiki.py --domain medical --limit 5
    python scripts/collect_wiki.py --domain coding --limit 20 --pause 0.5
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from requests.utils import quote as _url_quote

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
WIKI_DIR = _PROJECT_ROOT / "data" / "wiki"

WIKI_API = "https://ko.wikipedia.org/w/api.php"
WIKI_PAGE_BASE = "https://ko.wikipedia.org/wiki/"
USER_AGENT = "prompt-clinic/0.2 (data collection; contact=prompt-clinic maintainers)"

# `utils/vector_store.py::_WIKI_FOLDER_TO_DOMAIN`의 폴더명과 동일 키.
DOMAIN_SEEDS: dict[str, list[str]] = {
    "medical": [
        "질병", "증상", "약물", "진단", "치료",
        "해부학", "생리학", "면역계", "감염병", "암",
        "외과학", "내과학", "정신의학", "간호학", "공중보건학",
        "의료 윤리", "임상시험", "의약품", "예방의학", "의료기기",
    ],
    "law": [
        "법", "민법", "형법", "주식회사", "헌법",
        "행정법", "노동법", "국제법", "계약", "지식재산권",
        "저작권", "특허", "소송", "판례", "변호사",
        "형사소송법", "민사소송법", "법학", "법원", "대한민국 헌법",
    ],
    "coding": [
        "알고리즘", "자료 구조", "소프트웨어 공학", "프로그래밍 언어", "객체 지향 프로그래밍",
        "함수형 프로그래밍", "데이터베이스", "운영 체제", "컴퓨터 네트워크", "웹 개발",
        "자료형", "버전 관리", "소프트웨어 테스트", "디자인 패턴", "API",
        "파이썬 (프로그래밍 언어)", "자바스크립트", "깃 (소프트웨어)", "리눅스", "클라우드 컴퓨팅",
    ],
    "design": [
        "디자인", "그래픽 디자인", "사용자 경험", "사용자 인터페이스", "타이포그래피",
        "색상", "산업 디자인", "웹 디자인", "시각 디자인", "정보 디자인",
        "브랜드 아이덴티티", "로고", "반응형 웹 디자인", "상호작용 디자인", "디자인 사고",
        "게슈탈트 심리학", "웹 접근성", "아이콘", "편집 디자인", "어도비 포토샵",
    ],
    "science": [
        "과학", "과학적 방법", "물리학", "화학", "생물학",
        "지구과학", "천문학", "통계학", "데이터 과학", "수학",
        "실험", "가설", "과학철학", "자연과학", "공학",
        "인공지능", "기계 학습", "연구", "학술지", "논문",
    ],
    "marketing": [
        "마케팅", "브랜드", "광고", "소비자 행동", "시장 세분화",
        "STP 전략", "4P 마케팅 믹스", "디지털 마케팅", "소셜 미디어 마케팅", "콘텐츠 마케팅",
        "고객 경험", "고객 관계 관리", "판매", "전자상거래", "시장 조사",
        "검색 엔진 최적화", "브랜딩", "마케팅 믹스", "광고 캠페인", "퍼포먼스 마케팅",
    ],
    "general": [
        "글쓰기", "비판적 사고", "논리학", "수사학", "의사소통",
        "읽기", "요약", "번역", "편집", "저널리즘",
        "학습", "문제 해결", "창의성", "프레젠테이션", "말하기",
        "회의", "의사 결정", "리더십", "협업", "프로젝트 관리",
    ],
}

DEFAULT_LIMIT = 15
DEFAULT_PAUSE = 0.3
DEFAULT_MIN_CHARS = 500  # 이보다 짧은 extract는 스텁으로 보고 스킵


def _slugify(text: str) -> str:
    """파일명 안전 슬러그 — 한글 유지, 공백/특수문자 제거."""
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text[:80] or "page"


def _wiki_api_get(params: dict[str, Any]) -> dict[str, Any]:
    merged = {**params, "format": "json", "formatversion": "2"}
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(WIKI_API, params=merged, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _search_top_title(query: str) -> str:
    data = _wiki_api_get(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 1,
        }
    )
    results = data.get("query", {}).get("search", []) or []
    if not results:
        return ""
    return str(results[0].get("title") or "").strip()


def _fetch_extract(title: str) -> tuple[str, str]:
    """제목에 대한 플레인 텍스트 추출. (canonical_title, text)."""
    data = _wiki_api_get(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "true",
            "redirects": "1",
            "titles": title,
        }
    )
    pages = data.get("query", {}).get("pages", []) or []
    if not pages:
        return "", ""
    page = pages[0]
    if page.get("missing"):
        return "", ""
    return str(page.get("title") or title), str(page.get("extract") or "")


def _page_url(title: str) -> str:
    encoded = _url_quote(title.replace(" ", "_"), safe="()_")
    return WIKI_PAGE_BASE + encoded


def _write_markdown(
    out_dir: Path,
    title: str,
    text: str,
    domain: str,
    source_query: str,
) -> Path:
    slug = _slugify(title)
    path = out_dir / f"{slug}.md"
    now = datetime.now().isoformat(timespec="seconds")
    frontmatter = "\n".join(
        [
            "---",
            f"title: {title}",
            f"domain: {domain}",
            "source: ko.wikipedia.org",
            f"url: {_page_url(title)}",
            f"query: {source_query}",
            f"collected_at: {now}",
            "---",
            "",
        ]
    )
    body = f"# {title}\n\n{text.strip()}\n"
    path.write_text(frontmatter + body, encoding="utf-8")
    return path


def collect_domain(
    domain: str,
    seeds: list[str],
    limit: int,
    pause_sec: float,
    min_chars: int,
) -> int:
    out_dir = WIKI_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_titles: set[str] = set()
    written = 0
    skipped_empty = 0
    skipped_missing = 0
    skipped_short = 0
    for seed in seeds:
        if written >= limit:
            break
        try:
            title = _search_top_title(seed)
            if not title:
                skipped_missing += 1
                print(f"  [{domain}] no search result: {seed}")
                continue
            canonical, text = _fetch_extract(title)
            if not canonical or not text.strip():
                skipped_empty += 1
                print(f"  [{domain}] empty extract: {title}")
                continue
            if len(text) < min_chars:
                skipped_short += 1
                print(
                    f"  [{domain}] stub skipped: {canonical} "
                    f"({len(text):,} < {min_chars} chars)"
                )
                continue
            if canonical in seen_titles:
                continue
            seen_titles.add(canonical)
            path = _write_markdown(out_dir, canonical, text, domain, seed)
            written += 1
            print(
                f"  [{domain}] {written}/{limit} saved: {canonical} "
                f"({len(text):,} chars) → {path.relative_to(_PROJECT_ROOT)}"
            )
        except requests.HTTPError as exc:
            print(f"  [{domain}] HTTP error for '{seed}': {exc}")
        except Exception as exc:
            print(f"  [{domain}] error for '{seed}': {type(exc).__name__}: {exc}")
        time.sleep(pause_sec)
    print(
        f"[{domain}] done. wrote={written}, skipped_missing={skipped_missing}, "
        f"skipped_empty={skipped_empty}, skipped_short={skipped_short}"
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wikipedia 도메인 지식 수집 (옵션 A)")
    parser.add_argument(
        "--domain",
        action="append",
        choices=sorted(DOMAIN_SEEDS.keys()),
        help="특정 도메인만 수집 (여러 번 지정 가능, 미지정 시 전체)",
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"도메인당 최대 파일 수 (기본 {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--pause", type=float, default=DEFAULT_PAUSE,
        help=f"요청 간 대기 초 (기본 {DEFAULT_PAUSE})",
    )
    parser.add_argument(
        "--min-chars", type=int, default=DEFAULT_MIN_CHARS,
        help=f"스텁 문서 최소 길이 (기본 {DEFAULT_MIN_CHARS}자 미만 스킵)",
    )
    args = parser.parse_args(argv)

    if args.domain:
        targets = {d: DOMAIN_SEEDS[d] for d in args.domain}
    else:
        targets = DOMAIN_SEEDS

    total = 0
    for domain, seeds in targets.items():
        print(
            f"\n[collect_wiki] domain={domain} "
            f"seeds={len(seeds)} limit={args.limit}"
        )
        total += collect_domain(
            domain, seeds, args.limit, args.pause, args.min_chars
        )
    print(f"\n[collect_wiki] done. total files written: {total}")
    print(
        "  다음 단계: RAG_ENABLED=true 후 "
        "`python -c \"from utils.vector_store import build_index; build_index()\"`"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
