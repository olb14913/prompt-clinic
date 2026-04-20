"""Streamlit: 프롬프트 진단 클리닉 메인 앱."""

from __future__ import annotations

import base64
import html
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory

from chains.model_router import (
    build_opus_llm,
    build_openai_rewrite_llm,
    make_openai_llm,
    read_routing_config,
)
from chains.pipeline import build_chain_segments
from chains.drift_chain import (
    build_drift_chain,
    compute_drift_score,
    prep_drift_input,
)
from chains.gate_chain import (
    build_gate_chain,
    build_question_chain,
    compute_gate_total_score,
    prep_gate_input,
    prep_question_input,
)
from chains.self_improve_chain import apply_goal_weights, run_self_improve_loop
from utils.data_pipeline import sync_learning_data

load_dotenv()

# 와이어 기본값 "텍스트" — 과거 스냅샷의 "글"은 prefill 시 "텍스트"로 치환
OUTPUT_FORMAT_OPTIONS = ["텍스트", "리스트", "표", "코드", "JSON"]
_OUTPUT_FORMAT_LEGACY = {"글": "텍스트"}
IMPROVEMENT_OPTIONS = [
    "출력 품질 높이기",
    "맥락 보완",
    "구조화",
    "일관성 높이기",
    "토큰 줄이기",
]

CRITERION_LABELS = {
    "clarity": "명확성",
    "constraint": "제약조건",
    "output_format": "출력형식",
    "context": "맥락반영도",
}

CRITERION_ICONS = {
    "clarity": "🎯",
    "constraint": "🔒",
    "output_format": "📋",
    "context": "🧩",
}

# 디자인 토큰 (Figma 기준 — UI 렌더링 전용)
UI_PRIMARY_BLUE = "#285aff"
UI_AFTER_BG = "rgba(40,90,255,0.05)"
UI_BEFORE_BG = "#fcfcfc"
UI_CARD_BG = "#f6f6f6"
UI_DOVE_GRAY = "#636363"
UI_BADGE_GREEN = "#01a701"
UI_BADGE_YELLOW = "#ffd700"
UI_BADGE_RED = "#d40924"
UI_BORDER_ALTO = "#DEDEDE"

# 진단기록 행 토글 — 항목별 개선포인트(st.expander) 쉐브론과 유사한 크기·형태
_HISTORY_ROW_CHEVRON_SVG = (
    '<svg class="pc-h-chevron" width="12" height="12" viewBox="0 0 24 24" '
    'fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
    '<path d="M6 9l6 6 6-6" stroke="currentColor" stroke-width="2.25" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

# 브랜드 마크: 말풍선(좌하 꼬리) 가로 그라데이션 채움 + 흰색 외곽선·청진기 라인
_PC_BRAND_LOGO_SVG = """
<svg class="pc-brand-svg" width="56" height="56" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs>
<linearGradient id="pcBrandFillGrad" x1="0" y1="32" x2="64" y2="32" gradientUnits="userSpaceOnUse">
<stop stop-color="#7DD3FC"/><stop offset="1" stop-color="#5EEAD4"/>
</linearGradient>
</defs>
<path d="M32 7C18.2 7 8 17.4 8 30.5c0 8.2 4.6 15.3 11.5 18.4L9 55l12.5-7.2c2.8.9 5.8 1.4 9 1.4 13.8 0 24-10.4 24-23.5S45.8 7 32 7z"
fill="url(#pcBrandFillGrad)" stroke="#FFFFFF" stroke-width="2.2" stroke-linejoin="round"/>
<path d="M23 24c2.2-4 6.6-6.5 11.5-6.5 4.2 0 8 2 10.3 5.2" stroke="#FFFFFF" stroke-width="2.4" fill="none" stroke-linecap="round"/>
<path d="M26.5 24.5v11M37.5 24.5v11" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
<circle cx="32" cy="40" r="5.2" stroke="#FFFFFF" stroke-width="2.3" fill="none"/>
<path d="M32 34.8V30" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round"/>
</svg>
""".strip()

_LOGO_PATH = Path(__file__).resolve().parent / "prompt_clinic_logo.png"


def _brand_logo_html() -> str:
    """리포지토리 루트의 PNG 로고; 없으면 인라인 SVG 폴백."""
    if _LOGO_PATH.is_file():
        b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
        return (
            f'<img class="pc-brand-img" src="data:image/png;base64,{b64}" '
            'width="56" height="56" alt="" decoding="async" />'
        )
    return _PC_BRAND_LOGO_SVG


# figma.html(와이어) 등급 배지 문구 — 앱 로직의 grade 문자열과 매핑만 함
FIGMA_GRADE_BADGE: dict[str, tuple[str, str, str]] = {
    "우수": ("즉시사용가능", UI_BADGE_GREEN, "#ffffff"),
    # 와이어: 노란 배지도 라벨은 흰색(가독성은 디자인 시안 기준)
    "보통": ("보완권장", UI_BADGE_YELLOW, "#ffffff"),
    "개선필요": ("재작성 필요", UI_BADGE_RED, "#ffffff"),
}

PROMPT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9가-힣._\- ]+$")


def _normalize_output_format_option(val: str) -> str:
    v = (val or "").strip()
    return _OUTPUT_FORMAT_LEGACY.get(v, v)


def _render_improvement_point_buttons() -> list[str]:
    """개선 포인트 버튼형 다중 선택: min 1 / max 3."""
    if "improvement_goals_input" not in st.session_state:
        st.session_state.improvement_goals_input = []

    sel: list[str] = list(st.session_state.improvement_goals_input)
    max_selected = 3

    with st.container(key="pc_goal_pills"):
        gc = st.columns(5, gap="small")

        for i, opt in enumerate(IMPROVEMENT_OPTIONS):
            with gc[i]:
                active = opt in sel
                disabled = (not active) and (len(sel) >= max_selected)

                if st.button(
                    opt,
                    key=f"pc_goal_btn_{i}",
                    type="primary" if active else "secondary",
                    use_container_width=True,
                    disabled=disabled,
                ):
                    if active:
                        st.session_state.improvement_goals_input = [
                            x for x in sel if x != opt
                        ]
                    else:
                        st.session_state.improvement_goals_input = [*sel, opt]
                    st.rerun()

    return list(st.session_state.get("improvement_goals_input") or [])


def invoke_with_retry(
    invoke_fn: Callable[..., Any],
    *args: Any,
    on_retry: Callable[[int, int], None] | None = None,
    **kwargs: Any,
) -> Any:
    """최대 2회 재시도(간격 3초), 총 3회 시도.

    on_retry(retry_num, max_retries): 재시도 직전 호출되는 콜백.
    """
    delay_sec = 3
    max_retries = 2
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return invoke_fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                if on_retry is not None:
                    on_retry(attempt + 1, max_retries)
                time.sleep(delay_sec)
    assert last_error is not None
    raise last_error


def build_markdown_report(
    prompt_name: str,
    purpose: str,
    output_format: str,
    goals: list[str],
    user_prompt: str,
    weighted: dict[str, Any],
    improved: str,
    changes: list[dict[str, Any]],
) -> str:
    lines = [
        "# Prompt Clinic 진단 리포트",
        "",
        f"생성 시각: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 맥락",
        f"- 프롬프트 명: {prompt_name}",
        f"- 목적: {purpose or '(없음)'}",
        f"- 출력 형식: {output_format}",
        f"- 개선 목적: {', '.join(goals) if goals else '(없음)'}",
        "",
        "## 진단 (가중치 반영)",
        f"- **종합 점수**: {weighted['total_score']} / 100",
        f"- **등급**: {weighted['grade_badge']} {weighted['grade']}",
        "",
    ]
    for key, label in CRITERION_LABELS.items():
        sc = weighted["weighted_scores"][key]
        lines.append(f"- **{label}**: {sc} / 25")
        lines.append(f"  - 원인: {weighted['reasons'].get(key, '')}")
    lines.extend(
        [
            "",
            "## Before / After",
            "",
            "### Before",
            "```",
            user_prompt,
            "```",
            "",
            "### After",
            "```",
            improved,
            "```",
            "",
            "## 변경 이유",
        ]
    )
    for ch in changes:
        crit = ch.get("criterion", "")
        bf = ch.get("before", "")
        af = ch.get("after", "")
        rs = ch.get("reason", "")
        lines.append(
            f"- **{crit}**: `{bf}` → `{af}`  \n  {rs}"
        )
    return "\n".join(lines)


def _find_korean_font() -> str | None:
    candidates = [
        Path(__file__).parent / "data" / "fonts" / "NanumGothic.ttf",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def build_pdf_report(
    prompt_name: str,
    purpose: str,
    output_format: str,
    goals: list[str],
    user_prompt: str,
    weighted: dict[str, Any],
    improved: str,
    changes: list[dict[str, Any]],
) -> bytes:
    """Generate a Korean-friendly PDF report via fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("fpdf2 not installed. Run: pip install fpdf2")

    font_path = _find_korean_font()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    if font_path:
        pdf.add_font("NanumGothic", "", font_path)

    def _h(size: int, bold: bool = False) -> None:
        if font_path:
            pdf.set_font("NanumGothic", size=size)
        else:
            style = "B" if bold else ""
            pdf.set_font("Helvetica", style=style, size=size)

    def _cell(txt: str, size: int = 11, bold: bool = False, ln: bool = True) -> None:
        _h(size, bold)
        pdf.multi_cell(0, 7, txt)
        if ln:
            pdf.ln(1)

    _cell("Prompt Clinic \uc9c4\ub2e8 \ub9ac\ud3ec\ud2b8", size=18, bold=True)
    _cell(f"\uc0dd\uc131 \uc2dc\uac01: {datetime.now().isoformat(timespec='seconds')}", size=9)
    pdf.ln(4)

    _cell("[\ub9e5\ub77d]", size=13, bold=True)
    _cell(f"\u2022 \ud504\ub86c\ud504\ud2b8 \uba85: {prompt_name}")
    _cell(f"\u2022 \ubaa9\uc801: {purpose or '(\uc5c6\uc74c)'}")
    _cell(f"\u2022 \ucd9c\ub825 \ud615\uc2dd: {output_format}")
    _cell(f"\u2022 \uac1c\uc120 \ubaa9\uc801: {', '.join(goals) if goals else '(\uc5c6\uc74c)'}")
    pdf.ln(3)

    _cell("[\uc9c4\ub2e8 (\uac00\uc911\uce58 \ubc18\uc601)]", size=13, bold=True)
    _cell(f"\u2022 \uc885\ud569 \uc810\uc218: {weighted['total_score']} / 100")
    _cell(f"\u2022 \ub4f1\uae09: {weighted['grade_badge']} {weighted['grade']}")
    pdf.ln(2)
    for key, label in CRITERION_LABELS.items():
        sc = weighted["weighted_scores"][key]
        reason = weighted["reasons"].get(key, "")
        _cell(f"\u2022 {label}: {sc} / 25")
        if reason:
            _cell(f"  \uc6d0\uc778: {reason}", size=9)
    pdf.ln(3)

    _cell("[Before]", size=13, bold=True)
    _h(9)
    pdf.set_fill_color(245, 245, 245)
    pdf.multi_cell(0, 6, user_prompt, fill=True)
    pdf.ln(3)

    _cell("[After]", size=13, bold=True)
    _h(9)
    pdf.multi_cell(0, 6, improved, fill=True)
    pdf.ln(3)

    _cell("[\ubcc0\uacbd \uc774\uc720]", size=13, bold=True)
    for ch in changes:
        crit = ch.get("criterion", "")
        bf = ch.get("before", "")
        af = ch.get("after", "")
        rs = ch.get("reason", "")
        _cell(f"\u2022 {crit}: {bf} \u2192 {af}")
        if rs:
            _cell(f"  {rs}", size=9)
    pdf.ln(2)

    return bytes(pdf.output())


def build_obsidian_report(
    prompt_name: str,
    purpose: str,
    output_format: str,
    goals: list[str],
    user_prompt: str,
    weighted: dict[str, Any],
    improved: str,
    changes: list[dict[str, Any]],
    domain_action: str = "",
    domain_knowledge: str = "",
) -> str:
    """Generate Obsidian-compatible markdown with YAML frontmatter."""
    grade = str(weighted.get("grade") or "")
    tags = [t for t in ["prompt-clinic", grade, domain_action, domain_knowledge] if t]
    tag_yaml = "\n".join(f"  - {t}" for t in tags)
    goals_yaml = "\n".join(f"  - {g}" for g in goals) if goals else "  - (없음)"

    frontmatter = f"""---
title: "{prompt_name}"
date: "{datetime.now().isoformat(timespec='seconds')}"
purpose: "{purpose or '(없음)'}"
output_format: "{output_format}"
goals:
{goals_yaml}
total_score: {weighted['total_score']}
grade: "{grade}"
domain_action: "{domain_action}"
domain_knowledge: "{domain_knowledge}"
tags:
{tag_yaml}
---"""

    lines = [
        frontmatter,
        "",
        f"# 프롬프트 아카이빙: {prompt_name}",
        "",
        "## 메타데이터",
        f"- 생성 시각: {datetime.now().isoformat(timespec='seconds')}",
        f"- 종합 점수: {weighted['total_score']} / 100",
        f"- 등급: {weighted.get('grade_badge', '')} {grade}",
        f"- 행위 도메인: {domain_action or '(미분류)'}",
        f"- 학문 도메인: {domain_knowledge or '(미분류)'}",
        "",
        "## 원본 프롬프트 (Before)",
        "",
        "```",
        user_prompt,
        "```",
        "",
        "## 개선 프롬프트 (After)",
        "",
        "```",
        improved,
        "```",
        "",
        "## 진단 결과",
        "",
        "### 항목별 점수",
    ]
    for key, label in CRITERION_LABELS.items():
        sc = weighted["weighted_scores"][key]
        reason = weighted["reasons"].get(key, "")
        lines.append(f"- **{label}**: {sc} / 25")
        if reason:
            lines.append(f"  - 원인: {reason}")
    lines += [
        "",
        "## 변경 이유",
        "",
    ]
    for ch in changes:
        crit = ch.get("criterion", "")
        bf = ch.get("before", "")
        af = ch.get("after", "")
        rs = ch.get("reason", "")
        lines.append(f"- **{crit}**: `{bf}` → `{af}`")
        if rs:
            lines.append(f"  - {rs}")
    return "\n".join(lines)


def dynamic_text_area_height(text: str, min_px: int = 150, max_px: int = 400) -> int:
    t = text or ""
    line_count = max(1, t.count("\n") + 1)
    px = 52 + line_count * 22
    return max(min_px, min(max_px, px))


def criterion_expander_title(label: str, final: int, bonus_pts: int) -> str:
    _ = bonus_pts
    return f"{label} ({final} / 25)"


def score_progress_bar(score: int) -> None:
    ratio = min(1.0, max(0.0, score / 30.0))
    st.progress(ratio)
    badge = "🟢" if score >= 20 else ("🟡" if score >= 10 else "🔴")
    st.markdown(
        f"<p style='margin:0.2rem 0 0 0;font-size:1.1rem;line-height:1;'>{badge}</p>",
        unsafe_allow_html=True,
    )


def sanitize_prompt_name_for_filename(prompt_name: str) -> str:
    """다운로드 파일명에 안전한 프롬프트 명으로 정규화."""
    trimmed = (prompt_name or "").strip()
    safe = re.sub(r"[^A-Za-z0-9가-힣._-]", "_", trimmed)
    safe = safe.strip("._-")
    return safe or "prompt_clinic"


def render_clipboard_copy_button(
    text: str, label: str, dom_id: str, *, primary: bool = False
) -> None:
    """primary=True: 액션 블루. False: figma 와이어 스타일(연회색 필 + 테두리)."""
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    safe_id = "".join(c if c.isalnum() else "_" for c in dom_id)[:64]
    if primary:
        btn_style = (
            f"background-color:{UI_PRIMARY_BLUE};color:#fff;"
            f"border:1px solid {UI_PRIMARY_BLUE};"
        )
    else:
        btn_style = (
            f"background-color:{UI_BEFORE_BG};color:#0B0B0B;"
            f"border:1px solid {UI_BORDER_ALTO};"
            "box-shadow:0 1px 2px rgba(0,0,0,0.05);"
        )
    components.html(
        f"""
        <div style="display:flex;justify-content:flex-end;width:100%;">
          <button id="{safe_id}" type="button" style="
            {btn_style}
            padding:0.375rem 0.75rem;border-radius:0.25rem;cursor:pointer;
            font-size:0.875rem;font-family:sans-serif;
            width:122px;max-width:100%;white-space:nowrap;">
            {label}
          </button>
        </div>
        <script>
        (function() {{
          const binary = atob("{b64}");
          const len = binary.length;
          const bytes = new Uint8Array(len);
          for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
          const decoded = new TextDecoder("utf-8").decode(bytes);
          const el = document.getElementById("{safe_id}");
          if (el) el.onclick = () => {{
            navigator.clipboard.writeText(decoded).then(() => {{
              el.textContent = "✅ 복사 완료!";
              setTimeout(() => {{ el.textContent = "{label}"; }}, 2000);
            }});
          }};
        }})();
        </script>
        """,
        height=52,
    )

def inject_live_counter(*, container_key: str, limit: int) -> None:
    """textarea input 이벤트를 감지해 입력창 내부 우측에 실시간 글자 수 카운터 표시."""
    components.html(
        f"""
        <script>
        (function() {{
          const doc = window.parent.document;

          function mountCounter() {{
            const wrapper = doc.querySelector('.st-key-{container_key}');
            if (!wrapper) return false;

            const textAreaShell = wrapper.querySelector('[data-testid="stTextArea"]');
            if (!textAreaShell) return false;

            const textarea = textAreaShell.querySelector('textarea');
            if (!textarea) return false;

            textAreaShell.classList.add('pc-live-counter-target');

            let counter = textAreaShell.querySelector('.pc-live-counter');
            if (!counter) {{
              counter = doc.createElement('div');
              counter.className = 'pc-live-counter';
              textAreaShell.appendChild(counter);
            }}

            function renderCount() {{
              counter.textContent = `${{textarea.value.length}} / {limit}`;
            }}

            if (!textarea.dataset.pcCounterBound) {{
              textarea.addEventListener('input', renderCount);
              textarea.dataset.pcCounterBound = '1';
            }}

            renderCount();
            return true;
          }}

          let tries = 0;
          const timer = setInterval(() => {{
            const ok = mountCounter();
            tries += 1;
            if (ok || tries > 20) clearInterval(timer);
          }}, 150);
        }})();
        </script>
        """,
        height=0,
    )


def _inject_pc_theme_css() -> None:
    """앱 본문 UI 토큰 스타일 (결과·히스토리·푸터 공통, 한 번 주입)."""
    st.markdown(
        f"""
<style>
@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");
.stApp {{
  font-family: "Pretendard", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
}}
[data-testid="stAppViewContainer"] > .main {{
  background: #f8f8f8;
}}
section[data-testid="stSidebar"] {{
  display: none !important;
}}
.st-key-pc_input_shell {{
  background: #ffffff !important;
  border: 1px solid {UI_BORDER_ALTO} !important;
  border-radius: 8px !important;
  padding: 0.5rem 0.25rem 1rem !important;
}}
@media (min-width: 640px) {{
  .st-key-pc_input_shell {{
    padding: 0.75rem 1rem 1.25rem !important;
  }}
}}
.pc-wire-hero {{
  margin: 0 0 1.35rem 0;
}}
.pc-wire-hero-row {{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}}
.pc-wire-brand-mark .pc-brand-svg,
.pc-wire-brand-mark .pc-brand-img {{
  display: block;
  width: 56px;
  height: 56px;
}}
.pc-wire-brand-mark .pc-brand-img {{
  object-fit: contain;
}}
.pc-wire-brand-text {{
  text-align: left;
  min-width: 0;
}}
@media (max-width: 520px) {{
  .pc-wire-hero-row {{
    flex-direction: column;
  }}
  .pc-wire-brand-text {{
    text-align: center;
  }}
}}
.pc-wire-title {{
  margin: 0;
  font-size: 1.65rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #0b0b0b;
}}
.pc-wire-desc {{
  margin: 0.35rem 0 0;
  font-size: 0.95rem;
  font-weight: 500;
  color: {UI_DOVE_GRAY};
  line-height: 1.45;
}}
.pc-wire-section {{
  margin: 0 0 0.85rem 0;
  font-size: 16px;
  font-weight: 700;
  line-height: 1.25;
  color: #0b0b0b;
}}
.pc-wire-muted {{
  font-weight: 500;
  font-size: 0.875rem;
  color: {UI_DOVE_GRAY};
}}

/* 맥락 수집 헤더 + 필수 안내 문구 */
.pc-section-head-inline {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 0.45rem 0;
  flex-wrap: wrap;
}}

.pc-section-head-inline .pc-wire-section {{
  margin: 0;
  font-size: 16px;
  font-weight: 800;
  color: #0b0b0b;
}}

.pc-section-required-text {{
  font-size: 11px;
  font-weight: 500;
  color: #9a9a9a;
  line-height: 1.2;
}}

.pc-label-row-single {{
  margin-bottom: 0.35rem;
}}

.pc-input-counter-target {{
  position: relative;
}}

.pc-input-counter {{
  position: absolute;
  right: 14px;
  bottom: 10px;
  font-size: 12px;
  line-height: 1;
  color: #6c6c6c;
  background: rgba(255, 255, 255, 0.92);
  padding: 0 2px;
  z-index: 30;
  pointer-events: none;
}}

.pc-input-counter-target input {{
  padding-right: 56px !important;
}}
.st-key-prompt_name_field {{
  margin-bottom: 0.8rem;
}}

.pc-char-right {{
  text-align: right;
  font-size: 11px;
  line-height: 1.3;
  color: #6c6c6c;
  margin: -0.35rem 0 0.5rem 0;
}}
/* 개선 포인트 ↔ 진단 프롬프트 구분 (별도 hr markdown 제거 — 빈 element-container 방지) */
.st-key-pc_input_shell .st-key-user_prompt_field {{
  border-top: 1px solid {UI_BORDER_ALTO} !important;
  margin-top: 1.25rem !important;
  padding-top: 1.1rem !important;
}}
.pc-inline-err {{
  color: {UI_BADGE_RED};
  font-size: 13px;
  font-weight: 600;
  margin: 0;
}}
/* S-04 진행·재시도·완료·API 오류 — 와이어 흰 박스 가운데 정렬 */
.pc-phase-banner {{
  background: #ffffff !important;
  border: 1px solid {UI_BORDER_ALTO} !important;
  border-radius: 10px !important;
  padding: 12px 16px !important;
  margin: 8px 0 12px 0 !important;
  box-sizing: border-box !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  min-height: 48px !important;
  text-align: center !important;
}}
.pc-phase-banner-text {{
  font-size: 14px !important;
  font-weight: 600 !important;
  color: #0b0b0b !important;
  line-height: 1.4 !important;
}}
.pc-label-row {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 0.35rem;
}}
.pc-wire-label-strong {{
  font-weight: 700;
  font-size: 16px;
  color: #0b0b0b;
  letter-spacing: -0.01em;
}}
.pc-char-user-prompt {{
  text-align: right;
  font-size: 11px;
  line-height: 1.3;
  color: #6c6c6c;
  margin: 0.2rem 0 0.5rem 0;
}}
.pc-policy-note {{
  font-size: 11px;
  color: #8a8a8a;
  line-height: 1.6;
  margin: 0.35rem 0 0.55rem 0;
  white-space: nowrap;
}}

.pc-policy-note a {{
  color: #7a7a7a;
  font-weight: 700;
  text-decoration: underline;
}}

.pc-policy-note a:hover {{
  color: #5f5f5f;
}}
/* 개선 포인트: primary/secondary 동일 높이·작은 글자·한 줄 */
.st-key-pc_goal_pills div[data-testid="stHorizontalBlock"] {{
  gap: 0.2rem !important;
  flex-wrap: nowrap !important;
}}
.st-key-pc_goal_pills div[data-testid="stColumn"] {{
  min-width: 0 !important;
  flex: 1 1 0 !important;
  padding-left: 0.1rem !important;
  padding-right: 0.1rem !important;
}}
.pc-goal-pill-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin: 0.35rem 0 0.6rem 0;
}}

.st-key-pc_input_shell [data-baseweb="input"] input,
.st-key-pc_input_shell [data-baseweb="textarea"] textarea {{
  background-color: {UI_BEFORE_BG} !important;
  border-color: {UI_BORDER_ALTO} !important;
  border-radius: 6px !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
  font-size: 14px !important;
}}
.st-key-pc_input_shell [data-baseweb="select"] > div {{
  background-color: {UI_BEFORE_BG} !important;
  border-color: {UI_BORDER_ALTO} !important;
  border-radius: 6px !important;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
}}

.st-key-output_format_field [data-baseweb="select"] span {{
  color: #636363 !important;
  font-size: 14px !important;
  font-weight: 500 !important;
}}

.st-key-output_format_field [data-baseweb="select"] svg {{
  color: #636363 !important;
}}

.pc-context-field-label {{
  color: #636363 !important;
  font-weight : 500 !important;
}}

/* 맥락 수집 2열: 라벨 줄 높이·컨트롤 상단선 수평 정렬 */
.st-key-pc_name_format_row div[data-testid="stHorizontalBlock"] {{
  align-items: flex-start !important;
}}
.st-key-pc_name_format_row .pc-context-field-label {{
  display: block;
  min-height: 24px;
  margin-bottom: 6px;
  line-height: 1.35;
  font-size: 16px !important;
  font-weight: 700 !important;
}}
/* 맥락 수집: 위젯 래퍼 패딩 통일(높이 어긋남 방지) */
.st-key-purpose_field [data-testid="element-container"],
.st-key-output_format_field [data-testid="element-container"] {{
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}}
/* 사용목적 textarea ↔ 출력 형식 select 동일 박스(44px) + 리사이즈 제거 */
.st-key-purpose_field textarea {{
  resize: none !important;
}}
.st-key-purpose_field [data-baseweb="textarea"] {{
  min-height: 44px !important;
  height: 44px !important;
  max-height: 44px !important;
  box-sizing: border-box !important;
  overflow: hidden !important;
}}
.st-key-purpose_field [data-baseweb="textarea"] textarea {{
  resize: none !important;
  min-height: 100% !important;
  height: 100% !important;
  max-height: 100% !important;
  line-height: 1.35 !important;
  padding: 9px 10px !important;
  box-sizing: border-box !important;
  overflow-y: auto !important;
}}
.st-key-purpose_field [data-baseweb="textarea"] textarea::-webkit-resizer {{
  display: none !important;
  width: 0 !important;
  height: 0 !important;
}}
.st-key-output_format_field [data-baseweb="select"] {{
  min-height: 38px !important;
  height: 38px !important;
  max-height: 38px !important;
  box-sizing: border-box !important;
}}
.st-key-output_format_field [data-baseweb="select"] > div {{
  min-height: 38px !important;
  height: 38px !important;
  max-height: 38px !important;
  box-sizing: border-box !important;
  display: flex !important;
  align-items: center !important;
}}
.st-key-pc_input_shell label[data-testid="stWidgetLabel"] p {{
  font-size: 15px !important;
  font-weight: 700 !important;
  color: #0b0b0b !important;
}}
.st-key-pc_input_shell [data-testid="stMarkdownContainer"] p {{
  font-size: 14px;
}}
.st-key-pc_input_shell div[data-testid="stButton"] button[kind="primary"] {{
  background-color: {UI_PRIMARY_BLUE} !important;
  border-color: {UI_PRIMARY_BLUE} !important;
  color: #fff !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  min-height: 38px !important;
}}

/* 진단 시작 버튼만 회색 직사각형으로 */
.st-key-pc_run_diagnosis button[kind="secondary"] {{
  background-color: #9e9e9e !important;
  border: 1px solid #9e9e9e !important;
  color: #ffffff !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  min-height: 38px !important;
  width: 100% !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
}}

/* 진단 시작 버튼 disabled 상태 */
.st-key-pc_run_diagnosis button[kind="secondary"]:disabled {{
  background-color: #9e9e9e !important;
  border: 1px solid #9e9e9e !important;
  color: #ffffff !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  min-height: 38px !important;
  width: 100% !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
  opacity: 1 !important;
  cursor: not-allowed !important;
  -webkit-text-fill-color: #ffffff !important;
}}

/* 버튼 내부 텍스트/아이콘도 흰색 고정 */
.st-key-pc_run_diagnosis button[kind="secondary"] * {{
  color: #ffffff !important;
  fill: #ffffff !important;
  stroke: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}}

.st-key-pc_run_diagnosis button[kind="secondary"]:disabled * {{
  color: #ffffff !important;
  fill: #ffffff !important;
  stroke: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}}

/* 진단 시작 제외: 개선 포인트 pill만 전용 스타일 */
.st-key-pc_input_shell .st-key-pc_goal_pills div[data-testid="stButton"] button {{
  width: 100% !important;
  min-height: 34px !important;
  height: auto !important;
  max-height: none !important;
  padding: 4px 3px !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  line-height: 1.2 !important;
  border-radius: 999px !important;
  box-sizing: border-box !important;
}}

.st-key-pc_input_shell .st-key-pc_goal_pills div[data-testid="stButton"] button[kind="primary"] {{
  background-color: {UI_PRIMARY_BLUE} !important;
  border-color: {UI_PRIMARY_BLUE} !important;
  color: #fff !important;
}}
.st-key-pc_input_shell .st-key-pc_goal_pills div[data-testid="stButton"] button[kind="secondary"] {{
  background-color: #ffffff !important;
  border: 1px solid {UI_BORDER_ALTO} !important;
  color: #374151 !important;
}}
.st-key-pc_input_shell .st-key-pc_goal_pills div[data-testid="stButton"] button[kind="secondary"]:disabled {{
  background-color: #ffffff !important;
  border: 1px solid {UI_BORDER_ALTO} !important;
  color: #374151 !important;
  opacity: 1 !important;
  cursor: not-allowed !important;
  -webkit-text-fill-color: #374151 !important;
}}
.pc-bullet-blue {{
  color: {UI_PRIMARY_BLUE};
  font-weight: 700;
  margin-right: 0.25rem;
}}
.pc-change-line {{
  margin: 0.35rem 0;
  line-height: 1.45;
  color: #262730;
}}
.pc-prompt-box {{
  padding: 8px 12px;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  overflow: auto;
  box-sizing: border-box;
}}
.pc-prompt-before {{
  background: {UI_BEFORE_BG};
  border: 1px solid {UI_BORDER_ALTO};
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  border-radius: 6px;
  color: {UI_DOVE_GRAY};
}}
.pc-prompt-after {{
  background: {UI_AFTER_BG};
  border: 1px solid {UI_PRIMARY_BLUE};
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  border-radius: 6px;
}}
.pc-prompt-inner-before {{
  color: {UI_DOVE_GRAY};
  font-family: 'Segoe UI', 'Geist', system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.35;
}}
.pc-prompt-inner-after {{
  color: #0B0B0B;
  font-family: ui-monospace, 'Geist Mono', 'Cascadia Mono', monospace;
  font-size: 14px;
  line-height: 1.43;
}}
.pc-improve-h3 {{
  font-size: 14px;
  font-weight: 700;
  color: #0B0B0B;
  margin: 0 0 0.5rem 0;
}}
.pc-improve-li-label {{
  color: {UI_DOVE_GRAY};
  font-weight: 700;
}}
/* 히스토리 바깥 카드 — 마크다운 래퍼용(필요 시) */
.pc-hist-outer-card {{
  background: #FFFFFF;
  border: 1px solid {UI_BORDER_ALTO};
  border-radius: 8px;
  padding: 1rem 1.25rem 1.25rem 1.25rem;
}}
/* 액션 행: 다운로드가 포함된 가로 블록만 primary 블루로 통일 (진단 시작 버튼과 분리) */
div[data-testid="stHorizontalBlock"]:has([data-testid="stDownloadButton"]) button {{
  background-color: {UI_PRIMARY_BLUE} !important;
  color: #fff !important;
  border-color: {UI_PRIMARY_BLUE} !important;
  font-size: 12px !important;
  line-height: 1.2 !important;
  white-space: nowrap !important;
  min-height: 38px !important;
}}
div[data-testid="stHorizontalBlock"]:has([data-testid="stDownloadButton"])
  [data-testid="stDownloadButton"] button {{
  background-color: {UI_PRIMARY_BLUE} !important;
  color: #fff !important;
  border-color: {UI_PRIMARY_BLUE} !important;
  font-size: 12px !important;
  line-height: 1.2 !important;
  white-space: nowrap !important;
  min-height: 38px !important;
}}
div[data-testid="stHorizontalBlock"]:has([data-testid="stDownloadButton"])
  [data-testid="stButton"] button {{
  font-size: 12px !important;
  line-height: 1.2 !important;
  white-space: nowrap !important;
  min-height: 38px !important;
}}
.pc-hist-badge {{
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  color: #fff;
}}

.pc-footer-wrap {{
  text-align: center;
  margin-top: 2rem;
  padding-top: 1rem;
}}

.pc-footer-links {{
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  font-size: 13px;
  color: {UI_DOVE_GRAY};
  margin-bottom: 6px;
}}

.pc-footer-links a {{
  color: {UI_DOVE_GRAY};
  text-decoration: underline;
}}

.pc-footer-links a:hover {{
  color: #4b4b4b;
}}

.pc-footer-divider {{
  color: #9a9a9a;
}}

.pc-footer-copy {{
  color: {UI_DOVE_GRAY};
  font-size: 0.85rem;
}}

.block-container {{
  max-width: 819px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 2rem !important;
  padding-bottom: 2rem !important;
}}
.pc-card {{
  background: #fff;
  border: 1px solid {UI_BORDER_ALTO};
  border-radius: 8px;
  padding: 24px;
  margin: 0 0 12px 0;
}}
.pc-card-title {{
  margin: 0 0 0.7rem 0;
  font-size: 18px;
  line-height: 28px;
  font-weight: 700;
  color: #0B0B0B;
}}
.pc-score-card {{
  background: #F7F9FA;
  border: 1px solid #DBDBDB;
  border-radius: 12px;
  padding: 12px 12px 10px 12px;
  height: 108px;
  box-sizing: border-box;
}}
.pc-score-head {{
  font-size: 13px;
  color: {UI_DOVE_GRAY};
  margin: 0 0 4px 0;
  font-weight: 500;
}}
.pc-score-value {{
  margin: 0 0 7px 0;
  font-size: 19px;
  font-weight: 700;
  color: #0B0B0B;
  text-align: right;
}}
.pc-score-bar {{
  width: 100%;
  height: 8px;
  border-radius: 999px;
  background: rgba(40, 90, 255, 0.2);
  overflow: hidden;
}}
.pc-score-fill {{
  height: 100%;
  background: {UI_PRIMARY_BLUE};
}}
.pc-total-row {{
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
  margin: 0.55rem 0 0.35rem 0;
}}
.pc-total-text {{
  font-size: 30px;
  font-weight: 700;
  color: #111827;
  line-height: 1.05;
}}
.pc-total-text small {{
  font-size: 17px;
  font-weight: 500;
  color: #374151;
}}
.pc-total-prefix {{
  font-size: 29px;
  font-weight: 700;
  color: #111827;
  line-height: 1;
}}
.pc-pill {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  padding: 2px 8px;
  min-height: 26px;
  font-size: 11px;
  font-weight: 600;
  color: #FFFFFF;
}}
.pc-pill-green {{ background: {UI_BADGE_GREEN}; }}
.pc-pill-yellow {{ background: {UI_BADGE_YELLOW}; color: #FFFFFF; }}
.pc-pill-red {{ background: {UI_BADGE_RED}; }}
.pc-top-btn {{
  position: fixed;
  right: 20px;
  bottom: 24px;
  width: 44px;
  height: 44px;
  border-radius: 999px;
  border: 1px solid #DEDEDE;
  background: #FFFFFF;
  color: #374151;
  font-size: 20px;
  font-weight: 700;
  cursor: pointer;
  display: none;
  z-index: 9999;
  box-shadow: 0 6px 12px rgba(0, 0, 0, 0.12);
}}

/* F-11-2 실시간 글자 수 카운터 */
.pc-live-counter-target {{
  position: relative;
}}

.pc-live-counter {{
  position: absolute;
  right: 12px;
  bottom: 8px;
  font-size: 11px;
  line-height: 1;
  color: #6c6c6c;
  background: rgba(255, 255, 255, 0.92);
  padding: 2px 6px;
  border-radius: 999px;
  z-index: 30;
  pointer-events: none;
}}

.pc-live-counter-target textarea {{
  padding-bottom: 28px !important;
}}

/* 로딩 상태 바 (피그마 S-03 — 맥락 수집 카드 하단 외부) */
.pc-loading-bar {{
  display: flex;
  align-items: center;
  gap: 10px;
  background: #ffffff;
  border: 1px solid {UI_BORDER_ALTO};
  border-radius: 8px;
  padding: 12px 16px;
  margin: 8px 0 12px 0;
  min-height: 48px;
}}
.pc-loading-text {{
  font-size: 14px;
  font-weight: 600;
  color: #0b0b0b;
  line-height: 1.4;
}}
@keyframes pc-dot-blink {{
  0%, 80%, 100% {{ opacity: 0; }}
  40% {{ opacity: 1; }}
}}
.pc-dot {{
  animation: pc-dot-blink 1.4s infinite;
  font-size: 14px;
  font-weight: 600;
  color: #0b0b0b;
}}
.pc-d1 {{ animation-delay: 0s; }}
.pc-d2 {{ animation-delay: 0.23s; }}
.pc-d3 {{ animation-delay: 0.46s; }}

</style>
""",
        unsafe_allow_html=True,
    )


def _norm_criterion_label(raw: Any) -> str:
    s = str(raw or "").strip()
    if s in CRITERION_LABELS:
        return CRITERION_LABELS[s]
    for _k, lab in CRITERION_LABELS.items():
        if lab == s:
            return lab
    return s or "기타"


def _figma_grade_badge(grade: str) -> tuple[str, str, str]:
    """와이어(figma.html) 배지 문구·색 — (표시문구, 배경, 글자색)."""
    g = (grade or "").strip()
    if g in FIGMA_GRADE_BADGE:
        return FIGMA_GRADE_BADGE[g]
    return (g or "-", UI_DOVE_GRAY, "#ffffff")


def _grade_emoji_label(grade: str) -> str:
    g = (grade or "").strip()
    if g == "우수":
        return "즉시사용가능"
    if g == "보통":
        return "보완권장"
    if g == "개선필요":
        return "재작성 필요"
    return g or "-"


def _first_two_lines_ellipsis(text: str, max_chars: int = 110) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    lines = raw.splitlines()
    chosen = lines[:2]
    merged = " ".join(chosen).strip()
    if len(lines) > 2 or len(merged) > max_chars:
        return (merged[:max_chars].rstrip() + "...") if len(merged) > max_chars else (merged + "...")
    return merged


def _render_prompt_readonly_box(text: str, *, variant: str) -> None:
    """Before/After 본문 박스 (figma 스펙: Before Dove Gray, After Mono)."""
    h = dynamic_text_area_height(text)
    raw = text or ""
    escaped = html.escape(raw)
    cls = "pc-prompt-before" if variant == "before" else "pc-prompt-after"
    inner_cls = (
        "pc-prompt-inner-before" if variant == "before" else "pc-prompt-inner-after"
    )
    st.markdown(
        f'<div class="pc-prompt-box {cls}" style="min-height:{h}px;max-height:420px;">'
        f'<span class="{inner_cls}">{escaped}</span></div>',
        unsafe_allow_html=True,
    )


def _render_diagnosis_score_cards(
    weighted: dict[str, Any], crit_keys: list[str], bonus_map: dict[str, Any]
) -> None:
    cols_score = st.columns(4)
    for i, key in enumerate(crit_keys):
        sc = weighted["weighted_scores"][key]
        _ = bonus_map
        icon = CRITERION_ICONS.get(key, "•")
        label = f"{icon} {CRITERION_LABELS[key]}"
        fill = max(0, min(100, int(round((sc / 25) * 100))))
        with cols_score[i]:
            st.markdown(
                f"""
<div class="pc-score-card">
  <p class="pc-score-head">{label}</p>
  <p class="pc-score-value">{sc} <span style="font-size:14px;font-weight:500;color:#6B7280;">/ 25</span></p>
  <div class="pc-score-bar"><div class="pc-score-fill" style="width:{fill}%;"></div></div>
</div>
""",
                unsafe_allow_html=True,
            )


def _render_improvement_bullets(changes: list[dict[str, Any]]) -> None:
    """figma: '항목 별 개선내용' — 불릿 #285AFF, 기준명 #636363 굵게, 본문 #0B0B0B."""
    st.markdown(
        '<p class="pc-improve-h3">항목 별 개선내용</p>',
        unsafe_allow_html=True,
    )
    if not changes:
        st.caption("항목별 변경 내역이 없습니다.")
        return
    for ch in changes:
        lab = _norm_criterion_label(ch.get("criterion"))
        reason = (ch.get("reason") or "").strip()
        if not reason:
            b = (ch.get("before") or "").strip()
            a = (ch.get("after") or "").strip()
            reason = (f"{b} → {a}" if (b or a) else "(내용 없음)")
        p_open = (
            '<p class="pc-change-line" style="display:flex;align-items:flex-start;'
            'gap:8px;margin:0.35rem 0;color:#0B0B0B;">'
        )
        st.markdown(
            p_open
            + f'<span class="pc-bullet-blue" style="flex-shrink:0;">•</span>'
            f'<span><span class="pc-improve-li-label">{html.escape(lab)}</span>'
            f" : {html.escape(reason)}</span></p>",
            unsafe_allow_html=True,
        )


def _pc_phase_banner(label: str) -> str:
    """S-04 와이어: 흰 박스·연한 테두리·가운데 정렬."""
    esc = html.escape(label, quote=False)
    return (
        f'<div class="pc-phase-banner">'
        f'<span class="pc-phase-banner-text">{esc}</span></div>'
    )


def _pc_loading_bar(label: str) -> str:
    """피그마 S-03 로딩 바: 카드 하단 외부, 아이콘 + 텍스트 + 말줄임표 애니메이션."""
    text = html.escape((label or "").rstrip(".").rstrip(), quote=False)
    return (
        '<div class="pc-loading-bar">'
        f'<span class="pc-loading-text">{text}</span>'
        '<span class="pc-dot pc-d1">.</span>'
        '<span class="pc-dot pc-d2">.</span>'
        '<span class="pc-dot pc-d3">.</span>'
        '</div>'
    )


def _make_retry_phase_cb(
    phase_slot: Any, step_label: str
) -> Callable[[int, int], None]:
    """invoke_with_retry 시 동일 스타일 재시도 배너."""
    def on_retry(n: int, total: int) -> None:
        phase_slot.markdown(
            _pc_phase_banner(f"⚠️ {step_label} 재시도 중 ({n}/{total})..."),
            unsafe_allow_html=True,
        )
    return on_retry


def _render_gate_ui() -> None:
    """F-20-2 맥락 보완 배너 + F-20-3 소크라테스 질문 expander."""
    gate_result = st.session_state.get("gate_result")
    if not gate_result:
        return
    if float(gate_result.get("total_score", 0.0)) <= 0.5:
        return

    weak_axes: list[str] = list(gate_result.get("weak_axes") or [])
    weak_str = "、".join(weak_axes) if weak_axes else "맥락 정보"

    # F-20-2: 안내 배너
    st.markdown(
        f'<div style="background:rgba(255,193,7,0.08);border:1px solid rgba(255,193,7,0.55);'
        f'border-radius:10px;padding:12px 16px;margin:4px 0 10px 0;">'
        f'<span style="font-size:14px;font-weight:600;color:#0b0b0b;line-height:1.5;">'
        f"💡 <strong>{html.escape(weak_str)}</strong> 항목을 보강하면 더 정확한 진단이 가능합니다. "
        f"사용목적 또는 개선 포인트를 위에서 보완해 보세요.</span></div>",
        unsafe_allow_html=True,
    )

    # F-20-3: 보완 질문 expander
    questions: list[str] = list(
        (st.session_state.get("gate_questions") or {}).get("questions") or []
    )
    with st.expander("💬 맥락 보완 질문 (선택사항)", expanded=True):
        st.caption(
            "아래 질문을 참고해 위 '사용목적'이나 '개선 포인트'를 수정하거나, "
            "수정 없이 '진단 계속하기'를 눌러 바로 진단할 수 있습니다."
        )
        for q in questions:
            st.markdown(
                f'<p style="margin:0.3rem 0;font-size:14px;color:#0b0b0b;">'
                f"• {html.escape(q)}</p>",
                unsafe_allow_html=True,
            )
        def _on_gate_proceed_click() -> None:
            st.session_state.gate_should_proceed = True
            st.session_state.gate_result = None
            st.session_state.gate_questions = None

        st.button(
            "진단 계속하기",
            key="pc_gate_proceed",
            type="primary",
            on_click=_on_gate_proceed_click,
        )


def init_session() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "rediagnose_prompt" not in st.session_state:
        st.session_state.rediagnose_prompt = ""
    if "last_snapshot" not in st.session_state:
        st.session_state.last_snapshot = None
    if "auto_diagnose" not in st.session_state:
        st.session_state.auto_diagnose = False
    if "lc_chat_history" not in st.session_state:
        st.session_state.lc_chat_history = InMemoryChatMessageHistory()
    if "notion_save_status" not in st.session_state:
        st.session_state.notion_save_status = None  # None | "success" | "error"
    if "notion_save_error" not in st.session_state:
        st.session_state.notion_save_error = None
    if "rediagnose_context" not in st.session_state:
        st.session_state.rediagnose_context = None
    if "rediagnose_prefill_pending" not in st.session_state:
        st.session_state.rediagnose_prefill_pending = False
    if "original_prompt" not in st.session_state:
        st.session_state.original_prompt = ""
    if "data_consent" not in st.session_state:
        st.session_state.data_consent = False
    if "domain_result" not in st.session_state:
        st.session_state.domain_result = None
    if "gate_result" not in st.session_state:
        st.session_state.gate_result = None
    if "gate_questions" not in st.session_state:
        st.session_state.gate_questions = None
    if "gate_pending_diagnosis" not in st.session_state:
        st.session_state.gate_pending_diagnosis = None
    if "gate_should_proceed" not in st.session_state:
        st.session_state.gate_should_proceed = False


def _save_to_notion_with_retry(snapshot: dict[str, Any]) -> str:
    """Notion 저장 1회 재시도. 성공 시 URL 반환, 실패 시 예외 raise."""
    from utils.notion import save_diagnosis_page

    last_err: Exception | None = None
    for attempt in range(2):
        try:
            return save_diagnosis_page(snapshot)
        except Exception as e:
            last_err = e
            if attempt == 0:
                time.sleep(1)
    assert last_err is not None
    raise last_err


def _run_diagnosis(
    prompt_name: str,
    purpose: str,
    output_format: str,
    improvement_goals: list[str],
    text: str,
    auto_trigger: bool,
    phase_slot: Any = None,
) -> None:
    """LLM 체인 실행 및 결과를 session_state에 저장."""
    st.session_state.gate_result = None
    st.session_state.gate_questions = None
    st.session_state.gate_pending_diagnosis = None
    st.session_state.notion_save_status = None
    # F-15-3: 진단 실행 자체를 학습 데이터 활용 동의로 간주
    st.session_state.data_consent = True
    # F-23-1: 최초 진단 시에만 원본 프롬프트 잠금 (재진단·재시도에서 덮어쓰기 방지)
    if not st.session_state.get("original_prompt"):
        st.session_state.original_prompt = text
    routing = read_routing_config()
    temperature = routing.temperature
    llm = make_openai_llm(routing.openai_diagnosis_model, temperature)
    context_r, diagnosis_r, rewrite_r = build_chain_segments(llm)
    base_input: dict[str, Any] = {
        "purpose": purpose,
        "output_format": output_format,
        "improvement_goals": list(improvement_goals),
        "user_prompt": text,
    }
    try:
        if phase_slot is None:
            phase_slot = st.empty()

        def _set_phase(msg: str) -> None:
            phase_slot.markdown(_pc_loading_bar(msg), unsafe_allow_html=True)

        _set_phase("🔍 맥락 충분성 분석 중...")
        context_profile = invoke_with_retry(
            context_r.invoke,
            base_input,
            on_retry=_make_retry_phase_cb(phase_slot, "맥락 분석"),
        )
        # F-25-1: 도메인 2축 분류 결과를 session_state에 저장 (UI 노출 없음)
        st.session_state.domain_result = {
            "domain_action": (context_profile or {}).get("domain_action", ""),
            "domain_knowledge": (context_profile or {}).get("domain_knowledge", ""),
            "confidence_action": float((context_profile or {}).get("confidence_action", 0.0)),
            "confidence_knowledge": float((context_profile or {}).get("confidence_knowledge", 0.0)),
        }
        merged = {**base_input, "context_profile": context_profile}
        _set_phase("📊 진단 중...")
        diagnosis = invoke_with_retry(
            diagnosis_r.invoke,
            merged,
            on_retry=_make_retry_phase_cb(phase_slot, "진단"),
        )
        weighted = apply_goal_weights(diagnosis, improvement_goals)
        _loop_history: list[dict[str, Any]] = []
        if routing.self_improve_enabled:
            _set_phase("✏️ 개선안 생성 중...")
            rewrite_openai_llm = build_openai_rewrite_llm(routing)
            rewrite_opus_llm = build_opus_llm(routing)
            _, _, rewrite_r_openai = build_chain_segments(rewrite_openai_llm)
            rewrite_r_opus = None
            if rewrite_opus_llm is not None:
                _, _, rewrite_r_opus = build_chain_segments(rewrite_opus_llm)

            _LOOP_MESSAGES = [
                "더 나은 프롬프트를 만들고 있습니다",
                "개선 방향을 다듬고 있습니다",
                "최적의 구조를 찾고 있습니다",
                "고성능 모델로 검증하고 있습니다",
                "결과를 검토하고 있습니다",
                "최종 결과를 선택하고 있습니다",
            ]
            _initial_score = int(weighted.get("total_score") or 0)
            _prev_score = [_initial_score]
            _curr_score = [_initial_score]

            def _on_loop_iteration(
                iter_no: int, max_iters: int, phase: str, score: int = 0
            ) -> None:
                if score > 0:
                    _curr_score[0] = score
                msg_idx = min(iter_no - 1, len(_LOOP_MESSAGES) - 1)
                label = (
                    f"🤔 {_prev_score[0]}점 → {_curr_score[0]}점 : {_LOOP_MESSAGES[msg_idx]}..."
                )
                if phase_slot is not None:
                    phase_slot.markdown(_pc_loading_bar(label), unsafe_allow_html=True)
                if score > 0:
                    _prev_score[0] = _curr_score[0]

            loop_result = invoke_with_retry(
                run_self_improve_loop,
                base_input=base_input,
                context_profile=context_profile,
                diagnosis_r=diagnosis_r,
                rewrite_r_openai=rewrite_r_openai,
                rewrite_r_opus=rewrite_r_opus,
                routing=routing,
                max_iters=routing.self_improve_max_iterations,
                invoke_with_retry_fn=invoke_with_retry,
                on_iteration=_on_loop_iteration,
                on_retry=_make_retry_phase_cb(phase_slot, "자가개선"),
            )
            _loop_history = list(loop_result.get("history") or [])
            best = loop_result.get("best") or {}
            rewrite = best.get("rewrite") or {}
            weighted = best.get("weighted") or weighted
            diagnosis = best.get("diagnosis_raw") or diagnosis
        else:
            merged = {**merged, "diagnosis": diagnosis}
            _set_phase("✏️ 개선안 생성 중...")
            rewrite = invoke_with_retry(
                rewrite_r.invoke,
                merged,
                on_retry=_make_retry_phase_cb(phase_slot, "개선안 생성"),
            )

        improved = str(rewrite.get("improved_prompt", ""))

        # F-23-2: 의도 드리프트 점수 산출 (UI 노출 없음, jsonl 내부 지표)
        _drift_score = 0.0
        _original_for_drift = str(st.session_state.get("original_prompt") or "")
        if _original_for_drift and improved:
            try:
                _drift_chain = build_drift_chain(llm)
                _drift_raw = invoke_with_retry(
                    _drift_chain.invoke,
                    prep_drift_input(_original_for_drift, improved),
                )
                _drift_score = compute_drift_score(_drift_raw)
            except Exception:
                pass

        st.session_state.last_snapshot = {
            "ts": datetime.now(),
            "prompt_name": prompt_name,
            "purpose": purpose,
            "output_format": output_format,
            "improvement_goals": list(improvement_goals),
            "user_prompt": text,
            "original_prompt": st.session_state.original_prompt,
            "domain_result": st.session_state.domain_result,
            "context_profile": context_profile,
            "diagnosis_raw": diagnosis,
            "weighted": weighted,
            "rewrite": rewrite,
            "drift_score": _drift_score,
            "loop_history": _loop_history,
        }

        notion_ready = bool(
            os.environ.get("NOTION_API_KEY") and os.environ.get("NOTION_DB_ID")
        )
        st.session_state.notion_save_status = None
        st.session_state.notion_save_error = None

        if notion_ready:
            try:
                _save_to_notion_with_retry(st.session_state.last_snapshot)
                st.session_state.notion_save_status = "success"
            except Exception as e:
                st.session_state.notion_save_status = "error"
                st.session_state.notion_save_error = f"{type(e).__name__}: {e}"

        sync_learning_data(st.session_state.last_snapshot)
        st.session_state.history.append(st.session_state.last_snapshot)
        st.session_state.lc_chat_history.add_user_message(text[:300])
        st.session_state.lc_chat_history.add_ai_message(improved[:300])
        st.session_state.pop("pc_pending_diagnosis", None)
        if phase_slot is not None:
            phase_slot.empty()
        if auto_trigger:
            st.session_state.auto_diagnose = False
    except Exception as e:
        if phase_slot is not None:
            phase_slot.empty()
        st.markdown(_pc_phase_banner("⚠️ API 오류"), unsafe_allow_html=True)
        st.caption(
            "재시도에 실패했습니다. 잠시 후 다시 시도해주세요. "
            "API 호출 중 오류가 발생했습니다."
        )
        with st.expander("오류 상세", expanded=False):
            st.code(f"{type(e).__name__}: {e}")
        if st.button("재시도", key="pc_diagnosis_api_retry", type="primary"):
            st.session_state.pc_manual_retry_diagnosis = True
            st.rerun()
        if auto_trigger:
            st.session_state.auto_diagnose = False


def _render_results_panel(snap: dict[str, Any]) -> bool:
    """진단 결과 + 개선 결과 패널 렌더링.

    Returns:
        sync_prompt_from_widget: 입력 위젯과 스냅샷 동기화 여부(항상 True).
    """
    sync = True
    weighted = snap["weighted"]
    improved = str(snap["rewrite"].get("improved_prompt", ""))
    changes = list(snap["rewrite"].get("changes") or [])
    original = snap["user_prompt"]
    crit_keys = ["clarity", "constraint", "output_format", "context"]
    bonus_map = weighted.get("bonus") or {k: 0 for k in crit_keys}

    st.markdown('<p class="pc-card-title">진단 결과</p>', unsafe_allow_html=True)
    _render_diagnosis_score_cards(weighted, crit_keys, bonus_map)
    grade_label, grade_bg, grade_fg = _figma_grade_badge(str(weighted.get("grade") or ""))
    grade_emoji = _grade_emoji_label(str(weighted.get("grade") or ""))
    st.markdown(
        f"""
<div class="pc-total-row">
  <span class="pc-total-prefix">종합점수 :</span>
  <span class="pc-total-text">{weighted['total_score']} <small>/ 100</small></span>
  <span class="pc-pill" style="background:{grade_bg};color:{grade_fg};">{grade_emoji}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("#### 항목별 개선포인트")
    for key in crit_keys:
        b_pts = int(bonus_map.get(key, 0))
        final_sc = weighted["weighted_scores"][key]
        icon = CRITERION_ICONS.get(key, "•")
        exp_title = criterion_expander_title(
            f"{icon} {CRITERION_LABELS[key]}", final_sc, b_pts
        )
        with st.expander(exp_title, expanded=True):
            reason_text = str(weighted["reasons"].get(key, "") or "").strip()
            lines = [ln.strip(" -•\t") for ln in reason_text.splitlines() if ln.strip()]
            if not lines and reason_text:
                lines = [reason_text]
            if not lines:
                st.write("-")
            else:
                st.markdown("\n".join([f"- {html.escape(ln)}" for ln in lines]), unsafe_allow_html=True)

    st.markdown(
        '<p style="font-size:18px;font-weight:700;color:#0B0B0B;margin:0.95rem 0 0.5rem 0;">'
        "개선결과</p>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<p style="margin:0 0 0.35rem 0;color:{UI_DOVE_GRAY};font-weight:500;'
            'font-size:14px;line-height:20px;">Before</p>',
            unsafe_allow_html=True,
        )
        _render_prompt_readonly_box(original, variant="before")
    with c2:
        st.markdown(
            f'<p style="margin:0 0 0.35rem 0;color:{UI_PRIMARY_BLUE};font-weight:500;'
            'font-size:14px;line-height:20px;">After</p>',
            unsafe_allow_html=True,
        )
        _render_prompt_readonly_box(improved, variant="after")
        # 반응형에서 잘리지 않도록 전체 너비 컨테이너 우측 정렬.
        copy_id = f"pc_copy_{abs(hash(improved)) % 10_000_000}"
        render_clipboard_copy_button(
            improved, "📋 프롬프트 복사", copy_id, primary=False
        )

    _render_improvement_bullets(changes)

    prompt_name = str(snap.get("prompt_name") or "prompt_clinic")
    fn = f"prompt_clinic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    md_body = build_markdown_report(
        prompt_name,
        snap["purpose"],
        snap["output_format"],
        snap["improvement_goals"],
        original,
        weighted,
        improved,
        changes,
    )

    fn_pdf = f"{prompt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    fn_obs = f"{prompt_name}_obsidian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    domain_result = snap.get("domain_result") or {}
    _d_action = str(domain_result.get("domain_action") or "")
    _d_knowledge = str(domain_result.get("domain_knowledge") or "")
    obs_body = build_obsidian_report(
        prompt_name,
        snap["purpose"],
        snap["output_format"],
        snap["improvement_goals"],
        original,
        weighted,
        improved,
        changes,
        domain_action=_d_action,
        domain_knowledge=_d_knowledge,
    )

    _btn_col1, _btn_col2 = st.columns(2)
    with _btn_col1:
        try:
            pdf_bytes = build_pdf_report(
                prompt_name,
                snap["purpose"],
                snap["output_format"],
                snap["improvement_goals"],
                original,
                weighted,
                improved,
                changes,
            )
            st.download_button(
                "⬇️ 리포트 다운로드 (.pdf)",
                data=pdf_bytes,
                file_name=fn_pdf,
                mime="application/pdf",
                type="secondary",
                use_container_width=True,
                key="download_report_pdf",
            )
        except Exception as _pdf_err:
            st.warning(f"PDF 생성 실패: {_pdf_err}")
            st.download_button(
                "⬇️ 리포트 다운로드 (.md)",
                data=md_body,
                file_name=fn,
                mime="text/markdown",
                type="secondary",
                use_container_width=True,
                key="download_report_md_fallback",
            )
    with _btn_col2:
        st.download_button(
            "⬇️ 프롬프트 아카이빙 (.md)",
            data=obs_body,
            file_name=fn_obs,
            mime="text/markdown",
            type="secondary",
            use_container_width=True,
            key="download_obsidian_md",
        )

    return sync


def _render_loop_history_cards() -> None:
    """F-13-3: 자가개선 반복 이력 카드 (SELF_IMPROVE_ENABLED=true 시 표시)."""
    snap = st.session_state.get("last_snapshot")
    if not snap:
        return
    loop_history = list(snap.get("loop_history") or [])
    if not loop_history:
        return
    with st.expander("🔄 개선 반복 이력", expanded=False):
        for record in loop_history:
            iter_no = int(record.get("iteration") or 0)
            weighted_r = record.get("weighted") or {}
            total_score = int(weighted_r.get("total_score") or 0)
            grade = str(weighted_r.get("grade") or "")
            changes = list((record.get("rewrite") or {}).get("changes") or [])
            model_key = str(record.get("rewrite_model_key") or "openai")
            grade_label, grade_bg, grade_fg = _figma_grade_badge(grade)
            changes_html = "".join(
                f'<p style="margin:4px 0 0 0;font-size:12px;color:#374151;">'
                f"• {html.escape(str(ch))}</p>"
                for ch in changes[:2]
            )
            st.markdown(
                f'<div style="border:1px solid #DEDEDE;border-radius:8px;background:#F6F6F6;'
                f'padding:10px 14px;margin:0 0 8px 0;">'
                f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
                f'<span style="font-size:13px;font-weight:700;color:#0B0B0B;">Iter {iter_no}</span>'
                f'<span style="font-size:13px;font-weight:700;color:#0B0B0B;">{total_score} / 100</span>'
                f'<span style="background:{grade_bg};color:{grade_fg};border-radius:6px;'
                f'padding:1px 8px;font-size:11px;font-weight:600;">{html.escape(grade_label)}</span>'
                f'<span style="font-size:11px;color:#6B7280;">{html.escape(model_key)}</span>'
                f"</div>"
                f"{changes_html}"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_history_panel() -> None:
    """세션 히스토리 패널 렌더링 (최신순·카드·펼침)."""
    hist: list[Any] = st.session_state.history
    if not hist:
        return
    st.markdown(
        '<p style="font-size:18px;font-weight:700;color:#0B0B0B;margin:0.35rem 0 0.55rem 0;">진단기록</p>',
        unsafe_allow_html=True,
    )
    _render_history_entries(hist)


def _hist_expander_body_html(orig: str, impr: str) -> tuple[str, str]:
    """진단기록 펼침 영역: 원본/개선 블록 HTML (짧은 줄 유지)."""
    orig_s = _first_two_lines_ellipsis(orig)
    impr_s = _first_two_lines_ellipsis(impr)
    p_orig = (
        f'<p style="margin:0.2rem 0 0.1rem 0;color:#111827;'
        f'font-size:12px;line-height:1.35;">• <strong>원본 :</strong> {html.escape(orig_s)}</p>'
    )
    p_imp = (
        f'<p style="margin:0.2rem 0 0.1rem 0;color:#111827;'
        f'font-size:12px;line-height:1.35;">• <strong>개선 :</strong> {html.escape(impr_s)}</p>'
    )
    return p_orig, p_imp


def _render_history_entries(hist: list[Any]) -> None:
    items: list[str] = []
    latest_pos = 0
    for pos, idx in enumerate(range(len(hist) - 1, -1, -1)):
        h = hist[idx]
        ts: datetime = h["ts"]
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
        prompt_summary = _first_two_lines_ellipsis(
            str(h.get("user_prompt") or ""), max_chars=34
        )
        hw = h.get("weighted")
        total_s = "-"
        grade = ""
        if isinstance(hw, dict) and "total_score" in hw:
            total_s = str(hw.get("total_score"))
            grade = str(hw.get("grade") or "")
        fig_label, _bg, _fg = _figma_grade_badge(grade)
        badge_cls = "red"
        if fig_label == "즉시사용가능":
            badge_cls = "green"
        elif fig_label == "보완권장":
            badge_cls = "yellow"

        orig = _first_two_lines_ellipsis(str(h.get("user_prompt") or ""), max_chars=130)
        impr = _first_two_lines_ellipsis(
            str(h.get("rewrite", {}).get("improved_prompt", "")), max_chars=130
        )
        open_cls = " open" if pos == latest_pos else ""
        item_html = (
            f'<div class="pc-h-item{open_cls}" data-i="{pos}">'
            f'<div class="pc-h-head" data-t="{pos}">'
            f'<span class="pc-h-time">{html.escape(ts_str)}</span>'
            f'<span class="pc-h-sum">{html.escape(prompt_summary)}</span>'
            f'<span class="pc-h-score">{html.escape(total_s)} / 100</span>'
            f'<span class="pc-h-badge {badge_cls}">{html.escape(fig_label)}</span>'
            f'<button class="pc-h-toggle" type="button" data-b="{pos}" '
            f'aria-label="펼치기">{_HISTORY_ROW_CHEVRON_SVG}</button>'
            f"</div>"
            f'<div class="pc-h-body" data-c="{pos}">'
            f'<p>• <strong>원본 :</strong> {html.escape(orig)}</p>'
            f'<p>• <strong>개선 :</strong> {html.escape(impr)}</p>'
            f"</div>"
            f"</div>"
        )
        items.append(item_html)

    components.html(
        f"""
<style>
.pc-h-list {{ display:flex; flex-direction:column; gap:10px; }}
.pc-h-item {{ border:1px solid #DEDEDE; border-radius:8px; background:#F6F6F6; }}
.pc-h-head {{
  display:grid; grid-template-columns:170px 1fr 95px 100px 20px; gap:10px;
  align-items:center; padding:10px 12px; cursor:pointer;
}}
.pc-h-time {{ color:#6B7280; font-size:12px; white-space:nowrap; }}
.pc-h-sum {{ color:#111827; font-size:14px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.pc-h-score {{ color:#111827; font-size:14px; font-weight:700; text-align:right; white-space:nowrap; }}
.pc-h-badge {{
  display:inline-flex; align-items:center; justify-content:center;
  border-radius:6px; height:26px; padding:0 8px; font-size:11px; font-weight:600; color:#fff;
}}
.pc-h-badge.green {{ background:#01A701; }}
.pc-h-badge.yellow {{ background:#FFD700; }}
.pc-h-badge.red {{ background:#D40924; }}
.pc-h-toggle {{
  border:none; background:transparent; color:rgba(49,51,63,0.8); padding:0;
  width:18px; height:18px; cursor:pointer;
  display:inline-flex; align-items:center; justify-content:center;
  flex-shrink:0;
}}
.pc-h-chevron {{
  display:block;
  transition: transform 0.2s ease;
}}
.pc-h-body {{ display:none; padding:0 12px 10px 12px; }}
.pc-h-body p {{ margin:4px 0; font-size:12px; line-height:1.35; color:#111827; }}
.pc-h-item.open .pc-h-body {{ display:block; }}
.pc-h-item.open .pc-h-chevron {{ transform: rotate(180deg); }}
</style>
<div class="pc-h-list">
  {''.join(items)}
</div>
<script>
(function() {{
  const root = document;
  const heads = root.querySelectorAll(".pc-h-head");
  heads.forEach((head) => {{
    head.addEventListener("click", (e) => {{
      const idx = head.getAttribute("data-t");
      const item = root.querySelector(`.pc-h-item[data-i="${{idx}}"]`);
      if (!item) return;
      item.classList.toggle("open");
      e.stopPropagation();
    }});
  }});
  const btns = root.querySelectorAll(".pc-h-toggle");
  btns.forEach((b) => b.addEventListener("click", (e) => {{
    const idx = b.getAttribute("data-b");
    const item = root.querySelector(`.pc-h-item[data-i="${{idx}}"]`);
    if (item) item.classList.toggle("open");
    e.stopPropagation();
  }}));
}})();
</script>
        """,
        height=max(120, 78 * len(items) + 70),
    )


def main() -> None:
    init_session()
    st.set_page_config(
        page_title="Prompt Clinic",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _inject_pc_theme_css()
    st.markdown(
        """
<style>
span[data-baseweb="tag"] {
    background-color: #1d6fa4 !important;
    color: white !important;
}
</style>
""",
        unsafe_allow_html=True,
    )
    if st.session_state.get("rediagnose_prefill_pending"):
        ctx = st.session_state.get("rediagnose_context") or {}
        st.session_state["prompt_name_input"] = str(ctx.get("prompt_name") or "")
        st.session_state["purpose_input"] = str(ctx.get("purpose") or "")
        fmt = _normalize_output_format_option(str(ctx.get("output_format") or ""))
        if not fmt or fmt not in OUTPUT_FORMAT_OPTIONS:
            fmt = OUTPUT_FORMAT_OPTIONS[0]
        st.session_state["output_format_input"] = fmt
        st.session_state["improvement_goals_input"] = list(
            ctx.get("improvement_goals") or []
        )
        st.session_state["user_prompt_input"] = str(
            st.session_state.get("rediagnose_prompt") or ""
        )
        st.session_state.rediagnose_prefill_pending = False

    st.markdown(
        f"""
<div class="pc-wire-hero">
  <div class="pc-wire-hero-row">
    <div class="pc-wire-brand-mark">{_brand_logo_html()}</div>
    <div class="pc-wire-brand-text">
      <h1 class="pc-wire-title">Prompt Clinic</h1>
      <p class="pc-wire-desc">프롬프트를 진단하고, 개선합니다.</p>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    auto_trigger = st.session_state.get("auto_diagnose", False)
    if auto_trigger:
        st.info("💡 개선된 프롬프트로 재진단을 시작합니다...")

    with st.container(border=True, key="pc_input_shell"):
        st.markdown(
            """
            <div class="pc-section-head-inline">
                <span class="pc-wire-section">맥락 수집</span>
                <span class="pc-section-required-text">모든 항목은 필수입력값입니다.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key="pc_name_format_row"):
            col_name, col_fmt = st.columns([1.85, 1], vertical_alignment="top")

            with col_name:
                st.markdown(
                    """
                    <div class="pc-label-row pc-label-row-single">
                        <span class="pc-wire-muted pc-context-field-label">프롬프트 명 (20자 이하)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.container(key="prompt_name_field"):
                    prompt_name = st.text_input(
                        "프롬프트명",
                        placeholder="예 : AI 챗봇",
                        key="prompt_name_input",
                        label_visibility="collapsed",
                    )

                components.html(
                    """
                    <script>
                    (function() {
                    const doc = window.parent.document;

                    function mountPromptNameCounter() {
                        const wrapper = doc.querySelector('.st-key-prompt_name_field');
                        if (!wrapper) return false;

                        const inputShell = wrapper.querySelector('[data-testid="stTextInput"]');
                        if (!inputShell) return false;

                        const input = inputShell.querySelector('input');
                        if (!input) return false;

                        inputShell.classList.add('pc-input-counter-target');

                        let counter = inputShell.querySelector('.pc-input-counter');
                        if (!counter) {
                        counter = doc.createElement('div');
                        counter.className = 'pc-input-counter';
                        inputShell.appendChild(counter);
                        }

                        function renderCount() {
                        counter.textContent = `${input.value.length} / 20`;
                        }

                        if (!input.dataset.pcPromptCounterBound) {
                        input.addEventListener('input', renderCount);
                        input.dataset.pcPromptCounterBound = '1';
                        }

                        renderCount();
                        return true;
                    }

                    let tries = 0;
                    const timer = setInterval(() => {
                        const ok = mountPromptNameCounter();
                        tries += 1;
                        if (ok || tries > 40) clearInterval(timer);
                    }, 150);
                    })();
                    </script>
                    """,
                    height=0,
                )

            with col_fmt:
                st.markdown(
                    '<span class="pc-wire-label-strong pc-context-field-label">'
                    "출력 형식</span>",
                    unsafe_allow_html=True,
                )
                with st.container(key="output_format_field"):
                    output_format = st.selectbox(
                        "output_format_internal",
                        OUTPUT_FORMAT_OPTIONS,
                        key="output_format_input",
                        label_visibility="collapsed",
                    )

        st.markdown(
            '<span class="pc-wire-muted pc-context-field-label">'
            "프롬프트 사용목적 (100자 이하)</span>",
            unsafe_allow_html=True,
        )
        with st.container(key="purpose_field"):
            purpose = st.text_area(
                "프롬프트 사용목적",
                placeholder="예 : AI 챗봇 생성을 위한 프롬프트 작성",
                key="purpose_input",
                label_visibility="collapsed",
                height=44,
            )

        inject_live_counter(container_key="purpose_field", limit=100)

        if len((purpose or "")) > 100:
            st.markdown(
                '<p class="pc-inline-err">100자 이하로 입력해주세요.</p>',
                unsafe_allow_html=True,
            )

        goals_err = len(st.session_state.get("improvement_goals_input") or []) == 0
        st.markdown(
            '<div class="pc-label-row">'
            '<span class="pc-wire-muted" style="margin:0;">개선 포인트 (최대 3개 선택 가능)</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        improvement_goals = _render_improvement_point_buttons()

        _up_prev = str(st.session_state.get("user_prompt_input") or "")
        _prompt_len_err_label = len(_up_prev) > 500


        _service_url = os.environ.get(
            "SERVICE_POLICY_URL",
            "https://www.notion.so/Prompt-Clinic-34340cb3731d80edb1cbefcf197078d7",
        )
        _privacy_url = os.environ.get(
            "PRIVACY_POLICY_URL",
            "https://www.notion.so/Prompt-Clinic-34340cb3731d80eeb2d8cad538a3fe67",
        )

        with st.container(key="user_prompt_field"):
            st.markdown(
                '<div class="pc-label-row">'
                '<span class="pc-wire-label-strong" style="margin:0;">'
                "진단할 프롬프트 (500자 이하)</span>"
                + (
                    '<span class="pc-inline-err">500자 이하로 작성해주세요.</span>'
                    if _prompt_len_err_label
                    else ""
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            user_prompt = st.text_area(
                "진단할 프롬프트",
                height=220,
                placeholder="예 : 너는 AI 챗봇이야. 사용자 질문에 잘 대답해줘. 친절하게 해줘.",
                key="user_prompt_input",
                label_visibility="collapsed",
            )
            st.markdown(
                f"""
                <div class="pc-policy-note">
                    <a href="{_service_url}" target="_blank" rel="noopener noreferrer">서비스이용정책</a>
                    및
                    <a href="{_privacy_url}" target="_blank" rel="noopener noreferrer">개인정보처리방침</a>
                    에 따라 입력하신 프롬프트는 서비스 품질 개선을 위한 학습 데이터로 활용될 수 있습니다.
                </div>
                """,
                unsafe_allow_html=True,
            )

        inject_live_counter(container_key="user_prompt_field", limit=500)

        p_len = len((purpose or ""))
        t_len = len((user_prompt or ""))
        pn = (prompt_name or "").strip()

        purpose_ok = bool((purpose or "").strip()) and p_len <= 100
        name_ok = (
            bool(pn)
            and len(pn) <= 20
            and bool(PROMPT_NAME_PATTERN.fullmatch(pn))
        )
        prompt_ok = bool((user_prompt or "").strip()) and t_len <= 500
        goals_ok = len(improvement_goals) > 0
        key_ok = bool(os.environ.get("OPENAI_API_KEY"))
        form_ready = name_ok and purpose_ok and prompt_ok and goals_ok and key_ok

        err_purpose_border = p_len > 100
        err_prompt_border = t_len > 500
        st.markdown(
            f"""
    <style>
    .st-key-purpose_field div[data-baseweb="input"] input,
    .st-key-purpose_field div[data-baseweb="textarea"] textarea {{
    border-color: {'#d40924' if err_purpose_border else '#DEDEDE'} !important;
    }}
    .st-key-user_prompt_field div[data-baseweb="textarea"] textarea {{
    border-color: {'#d40924' if err_prompt_border else '#DEDEDE'} !important;
    }}
    </style>
    """,
            unsafe_allow_html=True,
        )

        _left_spacer, run_col = st.columns([4.2, 1])

        with _left_spacer:
            st.empty()

        with run_col:
            is_enabled = form_ready or auto_trigger

            run = st.button(
                "🩺 진단 시작",
                type="primary" if is_enabled else "secondary",
                use_container_width=True,
                key="pc_run_diagnosis",
                disabled=not is_enabled,
            )
    sync_prompt_from_widget = True

    _loading_slot = st.empty()  # 로딩 바 슬롯: pc_input_shell 카드 하단 외부

    if st.session_state.get("pc_manual_retry_diagnosis"):
        st.session_state.pc_manual_retry_diagnosis = False
        pending = st.session_state.get("pc_pending_diagnosis")
        if pending:
            _run_diagnosis(
                str(pending["prompt_name"]),
                str(pending["purpose"]),
                str(pending["output_format"]),
                list(pending["improvement_goals"]),
                str(pending["text"]),
                False,
                phase_slot=_loading_slot,
            )

    # F-20: "진단 계속하기" 클릭 시 현재 위젯 값으로 진단 실행
    _skip_run_block = False
    if st.session_state.get("gate_should_proceed"):
        st.session_state.gate_should_proceed = False
        _gpending = st.session_state.get("gate_pending_diagnosis")
        # gate_result/questions는 _gpending 존재 여부와 무관하게 항상 클리어
        st.session_state.gate_result = None
        st.session_state.gate_questions = None
        st.session_state.gate_pending_diagnosis = None
        if _gpending:
            _gp_purpose = str(st.session_state.get("purpose_input") or _gpending["purpose"])
            _gp_text = str(st.session_state.get("user_prompt_input") or _gpending["text"])
            _gp_goals = list(
                st.session_state.get("improvement_goals_input") or _gpending["improvement_goals"]
            )
            st.session_state["pc_pending_diagnosis"] = {
                "prompt_name": _gpending["prompt_name"],
                "purpose": _gp_purpose,
                "output_format": _gpending["output_format"],
                "improvement_goals": _gp_goals,
                "text": _gp_text,
                "auto_trigger": _gpending.get("auto_trigger", False),
            }
            _run_diagnosis(
                str(_gpending["prompt_name"]),
                _gp_purpose,
                str(_gpending["output_format"]),
                _gp_goals,
                _gp_text,
                bool(_gpending.get("auto_trigger", False)),
                phase_slot=_loading_slot,
            )
        _skip_run_block = True

    if (run or auto_trigger) and not _skip_run_block:
        text = (user_prompt or "").strip()
        prompt_name_text = (prompt_name or "").strip()
        purpose_text = (purpose or "").strip()
        if not prompt_name_text:
            st.error("프롬프트 명을 입력해 주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif len(prompt_name_text) > 20:
            st.error("프롬프트 명은 20자 이하로 입력해 주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif not PROMPT_NAME_PATTERN.fullmatch(prompt_name_text):
            st.error(
                "프롬프트 명은 영문/숫자/한글과 -, _, ., "
                "공백 만 입력할 수 있습니다."
            )
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif not purpose_text:
            st.error("프롬프트 사용목적을 입력해 주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif len(purpose_text) > 100:
            st.error("100자 이하로 입력해주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif not text:
            st.error("프롬프트를 입력해 주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif len(text) > 500:
            st.error("500자 이하로 작성해주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif not improvement_goals:
            st.error("1개 이상 선택해주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif not os.environ.get("OPENAI_API_KEY"):
            st.error(".env에 OPENAI_API_KEY를 설정해 주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        else:
            # F-20-1: 맥락 모호성 게이트 분석
            _routing = read_routing_config()
            _gate_llm = make_openai_llm(_routing.openai_diagnosis_model, _routing.temperature)
            _loading_slot.markdown(
                _pc_loading_bar("🔍 맥락 충분성 분석 중..."), unsafe_allow_html=True
            )
            try:
                _gate_chain = build_gate_chain(_gate_llm)
                _gate_score: dict[str, Any] = invoke_with_retry(
                    _gate_chain.invoke,
                    prep_gate_input(purpose_text, text, improvement_goals),
                    on_retry=_make_retry_phase_cb(_loading_slot, "맥락 분석"),
                )
            except Exception:
                _gate_score = {
                    "goal_ambiguity": 0.0,
                    "constraint_ambiguity": 0.0,
                    "success_ambiguity": 0.0,
                    "weak_axes": [],
                }
            finally:
                _loading_slot.empty()

            _gate_total = compute_gate_total_score(_gate_score)
            _gate_score["total_score"] = _gate_total

            if _gate_total <= 0.5:
                # 게이트 통과: 바로 진단 실행
                st.session_state.gate_result = None
                st.session_state["pc_pending_diagnosis"] = {
                    "prompt_name": prompt_name_text,
                    "purpose": purpose_text,
                    "output_format": output_format,
                    "improvement_goals": list(improvement_goals),
                    "text": text,
                    "auto_trigger": auto_trigger,
                }
                _run_diagnosis(
                    prompt_name_text,
                    purpose_text,
                    output_format,
                    improvement_goals,
                    text,
                    auto_trigger,
                    phase_slot=_loading_slot,
                )
            else:
                # 게이트 경고: 질문 생성 후 배너+expander 표시
                st.session_state.gate_result = _gate_score
                _loading_slot.markdown(
                    _pc_loading_bar("💬 보완 질문 생성 중..."), unsafe_allow_html=True
                )
                try:
                    _q_chain = build_question_chain(_gate_llm)
                    _gate_questions: dict[str, Any] = invoke_with_retry(
                        _q_chain.invoke,
                        prep_question_input(
                            purpose_text,
                            text,
                            list(_gate_score.get("weak_axes") or []),
                        ),
                        on_retry=_make_retry_phase_cb(_loading_slot, "질문 생성"),
                    )
                except Exception:
                    _gate_questions = {"questions": []}
                finally:
                    _loading_slot.empty()

                st.session_state.gate_questions = _gate_questions
                st.session_state.gate_pending_diagnosis = {
                    "prompt_name": prompt_name_text,
                    "purpose": purpose_text,
                    "output_format": output_format,
                    "improvement_goals": list(improvement_goals),
                    "text": text,
                    "auto_trigger": auto_trigger,
                }

    # F-20-2/3: 게이트 배너 + 보완 질문 expander
    _render_gate_ui()

    snap = st.session_state.last_snapshot
    if snap:
        st.markdown(_pc_phase_banner("✅ 진단 완료!"), unsafe_allow_html=True)
        sync_prompt_from_widget = _render_results_panel(snap)
        if os.environ.get("SELF_IMPROVE_ENABLED", "false").lower() == "true":
            _render_loop_history_cards()

    _render_history_panel()

    components.html(
        """
        <script>
        (function () {
          const doc = (window.parent && window.parent.document && window.parent.document.body)
            ? window.parent.document
            : document;
          let btn = doc.getElementById("pc-scroll-top-btn");
          if (!btn) {
            btn = doc.createElement("button");
            btn.id = "pc-scroll-top-btn";
            btn.setAttribute("aria-label", "상단 이동");
            btn.textContent = "↑";
            btn.style.cssText =
              "position:fixed;right:20px;bottom:24px;width:52px;height:52px;border-radius:999px;" +
              "border:1px solid #CFCFCF;background:#FFFFFF;color:#111827;font-size:26px;line-height:1;font-weight:700;" +
              "cursor:pointer;display:none;z-index:2147483000;box-shadow:0 8px 14px rgba(0,0,0,0.15);";
            doc.body.appendChild(btn);
            btn.addEventListener("click", function () {
              try {
                if (window.top) window.top.scrollTo({ top: 0, behavior: "smooth" });
              } catch (e) {}
            });
          }
          btn.textContent = "↑";
          const onScroll = function () {
            const y = (window.parent && window.parent.scrollY) || window.scrollY || 0;
            btn.style.display = y > 280 ? "block" : "none";
          };
          window.addEventListener("scroll", onScroll, { passive: true });
          if (window.parent && window.parent !== window) {
            window.parent.addEventListener("scroll", onScroll, { passive: true });
          }
          onScroll();
        })();
        </script>
        """,
        height=0,
    )

    _privacy_url = os.environ.get(
        "PRIVACY_POLICY_URL",
        "https://www.notion.so/Prompt-Clinic-34340cb3731d80eeb2d8cad538a3fe67",
    )
    _service_url = os.environ.get(
        "SERVICE_POLICY_URL",
        "https://www.notion.so/Prompt-Clinic-34340cb3731d80edb1cbefcf197078d7",
    )

    st.markdown(
        f"""
        <div class="pc-footer-wrap">
            <div class="pc-footer-links">
                <a href="{_service_url}" target="_blank" rel="noopener noreferrer">서비스이용정책</a>
                <span class="pc-footer-divider">|</span>
                <a href="{_privacy_url}" target="_blank" rel="noopener noreferrer">개인정보처리방침</a>
            </div>
            <div class="pc-footer-copy">© 2026 Team 토큰부족. All rights reserved.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    if sync_prompt_from_widget:
        st.session_state.rediagnose_prompt = user_prompt


if __name__ == "__main__":
    main()
