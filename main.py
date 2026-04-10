"""Streamlit: 프롬프트 진단 클리닉 메인 앱."""

from __future__ import annotations

import base64
import html
import os
import re
import time
from datetime import datetime
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
from chains.self_improve_chain import apply_goal_weights, run_self_improve_loop
from utils.data_pipeline import sync_learning_data

load_dotenv()

OUTPUT_FORMAT_OPTIONS = ["글", "리스트", "표", "코드", "JSON"]
IMPROVEMENT_OPTIONS = [
    "토큰 줄이기",
    "일관성 높이기",
    "출력 품질 높이기",
    "구조화",
    "맥락 보완",
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

# figma.html(와이어) 등급 배지 문구 — 앱 로직의 grade 문자열과 매핑만 함
FIGMA_GRADE_BADGE: dict[str, tuple[str, str, str]] = {
    "우수": ("즉시사용가능", UI_BADGE_GREEN, "#ffffff"),
    # 와이어: 노란 배지도 라벨은 흰색(가독성은 디자인 시안 기준)
    "보통": ("보완권장", UI_BADGE_YELLOW, "#ffffff"),
    "개선필요": ("재작성 필요", UI_BADGE_RED, "#ffffff"),
}

PROMPT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9가-힣._\- ]+$")



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


def _inject_pc_theme_css() -> None:
    """앱 본문 UI 토큰 스타일 (결과·히스토리·푸터 공통, 한 번 주입)."""
    st.markdown(
        f"""
<style>
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
.pc-footer {{
  text-align: center;
  color: {UI_DOVE_GRAY};
  font-size: 0.85rem;
  margin-top: 2rem;
  padding-top: 1rem;
}}
.block-container {{
  max-width: 860px !important;
  padding-top: 2.5rem !important;
  padding-bottom: 2rem !important;
}}
[data-testid="stSidebar"] {{
  display: block !important;
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


def _make_retry_cb(status: Any, step_label: str) -> Callable[[int, int], None]:
    """st.status 재시도 콜백 팩토리."""
    def on_retry(n: int, total: int) -> None:
        status.update(
            label=f"⚠️ {step_label} 재시도 중 ({n}/{total})...",
            state="running",
        )
    return on_retry


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
    if "rediagnose_context" not in st.session_state:
        st.session_state.rediagnose_context = None
    if "rediagnose_prefill_pending" not in st.session_state:
        st.session_state.rediagnose_prefill_pending = False


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
) -> None:
    """LLM 체인 실행 및 결과를 session_state에 저장."""
    st.session_state.notion_save_status = None
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
        with st.spinner("처리 중..."):
            context_profile = invoke_with_retry(context_r.invoke, base_input)
            merged = {**base_input, "context_profile": context_profile}
            diagnosis = invoke_with_retry(diagnosis_r.invoke, merged)
            weighted = apply_goal_weights(diagnosis, improvement_goals)
            if routing.self_improve_enabled:
                rewrite_openai_llm = build_openai_rewrite_llm(routing)
                rewrite_opus_llm = build_opus_llm(routing)
                _, _, rewrite_r_openai = build_chain_segments(rewrite_openai_llm)
                rewrite_r_opus = None
                if rewrite_opus_llm is not None:
                    _, _, rewrite_r_opus = build_chain_segments(rewrite_opus_llm)
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
                    on_retry=None,
                )
                best = loop_result.get("best") or {}
                rewrite = best.get("rewrite") or {}
                weighted = best.get("weighted") or weighted
                diagnosis = best.get("diagnosis_raw") or diagnosis
            else:
                merged = {**merged, "diagnosis": diagnosis}
                rewrite = invoke_with_retry(rewrite_r.invoke, merged)
        improved = str(rewrite.get("improved_prompt", ""))

        st.session_state.last_snapshot = {
            "ts": datetime.now(),
            "prompt_name": prompt_name,
            "purpose": purpose,
            "output_format": output_format,
            "improvement_goals": list(improvement_goals),
            "user_prompt": text,
            "context_profile": context_profile,
            "diagnosis_raw": diagnosis,
            "weighted": weighted,
            "rewrite": rewrite,
        }
        sync_learning_data(st.session_state.last_snapshot)
        st.session_state.history.append(st.session_state.last_snapshot)
        st.session_state.lc_chat_history.add_user_message(text[:300])
        st.session_state.lc_chat_history.add_ai_message(improved[:300])
        st.success("진단이 완료되었습니다. 아래에서 결과를 확인하세요.")
        if auto_trigger:
            st.session_state.auto_diagnose = False
    except Exception as e:
        st.error(
            "API 호출에 실패했습니다. 잠시 후 다시 시도해 주세요.\n\n"
            f"상세: `{type(e).__name__}: {e}`"
        )
        if auto_trigger:
            st.session_state.auto_diagnose = False


def _render_results_panel(snap: dict[str, Any]) -> bool:
    """진단 결과 + 개선 결과 패널 렌더링.

    Returns:
        sync_prompt_from_widget: 재진단 버튼이 클릭되면 False, 그 외 True.
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

    notion_ready = bool(
        os.environ.get("NOTION_API_KEY") and os.environ.get("NOTION_DB_ID")
    )
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.download_button(
            "⬇️ 리포트 다운로드 (.md)",
            data=md_body,
            file_name=fn,
            mime="text/markdown",
            type="primary",
            use_container_width=True,
            key="download_report_md",
        )
    with ac2:
        if notion_ready:
            if st.button(
                "🗂️ 프롬프트 아카이빙 (Notion)",
                type="primary",
                key="notion_save",
                use_container_width=True,
            ):
                try:
                    _save_to_notion_with_retry(snap)
                    st.session_state.notion_save_status = "success"
                    st.toast("✅ Notion에 자동 저장됐어요!")
                except Exception:
                    st.session_state.notion_save_status = "error"
                    st.toast("😢 Notion 저장에 실패했어요. 리포트를 다운로드해 보관하세요.")
        else:
            st.button(
                "🗂️ 프롬프트 아카이빙 (Notion)",
                type="primary",
                disabled=True,
                use_container_width=True,
                key="notion_save_disabled",
                help=(
                    "NOTION_API_KEY와 NOTION_DB_ID를 .env에 설정하면 "
                    "사용할 수 있어요."
                ),
            )
    with ac3:
        if st.button(
            "개선된 프롬프트로 재진단",
            type="primary",
            key="redo_diagnose",
            use_container_width=True,
        ):
            st.session_state.rediagnose_prompt = improved
            st.session_state.rediagnose_context = {
                "prompt_name": str(snap.get("prompt_name") or ""),
                "purpose": str(snap.get("purpose") or ""),
                "output_format": str(snap.get("output_format") or OUTPUT_FORMAT_OPTIONS[0]),
                "improvement_goals": list(snap.get("improvement_goals") or []),
            }
            st.session_state.rediagnose_prefill_pending = True
            st.session_state.auto_diagnose = True
            sync = False
            st.rerun()

    if st.session_state.get("notion_save_status") == "error":
        st.caption(
            "Notion 저장에 실패했을 때는 위 **리포트 다운로드**로 "
            "동일 내용을 저장할 수 있어요."
        )
    return sync


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
    st.set_page_config(page_title="Prompt Clinic", page_icon="🩺", layout="wide")
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
        fmt = str(ctx.get("output_format") or OUTPUT_FORMAT_OPTIONS[0])
        st.session_state["output_format_input"] = (
            fmt if fmt in OUTPUT_FORMAT_OPTIONS else OUTPUT_FORMAT_OPTIONS[0]
        )
        st.session_state["improvement_goals_input"] = list(
            ctx.get("improvement_goals") or []
        )
        st.session_state["user_prompt_input"] = str(
            st.session_state.get("rediagnose_prompt") or ""
        )
        st.session_state.rediagnose_prefill_pending = False
    st.title("🩺 Prompt Clinic")
    st.caption("프롬프트를 진단하고 개선안을 제안합니다.")
    with st.sidebar:
        st.header("맥락 수집")
        prompt_name = st.text_input(
            "프롬프트 명",
            placeholder="예: prompt_v1",
            key="prompt_name_input",
        )
        st.caption("추후 프롬프트 다운로드 및 저장 시, 파일명으로 사용됩니다.")
        purpose = st.text_area(
            "프롬프트 사용목적",
            placeholder="이 프롬프트를 어디에 사용하는지 적어주세요.",
            height=100,
            key="purpose_input",
        )
        output_format = st.selectbox(
            "출력 형식",
            OUTPUT_FORMAT_OPTIONS,
            key="output_format_input",
        )
        improvement_goals = st.multiselect(
            "개선 목적",
            IMPROVEMENT_OPTIONS,
            key="improvement_goals_input",
        )

    st.subheader("프롬프트 입력")
    user_prompt = st.text_area(
        "진단할 프롬프트",
        height=220,
        placeholder="프롬프트를 입력하세요...",
        key="user_prompt_input",
    )

    auto_trigger = st.session_state.get("auto_diagnose", False)
    if auto_trigger:
        st.info("💡 개선된 프롬프트로 재진단을 시작합니다...")

    _spacer, _run_col = st.columns([7.2, 1.4])
    with _run_col:
        run = st.button("진단 시작", type="primary", use_container_width=True)
    sync_prompt_from_widget = True

    if run or auto_trigger:
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
            st.error("프롬프트 사용목적은 100자 이하로 입력해 주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif not text:
            st.error("프롬프트를 입력해 주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif len(text) > 500:
            st.error("프롬프트는 500자 이하로 입력해 주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif not improvement_goals:
            st.error("개선 목적을 하나 이상 선택해 주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        elif not os.environ.get("OPENAI_API_KEY"):
            st.error(".env에 OPENAI_API_KEY를 설정해 주세요.")
            if auto_trigger:
                st.session_state.auto_diagnose = False
        else:
            _run_diagnosis(
                prompt_name_text,
                purpose_text,
                output_format,
                improvement_goals,
                text,
                auto_trigger,
            )

    snap = st.session_state.last_snapshot
    if snap:
        st.markdown(
            """
<div style="background:#FFFFFF;border:1px solid #DEDEDE;border-radius:8px;padding:12px 16px;margin:8px 0 12px 0;text-align:center;font-size:14px;font-weight:600;color:#0B0B0B;">
  ✅ 진단 완료!
</div>
""",
            unsafe_allow_html=True,
        )
        sync_prompt_from_widget = _render_results_panel(snap)

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

    st.markdown(
        '<p class="pc-footer">© 2026 Team 토큰부족. All rights reserved.</p>',
        unsafe_allow_html=True,
    )

    if sync_prompt_from_widget:
        st.session_state.rediagnose_prompt = user_prompt


if __name__ == "__main__":
    main()
