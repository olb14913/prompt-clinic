"""Streamlit: 프롬프트 진단 클리닉 메인 앱."""

from __future__ import annotations

import base64
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
    lines.extend(["", "## Before / After", "", "### Before", "```", user_prompt, "```", "", "### After", "```", improved, "```", "", "## 변경 이유"])
    for ch in changes:
        lines.append(
            f"- **{ch.get('criterion', '')}**: `{ch.get('before', '')}` → `{ch.get('after', '')}`  \n  {ch.get('reason', '')}"
        )
    return "\n".join(lines)


def dynamic_text_area_height(text: str, min_px: int = 150, max_px: int = 400) -> int:
    t = text or ""
    line_count = max(1, t.count("\n") + 1)
    px = 52 + line_count * 22
    return max(min_px, min(max_px, px))


def criterion_expander_title(label: str, final: int, bonus_pts: int) -> str:
    return f"{label} — {final}/25"


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


def render_clipboard_copy_button(text: str, label: str, dom_id: str) -> None:
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    safe_id = "".join(c if c.isalnum() else "_" for c in dom_id)[:64]
    components.html(
        f"""
        <button id="{safe_id}" type="button" style="
          background-color:#f0f2f6;color:#31333F;border:1px solid #d1d5db;
          padding:0.375rem 0.75rem;border-radius:0.25rem;cursor:pointer;
          font-size:0.875rem;font-family:sans-serif;width:100%;">
          {label}
        </button>
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
        with st.status("처리 중...", expanded=True) as status:
            status.update(label="🔍 프롬프트 분석 중...", state="running")
            context_profile = invoke_with_retry(
                context_r.invoke,
                base_input,
                on_retry=_make_retry_cb(status, "맥락 분석"),
            )
            merged = {**base_input, "context_profile": context_profile}
            status.update(label="📊 진단 중...", state="running")
            diagnosis = invoke_with_retry(
                diagnosis_r.invoke,
                merged,
                on_retry=_make_retry_cb(status, "품질 진단"),
            )
            weighted = apply_goal_weights(diagnosis, improvement_goals)
            status.update(label="✍️ 개선안 생성 중...", state="running")
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
                    on_iteration=lambda n, total, stage: status.update(
                        label=f"🔁 자가개선 {n}/{total} {stage}",
                        state="running",
                    ),
                    on_retry=_make_retry_cb(status, "자가개선 루프"),
                )
                best = loop_result.get("best") or {}
                rewrite = best.get("rewrite") or {}
                weighted = best.get("weighted") or weighted
                diagnosis = best.get("diagnosis_raw") or diagnosis
            else:
                merged = {**merged, "diagnosis": diagnosis}
                rewrite = invoke_with_retry(
                    rewrite_r.invoke,
                    merged,
                    on_retry=_make_retry_cb(status, "개선안 생성"),
                )
            status.update(label="✅ 완료!", state="complete")
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

    st.divider()
    st.subheader("진단 결과")
    cols_score = st.columns(4)
    for i, key in enumerate(crit_keys):
        sc = weighted["weighted_scores"][key]
        with cols_score[i]:
            st.metric(CRITERION_LABELS[key], f"{sc} / 25")
            score_progress_bar(sc)
    st.caption("등급 기준: 🟢 20점 이상 · 🟡 10~19점 · 🔴 9점 이하 (항목별 최종 점수 기준)")
    st.markdown(
        f'<p style="margin:0.5rem 0 0 0;font-size:1.15rem;line-height:1.35;">'
        f"<strong>종합 {weighted['total_score']} / 100</strong>&nbsp;"
        f'{weighted["grade_badge"]}&nbsp;<strong>{weighted["grade"]}</strong></p>',
        unsafe_allow_html=True,
    )
    st.markdown("#### 항목별 원인 (CoT)")
    for key in crit_keys:
        b_pts = int(bonus_map.get(key, 0))
        final_sc = weighted["weighted_scores"][key]
        exp_title = criterion_expander_title(CRITERION_LABELS[key], final_sc, b_pts)
        with st.expander(exp_title):
            st.write(weighted["reasons"].get(key, ""))

    st.divider()
    st.subheader("개선 결과")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before**")
        st.text_area(
            "before",
            value=original,
            height=dynamic_text_area_height(original),
            disabled=True,
            label_visibility="collapsed",
        )
    with c2:
        st.markdown("**After**")
        st.text_area(
            "after",
            value=improved,
            height=dynamic_text_area_height(improved),
            disabled=True,
            label_visibility="collapsed",
        )

    st.markdown("#### 항목별 변경 이유")
    for ch in changes:
        st.markdown(
            f"- **{ch.get('criterion', '')}**  \n"
            f"  - 변경 전: `{ch.get('before', '')}`  \n"
            f"  - 변경 후: `{ch.get('after', '')}`  \n"
            f"  - 이유: {ch.get('reason', '')}"
        )

    st.markdown("**개선 프롬프트**")
    copy_id = f"pc_copy_{abs(hash(improved)) % 10_000_000}"
    col_copy, col_gap, col_redo, _ = st.columns([1.15, 0.35, 1.15, 3.0])
    with col_copy:
        render_clipboard_copy_button(improved, "프롬프트 복사", copy_id)
    with col_gap:
        st.write("")
    with col_redo:
        if st.button(
            "개선된 프롬프트로 재진단",
            type="primary",
            key="redo_diagnose",
            use_container_width=True,
        ):
            st.session_state.rediagnose_prompt = improved
            sync = False
            st.session_state.auto_diagnose = True
            st.markdown(
                "<script>window.scrollTo(0, 0);</script>",
                unsafe_allow_html=True,
            )
            st.rerun()

    if os.environ.get("NOTION_API_KEY") and os.environ.get("NOTION_DB_ID"):
        col_notion, _ = st.columns([2, 4])
        with col_notion:
            if st.button("Notion에 저장", key="notion_save", use_container_width=True):
                try:
                    _save_to_notion_with_retry(snap)
                    st.session_state.notion_save_status = "success"
                    st.toast("Notion에 저장되었습니다!")
                except Exception as e:
                    st.session_state.notion_save_status = "error"
                    err_detail = f"{type(e).__name__}: {e}"
                    st.error(
                        "Notion 저장에 실패했습니다. 잠시 후 다시 시도하거나 "
                        f"아래 리포트 다운로드를 이용하세요.\n\n상세: `{err_detail}`"
                    )

    prompt_name = str(snap.get("prompt_name") or "prompt_clinic")
    safe_prompt_name = sanitize_prompt_name_for_filename(prompt_name)
    fn = f"{safe_prompt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
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
    if st.session_state.get("notion_save_status") == "error":
        st.info("Notion 저장에 실패했습니다. 아래 버튼으로 리포트를 저장하세요.")
    st.download_button(
        "리포트 다운로드 (.md)",
        data=md_body,
        file_name=fn,
        mime="text/markdown",
        type="secondary",
    )
    return sync


def _render_history_panel() -> None:
    """세션 히스토리 패널 렌더링."""
    st.divider()
    st.subheader("세션 히스토리")
    if not st.session_state.history:
        st.info("아직 진단 이력이 없습니다.")
        return
    for h in st.session_state.history:
        ts: datetime = h["ts"]
        preview = (h["user_prompt"] or "")[:80].replace("\n", " ")
        improved_preview = (
            str(h.get("rewrite", {}).get("improved_prompt", ""))[:80]
            .replace("\n", " ")
        )
        hist_weighted = h.get("weighted")
        if isinstance(hist_weighted, dict) and "total_score" in hist_weighted:
            grade_part = (
                f" | {hist_weighted['total_score']}/100"
                f" {hist_weighted.get('grade_badge', '')}"
                f" {hist_weighted.get('grade', '')}"
            )
        else:
            grade_part = ""
        st.markdown(
            f"**{ts.strftime('%Y-%m-%d %H:%M:%S')}**{grade_part}  \n"
            f"원본: {preview}…  \n"
            f"개선: {improved_preview}…"
        )


def main() -> None:
    init_session()
    st.set_page_config(page_title="Prompt Clinic", page_icon="🩺", layout="wide")
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
    if st.session_state.get("auto_diagnose"):
        st.markdown(
            "<script>window.scrollTo(0, 0);</script>",
            unsafe_allow_html=True,
        )
    st.title("🩺 Prompt Clinic")
    st.caption("프롬프트를 진단하고 개선안을 제안합니다.")

    with st.sidebar:
        st.header("맥락 수집")
        prompt_name = st.text_input("프롬프트 명", placeholder="예: prompt_v1")
        st.caption("추후 프롬프트 다운로드 및 저장 시, 파일명으로 사용됩니다.")
        purpose = st.text_area(
            "프롬프트 사용목적",
            placeholder="이 프롬프트를 어디에 사용하는지 적어주세요.",
            height=100,
        )
        output_format = st.selectbox("출력 형식", OUTPUT_FORMAT_OPTIONS, index=0)
        improvement_goals = st.multiselect("개선 목적", IMPROVEMENT_OPTIONS)

    st.subheader("프롬프트 입력")
    user_prompt = st.text_area(
        "진단할 프롬프트",
        value=st.session_state.rediagnose_prompt,
        height=220,
        placeholder="프롬프트를 입력하세요...",
    )

    auto_trigger = st.session_state.get("auto_diagnose", False)
    if auto_trigger:
        st.info("💡 개선된 프롬프트로 재진단을 시작합니다...")

    run = st.button("진단 시작", type="primary")
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
            st.error("프롬프트 명은 영문/숫자/한글과 -, _, ., 공백 만 입력할 수 있습니다.")
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
        sync_prompt_from_widget = _render_results_panel(snap)

    _render_history_panel()

    if sync_prompt_from_widget:
        st.session_state.rediagnose_prompt = user_prompt


if __name__ == "__main__":
    main()
