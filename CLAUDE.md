# Prompt Clinic — CLAUDE.md

## 프로젝트 한줄 설명
사용자의 LLM 프롬프트를 진단·개선하는 Streamlit + LangChain 웹앱.

---

## 파일 역할
- `main.py` : Streamlit UI, 세션 관리, 결과 렌더링
- `chains/context_chain.py` : Chain 1 — 맥락 프로필 생성 (ContextProfile Pydantic)
- `chains/diagnosis_chain.py` : Chain 2 — 4항목 진단 + CoT (DiagnosisResult Pydantic)
- `chains/rewrite_chain.py` : Chain 3 — 개선 프롬프트 생성 (RewriteResult Pydantic)
- `chains/pipeline.py` : 세 체인을 LCEL로 조립
- `utils/notion.py` : Notion API 연동 (현재 스텁, F-06 구현 대상)
- `.env` : API 키 (git 제외)
- `requirements.txt` : 의존성

---

## UI/UX 변경 이력 — 기술적 이유로 허용된 예외 목록
기능 구현 중 기술적 한계로 인해 UI/UX 동결 규칙의 예외가 발생한 경우 여기에 기록한다.
새 작업 시작 시 이 목록을 반드시 읽고 맥락을 파악할 것.

| 기능 ID | 변경 내용 | 이유 |
|---------|----------|------|
| F-04-2 | 복사 완료 피드백을 `st.toast` 대신 JS 내 버튼 텍스트 2초 변경으로 구현 | `st.toast`는 서버 사이드라 JS 클릭 이벤트에서 직접 호출 불가 |

## UI/UX 동결 규칙 — 절대 건드리지 말 것
- 레이아웃, 컬러, 폰트, 컴포넌트 배치, 여백 등 시각적 요소 일체 수정 금지
- 기존 `st.columns`, `st.expander`, `st.metric`, `st.progress` 구조 변경 금지
- 버튼 라벨, 플레이스홀더 텍스트, 캡션 문구 변경 금지
- 기능 구현 시 UI는 기존 패턴을 그대로 따른다 (새 컴포넌트 추가 시 기존 스타일 준수)
- UI/UX 변경이 필요하다고 판단되면 구현하지 말고 사용자에게 먼저 물어볼 것
- 기술적 한계로 UI 변경이 불가피한 경우: 구현 전 사용자에게 이유와 방법 보고 후 승인받을 것. 승인 후 위 "UI/UX 변경 이력" 섹션에 반드시 기록할 것

## Hard Rules — 절대 지킬 것
1. 모든 LLM 호출은 `main.py`의 `invoke_with_retry()`로 감싼다.
2. 모든 LLM 출력은 `JsonOutputParser` + Pydantic 모델로 파싱한다.
3. API 키는 `.env`에서만 읽는다. 절대 하드코딩 금지.
4. `session_state` 기존 키(`history`, `rediagnose_prompt`, `last_snapshot`, `auto_diagnose`)를 유지하고, 새 키는 snake_case로 추가.
5. 새 기능은 기존 chain/pipeline 구조를 건드리지 않고 확장한다.
6. 기존 함수명·시그니처를 변경하지 않는다.
7. Python 3.11+, type hint 필수, 함수 길이 50줄 이내 권장.
8. 문자열은 큰따옴표, import 순서: stdlib → third-party → local.
9. UI는 DB/세션에서만 읽는다. UI 레이어에서 외부 API(Notion 등)를 직접 호출하지 않는다. 외부 호출은 반드시 `utils/` 레이어에서만.
10. 히스토리는 append-only. `session_state.history` 리스트에 항목을 덮어쓰거나 삭제하지 않는다.
11. 미완성 기능은 기본 OFF 상태로 둔다. 환경변수 미설정 시 해당 기능 UI를 숨기거나 비활성화한다. (예: `NOTION_API_KEY` 없으면 Notion 저장 시도 안 함)

---

## 환경변수 (.env)
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o              # 선택, 기본값: gpt-4o
OPENAI_TEMPERATURE=0.2           # 선택, 기본값: 0.2
NOTION_API_KEY=secret_...        # F-06 구현 시 필요
NOTION_DB_ID=...                 # F-06 구현 시 필요
```

---

## ✅ 세션 시작 시 첫 번째로 할 일 (FIRST-RUN CHECKLIST)

**지금이 처음 세션이라면 반드시 아래 순서를 따를 것.**

### Step 1 — 전체 코드 정독
아래 파일들을 전부 읽고 실제 구현 상태를 파악한다.
```
main.py
chains/context_chain.py
chains/diagnosis_chain.py
chains/rewrite_chain.py
chains/pipeline.py
utils/notion.py
requirements.txt
```

### Step 2 — 기능 구현 상태 체크
각 파일을 읽은 뒤 아래 체크리스트와 대조해 실제 구현 여부를 직접 확인한다.
코드를 보지 않고 판단하지 말 것.

### Step 3 — 미구현 항목 보고
체크리스트 대조 결과를 사용자에게 보고한 뒤, 어떤 기능부터 구현할지 확인받는다.
임의로 순서를 정하지 말 것.

---

## 기능 구현 체크리스트

### F-01 맥락수집
- [x] F-01-1 목적 입력 — `st.text_area("목적")`
- [x] F-01-2 출력형식 선택 — `st.selectbox`
- [x] F-01-3 개선 목적 선택 — `st.multiselect`
- [x] F-01-4 맥락 프로필 생성 — `context_chain.py`

### F-02 품질분석
- [x] F-02-1 프롬프트 입력창 — 최소 10자 검증 포함
- [x] F-02-2 4개 기준 분석 — `diagnosis_chain.py`, 가중치 반영
- [x] F-02-3 항목별 점수 산출 — `apply_goal_weights()`, 등급 배지
- [x] F-02-4 문제 원인 설명 — 14점 이하 항목만 expander 표시. 전체 통과 시 "개선 필요 항목 없음" 안내.

### F-03 재작성
- [x] F-03-1 개선 프롬프트 생성 — `rewrite_chain.py`
- [x] F-03-2 변경 이유 제시 — `changes` 항목별 before/after/reason

### F-04 결과시각화
- [x] F-04-1 Before/After 나란히 표시 — `st.columns(2)`
- [x] F-04-2 복사 버튼 — JS `.then()` 콜백으로 버튼 텍스트 "✅ 복사 완료!" 2초 표시 (components.html 구조 유지).
- [x] F-04-3 재진단 버튼 — `auto_diagnose` 플래그로 재실행

### F-05 파일다운로드
- [x] F-05-1 md 파일 변환 — `build_markdown_report()`
- [ ] F-05-2 — ⏸ 보류: F-06-2 완료 후 진행

### F-06 자동아카이빙
- [ ] F-06-1 — ⏸ 보류: NOTION_API_KEY, NOTION_DB_ID 팀장 세팅 후 진행
- [ ] F-06-2 — ⏸ 보류: F-06-1 완료 후 진행

### F-07 히스토리
- [x] F-07-1 세션 히스토리 — `InMemoryChatMessageHistory`로 LangChain Memory 적용, 개선 프롬프트 요약 표시 추가.

### F-08 안정성
- [x] F-08-1 입력값 검증 — 프롬프트 10자 + 목적 입력란 10자 미만 제출 차단 추가.
- [x] F-08-2 API 재시도 카운터 UI — `invoke_with_retry`에 `on_retry` 콜백 추가, `status.update`로 `(n/2)` 표시.
- [x] F-08-3 Fallback 메시지 — `st.error`로 출력
- [x] F-08-4 로딩 3단계 메시지 — `🔍 프롬프트 분석 중 → 📊 진단 중 → ✍️ 개선안 생성 중 → ✅ 완료!` 정상 작동.

---

## 미구현/보류 항목 요약

| 기능 ID | 파일 | 내용 | 상태 |
|---------|------|------|------|
| F-06-1 | utils/notion.py | Notion API 실제 구현 | ⏸ 보류 |
| F-06-2 | main.py | Notion 저장 토스트 + 1회 재시도 + fallback 연결 | ⏸ 보류 |
| F-05-2 | main.py | F-06-2 실패 시 md 다운로드 버튼 자동 활성화 연동 | ⏸ 보류 |

---

## 작업 규칙
- 한 번에 한 기능 ID만 구현한다.
- 구현 전 반드시 해당 파일을 Read로 열어 현재 상태를 확인한다.
- 구현 후 변경된 부분을 사용자에게 요약 보고한다.
- 다음 기능으로 넘어가기 전 사용자 확인을 받는다.
- 기능 완료 시 위 체크리스트의 해당 항목을 `[ ]` → `[x]`로 업데이트한다.
- 새 라이브러리가 필요한 경우 `requirements.txt`도 반드시 함께 수정한다.
  - F-06-1 구현 시: `requests>=2.31.0` 추가 필요

## Verification Loop — 기능 구현 완료 후 필수 4단계
기능 하나 완료할 때마다 아래 4단계를 순서대로 실행한다. 하나라도 실패하면 수정 후 재실행.

```bash
# 1. Build — 임포트 오류 없는지 확인
python -c "import main; print('Build OK')"

# 2. Lint — 스타일/문법 오류 확인
ruff check . --select E,W,F

# 3. Type — 타입 힌트 오류 확인
mypy main.py chains/ utils/ --ignore-missing-imports

# 4. Diff — 의도치 않은 변경 없는지 확인
git diff --stat HEAD
```

## De-Sloppify — 기능 구현 완료 후 별도 컨텍스트에서 정리
Verification Loop PASS 후, **새 대화 세션**에서 아래 항목을 점검한다.
(같은 세션에서 하지 않는 이유: 구현 컨텍스트가 오염되면 놓치기 쉬움)

- `print()` 디버그 출력 제거 (Streamlit 앱은 `st.write` / `st.error` 사용)
- 죽은 코드(호출되지 않는 함수/변수) 제거
- 50줄 초과 함수 분리
- 변수명 불명확한 것 명확화
- `TODO / FIXME` 주석 처리 또는 이슈로 등록

## 코드 리뷰 기준 — 80% 확신 미만은 리포트하지 않는다
코드를 수정하거나 피드백할 때 확신도 기준을 적용한다.

| 확신도 | 처리 |
|--------|------|
| 80%+ | 반드시 수정 (MUST FIX) |
| 60~79% | 수정 권장, 사용자에게 판단 위임 |
| 60% 미만 | 리포트하지 않음 |

확신 없는 피드백을 남발하면 노이즈가 되어 중요한 이슈를 묻는다.
