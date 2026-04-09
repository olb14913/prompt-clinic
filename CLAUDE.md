# Prompt Clinic — CLAUDE.md

## 프로젝트 한줄 설명
사용자의 LLM 프롬프트를 진단·개선하고, 학습 데이터로 누적하는 Streamlit + LangChain 웹앱.

---

## 파일 역할 (최신)
- `main.py` : Streamlit UI, 세션 관리, 진단 실행 오케스트레이션
- `chains/context_chain.py` : Chain 1 — 맥락 프로필 생성
- `chains/diagnosis_chain.py` : Chain 2 — 4항목 진단 + Few-shot 로드
- `chains/rewrite_chain.py` : Chain 3 — 개선 프롬프트 생성
- `chains/pipeline.py` : 기본 3체인 조립 (LCEL)
- `chains/model_router.py` : 점수 기반 모델 라우팅(OpenAI ↔ Opus) 설정
- `chains/self_improve_chain.py` : F-09 자가개선 루프(개선→재진단 반복)
- `utils/notion.py` : Notion 저장 + Notion few-shot 조회 유틸
- `utils/data_pipeline.py` : 원천 실행로그(`prompt_runs.jsonl`) 적재 + few-shot 자동 갱신
- `data/fewshot_examples.json` : 진단 체인 few-shot 예시(JSON, fallback)
- `data/prompt_runs.jsonl` : 진단/개선 실행 원천 로그(JSONL, append-only)
- `.env` : API 키 및 라우팅 옵션 (git 제외)
- `requirements.txt` : 의존성

---

## UI/UX 변경 이력 — 기술적 이유로 허용된 예외 목록
기능 구현 중 기술적 한계로 인해 UI/UX 동결 규칙의 예외가 발생한 경우 여기에 기록한다.

| 기능 ID | 변경 내용 | 이유 |
|---------|----------|------|
| F-04-2 | 복사 완료 피드백을 `st.toast` 대신 JS 내 버튼 텍스트 2초 변경으로 구현 | `st.toast`는 서버 사이드라 JS 클릭 이벤트에서 직접 호출 불가 |
| F-02-4 | 진단 원인 설명을 14점 이하 항목만 표시 → 전체 항목 표시로 변경 | 팀장님 피드백 반영, 회의 결정사항 (2026-04-09) |

## UI/UX 동결 규칙 — 절대 건드리지 말 것
- 레이아웃, 컬러, 폰트, 컴포넌트 배치, 여백 등 시각적 요소 수정 금지
- 기존 `st.columns`, `st.expander`, `st.metric`, `st.progress` 구조 변경 금지
- 버튼 라벨/플레이스홀더/캡션 임의 변경 금지
- UI 변경 필요 시 구현 전에 사용자 확인 필수

## Hard Rules — 절대 지킬 것
1. 모든 LLM 호출은 `main.py`의 `invoke_with_retry()`로 감싼다.
2. 모든 LLM 출력은 `JsonOutputParser` + Pydantic 모델로 파싱한다.
3. API 키는 `.env`에서만 읽는다. 하드코딩 금지.
4. `session_state` 기존 키(`history`, `rediagnose_prompt`, `last_snapshot`, `auto_diagnose`) 유지.
5. 히스토리는 append-only (`session_state.history` 덮어쓰기/삭제 금지).
6. UI에서 외부 API 직접 호출 금지. 외부 연동은 `utils/`에서만.
7. 미완성 기능은 기본 OFF (`feature flag` 또는 환경변수) 상태로 둔다.

---

## 환경변수 (.env) — 최신
```bash
# 필수
OPENAI_API_KEY=sk-...

# 선택 — OpenAI 모델 설정 (기본값으로 동작, gpt-4o 단일 모델 사용)
OPENAI_MODEL=gpt-4o
OPENAI_TEMPERATURE=0.2
OPENAI_DIAGNOSIS_MODEL=gpt-4o    # 진단 체인 모델 (기본: OPENAI_MODEL)
OPENAI_REWRITE_MODEL=gpt-4o      # 재작성 체인 모델 (기본: OPENAI_MODEL)

# F-09 / Model routing
SELF_IMPROVE_ENABLED=false
SELF_IMPROVE_MAX_ITERS=3
OPUS_SCORE_THRESHOLD=70
ANTHROPIC_API_KEY=               # Opus 활성화 시 필수
ANTHROPIC_MODEL_OPUS=claude-3-opus-20240229

# Notion 저장
NOTION_API_KEY=secret_...
NOTION_DB_ID=...

# Notion few-shot source (기본 OFF)
FEWSHOT_SOURCE_NOTION=false
NOTION_FEWSHOT_ENABLED=false
NOTION_FEWSHOT_DB_ID=
NOTION_FEWSHOT_PER_LEVEL=2
```

---

## ✅ 세션 시작 시 FIRST-RUN CHECKLIST
### Step 1 — 전체 코드 정독
```text
main.py
chains/context_chain.py
chains/diagnosis_chain.py
chains/rewrite_chain.py
chains/pipeline.py
chains/model_router.py
chains/self_improve_chain.py
utils/notion.py
utils/data_pipeline.py
requirements.txt
```

### Step 2 — 구현 상태 체크
아래 체크리스트와 실제 코드를 반드시 대조한다.

### Step 3 — 미구현 항목 보고
체크 결과를 사용자에게 보고하고 다음 구현 항목을 확인받는다.

---

## 기능 구현 체크리스트 (최신)

### F-01 맥락수집
- [x] F-01-1 `프롬프트 명` 입력 + 캡션 추가
- [x] F-01-2 `프롬프트 사용목적` 입력
- [x] F-01-3 출력형식 선택 (`st.selectbox`)
- [x] F-01-4 개선 목적 선택 (`st.multiselect`)
- [x] F-01-5 맥락 프로필 생성 (`context_chain.py`)

### F-02 품질분석
- [x] F-02-1 프롬프트 입력창
- [x] F-02-2 4개 기준 분석 (`diagnosis_chain.py`)
- [x] F-02-3 항목별 점수 산출 (`apply_goal_weights()`)
- [x] F-02-4 문제 원인 설명 전체 항목 표시

### F-03 재작성
- [x] F-03-1 개선 프롬프트 생성 (`rewrite_chain.py`)
- [x] F-03-2 변경 이유 제시 (`changes`)

### F-04 결과시각화
- [x] F-04-1 Before/After 나란히 표시
- [x] F-04-2 복사 버튼 완료 피드백(JS)
- [x] F-04-3 개선안 재진단 버튼

### F-05 파일다운로드
- [x] F-05-1 md 리포트 변환 (`build_markdown_report`)
- [x] F-05-2 Notion 실패 시 다운로드 fallback 안내
- [x] F-05-3 파일명에 프롬프트 명 반영 (`{prompt_name}_YYYYMMDD_HHMMSS.md`)

### F-06 자동아카이빙
- [x] F-06-1 Notion 저장 실구현 (`save_diagnosis_page`)
- [x] F-06-2 Notion 저장 버튼 + 1회 재시도 + fallback
- [x] F-06-3 DB 스키마 동적 매핑(title/select/multi_select 등)

### F-07 히스토리
- [x] F-07-1 세션 히스토리 표시
- [x] F-07-2 LangChain 메모리(`InMemoryChatMessageHistory`) 저장

### F-08 안정성/검증
- [x] F-08-1 입력 정책 검증
  - 프롬프트 명 필수, 20자 이하, 허용문자 제한
  - 프롬프트 사용목적 필수, 100자 이하
  - 진단할 프롬프트 필수, 500자 이하
  - 개선 목적 1개 이상 필수
- [x] F-08-2 API 재시도 카운터 UI
- [x] F-08-3 실패 fallback 메시지
- [x] F-08-4 로딩 단계 메시지

### F-09 자가개선 루프 / 모델 라우팅
- [x] F-09-1 `chains/self_improve_chain.py` 추가 (개선→재진단 반복)
- [x] F-09-2 점수 기반 rewrite 모델 라우팅(OpenAI ↔ Opus)
- [x] F-09-3 feature flag (`SELF_IMPROVE_ENABLED`) 기본 OFF
- [ ] F-09-4 루프 전략 다변화(점수 정체 시 대안 전략 탐색)
- [ ] F-09-5 루프 상세 결과 시각화(UI)

### F-10 학습 데이터 파이프라인 / few-shot 자동화
- [x] F-10-1 원천 실행로그 `data/prompt_runs.jsonl` 적재 (append-only)
- [x] F-10-2 프롬프트 4단계 레벨 판정 로직 저장
- [x] F-10-3 run 로그 기반 `fewshot_examples.json` 자동 갱신
- [x] F-10-4 Notion few-shot 동적 로드(환경변수 ON 시)
- [x] F-10-5 레벨 균형 + 고득점 우선 few-shot 샘플링
- [ ] F-10-6 Notion에서 좋은/나쁜 사례 자동 write-back 루프

---

## 미구현/보류 항목 요약
| 기능 ID | 파일 | 내용 | 상태 |
|---------|------|------|------|
| F-09-4 | chains/self_improve_chain.py, main.py | 정체 구간 대체 전략(다양한 rewrite 정책) | ⏳ |
| F-09-5 | main.py | 자가개선 반복 히스토리 UI 표시 | ⏳ |
| F-10-6 | utils/notion.py, utils/data_pipeline.py | Notion good/bad 사례 자동 write-back | ⏳ |
| RAG-01 | (신규 예정) | 벡터 인덱스 구축 및 retrieval 연동 | 보류 |

---

## 작업 규칙
- 한 번에 한 기능 ID 단위로 구현한다.
- 구현 전 관련 파일을 반드시 읽고 현재 상태를 확인한다.
- 구현 후 변경 요약 + 검증 결과를 사용자에게 보고한다.

## Verification Loop — 기능 구현 완료 후 필수 4단계
```bash
# 1. Build
python3 -c "import main; print('Build OK')"

# 2. Lint
ruff check . --select E,W,F

# 3. Type
mypy main.py chains/ utils/ --ignore-missing-imports

# 4. Diff
git diff --stat HEAD
```

## 코드 리뷰 기준 — 80% 확신 미만은 리포트하지 않는다
| 확신도 | 처리 |
|--------|------|
| 80%+ | MUST FIX |
| 60~79% | 수정 권장(사용자 판단 위임) |
| 60% 미만 | 리포트하지 않음 |

---

## 향후 업그레이드 계획 (최신)

### A. F-09 고도화
- 루프 전략 다변화(temperature/constraint 강화/역할 프롬프트 전환)
- 반복 이력 기반 best-of-n 선택 로직 고도화
- Opus 사용 정책(히스테리시스, 토큰 예산 기반 제한) 정교화

### B. Few-shot/Notion 고도화
- Notion DB에서 좋은/나쁜 사례 자동 선별 후 동기화
- 팀 아카이브 DB와 few-shot 소스 DB 분리 운영
- 사용자 동의 기반 데이터만 학습풀 반영

### C. RAG (보류)
- Notion/실행로그를 임베딩 인덱스로 변환
- 레벨/점수/도메인 메타 필터 기반 retrieval
- 진단/재작성 단계별 retrieval 전략 분리
