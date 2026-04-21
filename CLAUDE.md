# Prompt Clinic — CLAUDE.md

## 프로젝트 한줄 설명
사용자의 LLM 프롬프트를 진단·개선하고, 학습 데이터로 누적하는 Streamlit + LangChain 웹앱.

---

## 파일 역할 (최신)
- `main.py` : Streamlit UI, 세션 관리, 진단 실행 오케스트레이션
- `chains/context_chain.py` : Chain 1 — 맥락 프로필 생성 + F-25-1 도메인 2축 분류
- `chains/diagnosis_chain.py` : Chain 2 — 4항목 진단 + Few-shot 로드 + F-25-2 행위축 가중치 + F-16-3 RAG 주입
- `chains/rewrite_chain.py` : Chain 3 — 개선 프롬프트 생성 (persona_instruction 선택 주입) + F-16-3 RAG 주입
- `chains/pipeline.py` : 기본 3체인 조립 (LCEL)
- `chains/model_router.py` : Phase 기반 모델 라우팅 설정 (Phase 1=OpenAI, Phase 2=Opus)
- `chains/self_improve_chain.py` : F-09/F-13 자가개선 루프 (Phase 1=OpenAI 최대 3회, Phase 2=Opus 최대 3회) + Best-of-N + 전략/페르소나 전환(F-22-1/2/3 통합)
- `chains/gate_chain.py` : F-20-1/2/3 모호성 게이트 체인 + 소크라테스 질문 체인
- `chains/drift_chain.py` : F-23-2 의도 드리프트 점수 산출 (3축 보존도 평가)
- `utils/notion.py` : Notion 저장 + Notion few-shot 조회 + F-15-2 fewshot DB write-back
- `utils/data_pipeline.py` : 원천 실행로그(`prompt_runs.jsonl`) 적재 + few-shot 자동 갱신 + F-15-1 quality_tag 태깅
- `utils/vector_store.py` : F-16-1 Chroma Vector DB 구축 + F-16-3 search_diagnosis/search_rewrite
- `data/fewshot_examples.json` : 진단 체인 few-shot 예시(JSON, fallback). `scripts/merge_collected.py`로 외부 소스 병합 가능
- `data/prompt_runs.jsonl` : 진단/개선 실행 원천 로그(JSONL, append-only)
- `data/guides/` : 외부 가이드북 PDF/MD/TXT (수동 관리, 재작성 인덱스 대상, domain_knowledge: "일반")
- `data/wiki/{domain}/` : 도메인별 LLM Wiki (medical/law/design/coding/science/marketing/general). `scripts/collect_wiki.py`로 한국어 Wikipedia에서 도메인당 15건 자동 수집 (YAML frontmatter + 본문 Markdown)
- `scripts/_classify.py` : 행위축/학문축 키워드 기반 분류 공용 모듈 (F-25 2축과 동일 라벨)
- `scripts/collect_huggingface.py` : HuggingFace `fka/awesome-chatgpt-prompts` → `data/collected_huggingface.jsonl`
- `scripts/collect_github.py` : GitHub `f/awesome-chatgpt-prompts/prompts.csv` → `data/collected_github.jsonl`
- `scripts/collect_wiki.py` : 도메인 시드 → 한국어 Wikipedia API → `data/wiki/{domain}/*.md` (`--domain` 다중 지정, `--min-chars` 스텁 필터)
- `scripts/merge_collected.py` : 수집물 + 기존 `fewshot_examples.json` 병합/중복제거 후 재저장 + 도메인 통계 출력
- `prompt_clinic_logo.png` : 브랜드 로고 PNG (없으면 인라인 SVG 폴백)
- `.env` : API 키 및 라우팅 옵션 (git 제외)
- `requirements.txt` : 의존성 (tiktoken, chromadb, langchain-community, markitdown, datasets 포함)

---

## UI/UX 변경 이력 — 기술적 이유로 허용된 예외 목록
기능 구현 중 기술적 한계로 인해 UI/UX 동결 규칙의 예외가 발생한 경우 여기에 기록한다.

| 기능 ID | 변경 내용 | 이유 |
|---------|----------|------|
| F-04-2 | 복사 완료 피드백을 `st.toast` 대신 JS 내 버튼 텍스트 2초 변경으로 구현 | `st.toast`는 서버 사이드라 JS 클릭 이벤트에서 직접 호출 불가 |
| F-02-4 | 진단 원인 설명을 14점 이하 항목만 표시 → 전체 항목 표시로 변경 | 팀장님 피드백 반영, 회의 결정사항 (2026-04-09) |
| F-20-2 | v0.2 설계(버튼 비활성화) → 안내 배너만 표시로 변경 | 진단 버튼 강제 차단 없음 원칙 적용. 사용자 흐름 보장 |
| F-20-4 | 독립 UI 컴포넌트 폐기 → F-13-3 대기 화면 내 통합 | 별도 UI 불필요. 루프 진행 상태 인디케이터로 통합 |
| F-16-5 | RAG 토큰 절감 시각화 UI 미구현 → jsonl 데이터 저장으로 대체 완료 처리 | before/after_token_count 이미 축적 중. 데이터 충분 시 통계 활용 예정 |

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

# F-09/F-13 자가개선 루프
SELF_IMPROVE_ENABLED=false
# SELF_IMPROVE_MAX_ITERS는 현재 미사용 — Phase 1은 3회, Phase 2는 3회로 하드코딩
ANTHROPIC_API_KEY=               # Opus(Phase 2) 활성화 시 필수
ANTHROPIC_MODEL_OPUS=claude-3-opus-20240229

# Notion 저장 (진단 결과 수동 아카이빙)
NOTION_API_KEY=secret_...
NOTION_DB_ID=...

# Notion few-shot source (기본 OFF)
FEWSHOT_SOURCE_NOTION=false
NOTION_FEWSHOT_ENABLED=false
NOTION_FEWSHOT_DB_ID=            # F-15-2 good/bad write-back 대상 DB (미설정 시 skip)
NOTION_FEWSHOT_PER_LEVEL=2

# F-16 RAG (기본 OFF — RAG_ENABLED=true로 변경 시 build_index() 먼저 실행 필요)
RAG_ENABLED=false
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
chains/gate_chain.py
chains/drift_chain.py
utils/notion.py
utils/data_pipeline.py
utils/vector_store.py
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
- [x] F-09-2 Phase 기반 모델 라우팅 (Phase 1=OpenAI 전담, Phase 2=Opus 전담 — 점수 기반 라우팅 제거)
- [x] F-09-3 feature flag (`SELF_IMPROVE_ENABLED`) 기본 OFF
- [x] F-09-4 루프 전략 다변화 → **F-13-2로 구현 완료** (정체 패턴 감지 + 페르소나 전환)
- [x] F-09-5 루프 상세 결과 시각화 → **F-13-3으로 구현 완료** (반복 이력 카드 UI)

### F-10 학습 데이터 파이프라인 / few-shot 자동화
- [x] F-10-1 원천 실행로그 `data/prompt_runs.jsonl` 적재 (append-only)
- [x] F-10-2 프롬프트 4단계 레벨 판정 로직 저장
- [x] F-10-3 run 로그 기반 `fewshot_examples.json` 자동 갱신
- [x] F-10-4 Notion few-shot 동적 로드(환경변수 ON 시)
- [x] F-10-5 레벨 균형 + 고득점 우선 few-shot 샘플링
- [ ] F-10-6 Notion에서 좋은/나쁜 사례 자동 write-back 루프 — **다음 작업 예정**

### F-12 레벨 뱃지 UI — ⛔ 폐기
> v0.2 범위에서 제외. 레벨 판정 로직(data_pipeline.py)은 내부 지표로만 활용.

### F-13 자가개선 루프 고도화 — ✅ Phase 3 완료
- [x] F-13-1 Best-of-N 선택 로직 (`_select_best_iteration()` — total_score 기준, 동점 시 later iter 우선)
- [x] F-13-2 전략 전환 트리거 + 페르소나 회전 (F-22-1/2/3 통합 — 4가지 정체 패턴 감지 → 5가지 페르소나 매핑)
- [x] F-13-3 반복 이력 카드 UI (`SELF_IMPROVE_ENABLED=true` 시 활성, F-20-4 통합)
- [ ] F-13-4 점수 추이 미니 차트 (`st.line_chart` 인라인 시각화) — 보류
- [x] F-13-5 tiktoken 토큰 카운팅 (UI 없음. `before_token_count`/`after_token_count` jsonl 저장)

### F-15 학습 데이터 자동화 — ✅ Phase 5 완료
- [x] F-15-1 우수/불량 사례 자동 태깅 (`_score_to_quality_tag()` — 80점↑ good / 40점↓ bad / 나머지 neutral)
- [x] F-15-2 Notion few-shot DB write-back (`push_fewshot_record()` — good/bad만, NOTION_FEWSHOT_DB_ID 필요)
- [x] F-15-3 학습 데이터 수집 동의 (진단 실행 자체를 동의로 간주, 정책 URL 링크 상시 노출)

### F-20 품질게이트 (모호성 게이팅)
> `chains/gate_chain.py` 신규. 진단 버튼 강제 차단 없음 — 안내 배너 + 소크라테스 질문만.
- [x] F-20-1 맥락 정보 모호성 사전 분석 (3축: 목표 40% / 제약 30% / 성공기준 30%, 임계값 0.5)
- [x] F-20-2 맥락 보완 권장 배너 (버튼 비활성화 제거 → 안내 배너로 변경)
- [x] F-20-3 소크라테스식 보완 질문 생성 (`st.expander` + "진단 계속하기" 버튼)
- [x] F-20-4 프로세스 상태 인디케이터 → **F-13-3에 통합** (독립 UI 폐기. 맥락 분석 중 › 목표 확인 완료 › 제약사항 검토 중)

### F-21 PAL 라우터 3단계 고도화 — ⛔ 폐기
> v0.2 범위 외. Phase 1(OpenAI) → Phase 2(Opus) 2단계 루프로 대체 구현.

### F-22 페르소나 회전 전략 — ✅ F-13-2에 통합 완료
> F-22-1(정체 패턴 감지) / F-22-2(페르소나 자동 선택) / F-22-3(페르소나 재작성 지시문) 모두
> `chains/self_improve_chain.py` 내 F-13-2로 통합 구현. 별도 파일 없음.

### F-23 의도 드리프트 측정 & 원본 잠금
> `chains/drift_chain.py` 신규. UI 노출 없음 — 내부 성능 지표.
- [x] F-23-1 원본 프롬프트 불변 잠금 (`session_state.original_prompt` frozen)
- [x] F-23-2 의도 드리프트 점수 산출 (3축 보존도, `drift_score` jsonl 적재)
- ⛔ F-23-3 드리프트 경고 배너 — 제외
- ⛔ F-23-4 원본 vs 개선안 의도 비교 뷰 — 제외

### F-24 멀티 모델 컨센서스 평가 — ⛔ 전체 폐기
> 현재 모델 2개 체계(OpenAI ↔ Opus) 유지. 추가 API 키 연동 없음.

### F-25 도메인 2축 분류 (진단 파이프라인 최전방)
> **2축 분류**: 행위축(코드/요약/글쓰기/분석/QA) + 학문축(의학/법률/코딩/디자인/마케팅/과학/일반).
> 행위축 → few-shot 필터 + 채점 가중치. 학문축 → RAG 필터 (F-16-3 연계).
- [x] F-25-1 도메인 2축 자동 분류 (`context_chain.py` 확장, LLM 단일 호출)
- [x] F-25-2 행위축 기반 채점 가중치 테이블 (`diagnosis_chain.py`)
- [x] F-25-3 행위축 기반 few-shot 필터링 (`data_pipeline.py`, fallback 포함)
- ⛔ F-25-4 도메인 표시 UI — 내부 처리 전용, UI 제외
- [ ] F-25-5 소크라테스 질문 기반 맥락 보완 재진단 (F-20-3 연동, 보류)

### F-16 RAG 인프라 — ✅ Phase 4 완료
> `utils/vector_store.py` 신규. `RAG_ENABLED=false` 기본값 — 활성화 전 `build_index()` 먼저 실행 필요.
> wiki 로더는 `.md` 상단의 YAML frontmatter(`---`…`---`)를 제거한 뒤 청킹한다.
- [x] F-16-1 통합 Vector DB 인덱스 구축 (`utils/vector_store.py`, Chroma)
  - 진단 컬렉션: `prompt_runs.jsonl` + `fewshot_examples.json` + `data/guides/` + `data/wiki/{domain}/` (행위축 필터)
  - 재작성 컬렉션: `data/guides/` PDF/MD/TXT + `data/wiki/{domain}/` + `fewshot_examples.json` (학문축 필터)
- [x] F-16-3 체인별 분리 인덱스 운영 (Chain 2: `search_diagnosis()` / Chain 3: `search_rewrite()`, 둘 다 k=3)
- [ ] F-16-4 RAG 정량적 성능 지표 — **0.2 개발 완료 후 착수**
- [x] F-16-5 RAG 토큰 절감 — UI 없음. `before/after_token_count` jsonl 축적으로 완료 처리. 데이터 충분 시 통계 활용 예정

### F-27 학습 데이터 수집 파이프라인 (scripts/) — ✅ 완료
> 외부 공개 소스에서 프롬프트/도메인 지식을 수집해 few-shot과 RAG 인덱스를 증강한다.
- [x] F-27-1 프롬프트 수집: HuggingFace(`fka/awesome-chatgpt-prompts`) + GitHub(`f/awesome-chatgpt-prompts`) → `data/collected_*.jsonl`
- [x] F-27-2 키워드 기반 2축 자동 분류(`scripts/_classify.py`) + `fewshot_examples.json` 병합/중복제거(`scripts/merge_collected.py`)
- [x] F-27-3 도메인 지식 수집: 한국어 Wikipedia API → `data/wiki/{domain}/*.md` (도메인당 15건, frontmatter + 본문 Markdown, 500자 미만 스텁 자동 스킵)
- [x] F-27-4 Chroma 재빌드 후 `search_rewrite`/`search_diagnosis`에서 도메인/행위축 필터로 활용

---

## 미구현/보류 항목 요약
| 기능 ID | 파일 | 내용 | 상태 |
|---------|------|------|------|
| F-10-6 | utils/notion.py, utils/data_pipeline.py | Notion good/bad 사례 자동 write-back 루프 | **다음 작업 예정** |
| F-13-4 | main.py | 점수 추이 미니 차트 (`st.line_chart`) | 보류 |
| F-16-4 | prompt_runs.jsonl 필드 확장 | RAG 정량 성능 지표 (latency, 적중률) | 0.2 완료 후 착수 |
| F-25-5 | main.py, chains/ | 소크라테스 질문 기반 맥락 보완 재진단 | 보류 |

---

## 폐기 확정 항목
| 기능 ID | 이유 |
|---------|------|
| F-12 레벨 뱃지 UI | v0.2 범위 외. 레벨 판정 로직은 내부 지표로만 유지 |
| F-20-2 버튼 비활성화 | 진단 버튼 강제 차단 없음 원칙. 안내 배너로 대체 구현 |
| F-20-4 독립 UI | F-13-3 대기 화면 내 상태 인디케이터로 통합 |
| F-21 PAL 라우터 전체 | v0.2 범위 외. 현행 2-tier 라우팅 유지 |
| F-22 (독립 파일) | F-13-2 내 통합 구현. 별도 파일 없음 |
| F-23-3 드리프트 배너 | UI 노출 없음 원칙 |
| F-23-4 의도 비교 뷰 | UI 노출 없음 원칙 |
| F-24 멀티 컨센서스 | 추가 API 키 연동 불필요. 모델 2개 체계 유지 |
| F-25-4 도메인 표시 UI | 내부 처리 전용 |
| F-13-6 iteration 히스토리 | F-13-3에 통합 완료 (F-26 통합 결정) |

---

## prompt_runs.jsonl 필드 목록 (최신)
| 필드명 | 타입 | 설명 |
|--------|------|------|
| `ts` | str | ISO 8601 타임스탬프 |
| `prompt_name` | str | 프롬프트 명 |
| `purpose` | str | 사용목적 |
| `output_format` | str | 출력 형식 |
| `improvement_goals` | list | 개선 목적 목록 |
| `user_prompt` | str | 원본 프롬프트 |
| `improved_prompt` | str | 최종 개선 프롬프트 |
| `total_score` | int | 종합 점수 (0~100) |
| `grade` | str | 등급 (우수/보통/개선필요) |
| `scores` | dict | 4항목 점수 |
| `prompt_level` | dict | 레벨 정보 (level, label) |
| `analysis_summary` | str | 진단 요약 |
| `domain_action` | str | 행위축 분류 결과 |
| `domain_knowledge` | str | 학문축 분류 결과 |
| `drift_score` | float | 의도 드리프트 점수 (0~1) |
| `before_token_count` | int | 원본 프롬프트 토큰 수 (tiktoken) |
| `after_token_count` | int | 개선 프롬프트 토큰 수 (tiktoken) |
| `loop_history` | list | 자가개선 루프 이력 (strategy/persona 포함) |
| `best_iteration_no` | int\|null | 최고점 이터레이션 번호 |
| `quality_tag` | str | F-15-1 자동 태깅 결과 (good/bad/neutral) |

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

### 다음 작업 — F-10-6 Notion write-back 루프
- F-15-2(push_fewshot_record) 완료로 선행 조건 충족
- `sync_learning_data()` 호출 흐름 내에서 good/bad 사례 Notion 자동 누적

### A. 보류 중인 기능
- F-13-4 점수 추이 미니 차트 (`st.line_chart` 인라인)
- F-25-5 소크라테스 질문 기반 맥락 보완 재진단 (F-20-3 연동)
- F-16-4 RAG 정량적 성능 지표 (0.2 개발 완료 후)

### B. Notion OAuth (사용자별 개인 DB 저장)
- 현재는 팀 공용 Notion DB에 저장 (`.env`의 `NOTION_DB_ID` 고정)
- 사용자 각자의 Notion 계정으로 OAuth 인증 후 개인 DB에 저장
- 다중 사용자 환경 대응

### C. 프롬프트 품질 트래킹 대시보드
- `data/prompt_runs.jsonl` 데이터 기반 시각화
- 사용자별/날짜별 점수 추이, 레벨 변화 그래프
- 행위축/학문축별 취약점 패턴 분석
