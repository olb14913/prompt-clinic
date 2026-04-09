# PR: Prompt Clinic 기능 구현 및 고도화 완료

---

## 1. 한줄 요약

초기 구현 완성(F-01~F-08) 이후 맥락수집 정책 개편, Notion 저장 안정화, 학습 데이터 파이프라인, F-09 자가개선 루프 1차, Notion few-shot 동적 로드까지 반영해 운영/확장 기반을 고도화했습니다.

---

## 2. 작업 배경 및 범위

이번 PR 계열에서 반영된 주요 범위:
- 맥락수집 입력 항목 To-be 반영(프롬프트 명 추가/입력 정책 강화)
- Notion 저장 스키마 동적 매핑 + fallback
- 학습 원천 데이터 적재(`prompt_runs.jsonl`) + few-shot 자동 갱신
- 점수 기반 모델 라우팅(OpenAI 저가 모델 ↔ Claude Opus)
- F-09 자가개선 루프 1차 구현(반복 개선 + 최고점 선택)
- Notion 기반 few-shot 동적 로드(기본 OFF + JSON fallback)
- 레벨(1~4) 균형 + 고득점 우선 few-shot 선별

---

## 3. 구현 완료 항목 (누적)

### F-02-4 — 진단 원인: 전체 항목 표시

**무엇을**: 진단 결과 화면에서 모든 항목의 원인 설명을 expander로 펼쳐 보여줍니다.
**왜**: 팀장님 피드백 및 회의 결정사항(2026-04-09)에 따라 변경. 초기 구현에서는 14점 이하 항목만 표시했으나, 전체 항목을 확인할 수 있어야 사용자가 진단 결과를 충분히 이해할 수 있다는 의견이 반영되었습니다.

---

### F-04-2 — 복사 버튼 완료 피드백

**무엇을**: "프롬프트 복사" 버튼을 누르면 버튼 텍스트가 "✅ 복사 완료!"로 2초간 바뀝니다.
**왜**: 복사가 됐는지 안 됐는지 알 수 없어 사용자가 반복 클릭하는 문제를 해소합니다.
**기술적 선택**: Streamlit의 `st.toast()`는 서버 측 코드라 JavaScript 클릭 이벤트에서 직접 호출할 수 없습니다. 기존 HTML 버튼 구조를 유지하면서 JS 내부에서 텍스트를 변경하는 방식으로 동일한 피드백을 구현했습니다.

---

### F-07-1 — 세션 히스토리 개선

**무엇을**: 히스토리 각 항목에 "원본 프롬프트 미리보기"와 "개선 프롬프트 요약"을 함께 표시합니다. LangChain의 `InMemoryChatMessageHistory`를 적용해 원본·개선 프롬프트 쌍을 메모리에 저장합니다.
**왜**: 이전에는 타임스탬프와 점수만 보여서 어떤 프롬프트를 진단했는지 히스토리만으로 확인할 수 없었습니다. 저장된 메모리는 추후 F-09(자가개선 루프) 기능에서 활용 예정입니다.

---

### F-08-1 — 목적 입력란 검증

**무엇을**: 목적 입력란에 1~9자를 입력한 채로 진단을 시작하면 오류 메시지를 표시하고 진단을 차단합니다. 비워두는 것은 허용됩니다.
**왜**: 너무 짧은 목적 설명은 맥락 프로필 품질을 낮춥니다.

---

### F-08-2 — API 재시도 카운터 UI

**무엇을**: API 호출 실패 시 자동으로 최대 2회 재시도하며, 로딩 화면에 "⚠️ 맥락 분석 재시도 중 (1/2)..." 메시지를 표시합니다.
**왜**: 기존에는 재시도 중에도 화면이 멈춘 것처럼 보여 사용자가 오동작으로 오해할 수 있었습니다.

---

### F-08-4 — 로딩 3단계 메시지

**무엇을**: 진단 처리 중 로딩 상태가 3단계로 바뀝니다.
`🔍 프롬프트 분석 중` → `📊 진단 중` → `✍️ 개선안 생성 중` → `✅ 완료!`
**왜**: 기존 코드는 2·3단계가 동일한 "분석 중" 메시지를 반복해 진행 상황을 알 수 없었습니다.

---

### 코드 품질 개선 (De-Sloppify)

기능 구현 외에 코드 건강을 위한 정리 작업을 병행했습니다.

| 변경 | 내용 |
|------|------|
| `main()` 함수 분리 | 270줄 → 55줄. `_run_diagnosis`, `_render_results_panel`, `_render_history_panel` 3개 함수로 분리 |
| `_make_retry_cb` 추출 | 5단계 중첩 내부 함수 → 모듈 레벨 함수로 이동 |
| 변수명 명확화 | 히스토리 루프의 `w` → `hist_weighted` |
| 불필요한 중간 변수 제거 | `api_key_ok` → `elif not os.environ.get(...)` 인라인 |

---

### Few-shot 예시 외부화 + 보통 등급 추가

**무엇을**: 진단 체인의 Few-shot 예시 2개를 `chains/diagnosis_chain.py` 소스 코드에서 꺼내 `data/fewshot_examples.json`으로 분리했습니다. 동시에 "보통(50~79점)" 등급 예시 1개를 추가해 총 3개(개선필요·보통·우수)로 구성했습니다.
**왜**: 소스 코드를 건드리지 않고 예시를 추가·수정할 수 있습니다. "보통" 예시가 없으면 AI가 중간 수준 프롬프트를 채점할 때 일관성이 낮아질 수 있습니다.

---

### 모델 설정 환경변수화

**무엇을**: 기존에 소스 코드에 박혀 있던 `gpt-4o`와 `0.2`(temperature)를 `.env`에서 읽도록 변경했습니다.
**왜**: 팀 운영 환경에서 모델을 바꾸거나 비용 절감을 위해 설정을 조정할 때 코드를 수정하지 않아도 됩니다.

### F-06-1 — Notion API 실제 구현

**무엇을**: `utils/notion.py`의 스텁을 실제 Notion API v1 연동 코드로 교체했습니다.
`save_diagnosis_page(snapshot)` 함수가 진단 결과를 Notion 데이터베이스 페이지로 저장하고 생성된 페이지 URL을 반환합니다.
**페이지 구조**: 제목(날짜 + 프롬프트 미리보기) / 맥락·점수 / Before·After 코드 블록 / 변경 이유
**왜**: DB 스키마에 의존하지 않도록 `Name`(title)만 properties로 쓰고 나머지는 body 블록으로 처리해 어떤 Notion DB에도 바로 연결할 수 있습니다.

---

### F-06-2 — Notion 저장 버튼 + 1회 재시도 + Fallback 연결

**무엇을**: 진단 결과 화면에 "Notion에 저장" 버튼을 추가했습니다.
- `NOTION_API_KEY`와 `NOTION_DB_ID` 모두 설정된 경우에만 버튼이 표시됩니다 (Rule 11 준수).
- 저장 성공 시 `st.toast("Notion에 저장되었습니다!")` 표시.
- 실패 시 `st.error` + 세션 상태(`notion_save_status = "error"`) 저장.
- 내부적으로 1초 간격 1회 재시도 후 그래도 실패 시 오류 처리.

---

### fix — Notion DB 스키마 동적 매핑 확장

**무엇을**: 저장 전 DB 스키마를 조회해 컬럼 타입(title/number/select/multi_select/rich_text)에 맞춰 동적으로 매핑하도록 확장했습니다.
**왜**: DB 컬럼명이 일부 변경되거나 워크스페이스마다 스키마가 달라도 저장 실패를 최소화하기 위해서입니다.

---

### F-05-2 — md 다운로드 Fallback 자동 활성화 연동

**무엇을**: Notion 저장에 실패한 경우 리포트 다운로드 버튼 위에 `st.info` 안내 메시지를 자동으로 표시합니다.
**왜**: 저장 실패 시 사용자가 데이터를 잃지 않도록 명확한 대체 경로를 안내합니다.

---

### F-01 정책 개편 — 맥락수집 To-be 반영

**무엇을**
- 사이드바에 `프롬프트 명` 필드 추가
- 안내 문구 추가: “추후 프롬프트 다운로드 및 저장 시, 파일명으로 사용됩니다.”
- 기존 `목적` 라벨을 `프롬프트 사용목적`으로 변경
- 입력 정책 강화:
  - 프롬프트 명: 필수, 20자 이하, 영문/숫자/한글 + `-`, `_`, `.`
  - 프롬프트 사용목적: 필수, 100자 이하
  - 진단할 프롬프트: 필수, 500자 이하
  - 개선목적: 1개 이상 선택 필수

**왜**: 회의 To-be 화면설계와 입력 정책을 코드에 정확히 반영하기 위해.

---

### 학습 데이터 파이프라인 추가

**무엇을**
- 신규 파일 `utils/data_pipeline.py`
- 진단 완료 시 실행 로그를 `data/prompt_runs.jsonl`에 append-only 저장
- `total_score` 기반 프롬프트 레벨(1~4) 자동 판정
- 누적 로그를 바탕으로 `data/fewshot_examples.json` 자동 갱신

**왜**: few-shot 수동 관리 의존도를 낮추고, 이후 F-09/레벨 시스템/RAG 확장을 위한 기반 데이터 축적이 필요하기 때문.

---

### F-09 자가개선 루프 1차 구현 + 모델 라우팅

**무엇을**
- 신규 파일 `chains/self_improve_chain.py`
  - 개선 → 재진단 반복
  - 점수 향상 시 keep, 정체/악화 시 조기 종료
  - 최고점 결과 반환
- 신규 파일 `chains/model_router.py`
  - 환경변수 기반 라우팅 설정
  - 점수 임계치(`OPUS_SCORE_THRESHOLD`, 기본 70) 기준으로 rewrite 모델 분기
  - Opus 미사용 환경(`ANTHROPIC_API_KEY` 없음)에서는 OpenAI 경로 자동 fallback
- `main.py` 연동
  - `SELF_IMPROVE_ENABLED=true`일 때만 루프 동작 (기본 OFF)

**왜**: 비용 최적화와 품질 개선 반복을 동시에 달성하기 위해.

---

### Notion few-shot 동적 로드 + 레벨 균형 샘플링

**무엇을**
- `utils/notion.py`에 `load_fewshot_examples_from_notion()` 추가
  - `NOTION_FEWSHOT_ENABLED=true`일 때 Notion DB 조회
  - `NOTION_FEWSHOT_DB_ID` 우선, 없으면 `NOTION_DB_ID` 사용
  - 실패/빈 결과 시 기존 JSON fallback 유지
- `chains/diagnosis_chain.py`
  - `FEWSHOT_SOURCE_NOTION=true`일 때 Notion 우선 로드
  - few-shot 섹션에 레벨/출처 메타 표시
- Notion 수집 예시 선별 정책
  - 레벨(1~4) 균형 우선
  - 레벨 내 고득점 우선
  - `NOTION_FEWSHOT_PER_LEVEL`로 레벨별 샘플 수 조절

**왜**: 동적 사례를 안정적으로 품질 학습에 반영하면서 편향을 줄이기 위해.

---

## 4. 새로 추가된 파일

| 파일 | 역할 | 추가 이유 |
|------|------|-----------|
| `CLAUDE.md` | 프로젝트 운영 규칙, 기능 체크리스트, 향후 계획, 작업 가이드 | AI 보조 개발 시 일관된 기준 유지 및 진행 상황 추적 |
| `ARCHITECTURE.md` | 전체 파일 구조·데이터 흐름·체인 연결·session_state 키 목록 | 신규 개발자·기획·디자인 팀원이 구조를 빠르게 파악할 수 있도록 |
| `data/fewshot_examples.json` | AI 진단 정확도를 높이는 예시 3건 (개선필요·보통·우수) | 하드코딩 제거, 예시 추가·수정을 코드 없이 가능하게 |
| `utils/notion.py` (실구현) | Notion API v1 연동 — 진단 결과를 DB 페이지로 저장 | F-06-1 스텁 → 실제 구현 교체 |
| `utils/data_pipeline.py` | 학습 원천 로그 적재 + few-shot 자동 갱신 | 수동 운영 최소화 및 확장 기반 마련 |
| `chains/model_router.py` | 점수 기반 모델 라우팅(OpenAI/Opus) | 비용/품질 균형 제어 |
| `chains/self_improve_chain.py` | F-09 자가개선 루프 | 반복 개선 자동화 |

---

## 5. 보류 항목

**보류 항목(차기 작업)**
- 사용자 Notion OAuth 연동(개인 워크스페이스 저장)
- RAG 인덱싱 파이프라인 구축(Notion/로그 기반)
- F-09 루프 전략 다양화(다중 rewrite 전략, 조기 중단 규칙 고도화)

---

## 6. 팀장님이 할 일

### `.env` 파일에 필요한 항목

```
# 필수 — 없으면 앱이 시작되지 않습니다
OPENAI_API_KEY=sk-...

# 선택 — 없으면 기본값으로 동작합니다
OPENAI_MODEL=gpt-4o-mini     # 기본 진단/개선 경로에 사용할 OpenAI 모델
OPENAI_TEMPERATURE=0.2       # 응답 다양성 조절 0.0~1.0 (기본: 0.2)

# Notion 자동 저장 사용 시 — 두 값 모두 있어야 버튼이 표시됩니다
NOTION_API_KEY=secret_...    # Notion Integration 토큰
NOTION_DB_ID=...             # 진단 결과를 저장할 Notion 데이터베이스 ID

# 자가개선 루프 + 모델 라우팅 (기본 OFF)
SELF_IMPROVE_ENABLED=false
SELF_IMPROVE_MAX_ITERS=3
OPUS_SCORE_THRESHOLD=70
OPENAI_CHEAP_MODEL=gpt-4o-mini
OPENAI_DIAGNOSIS_MODEL=gpt-4o-mini
OPENAI_REWRITE_MODEL=gpt-4o-mini
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL_OPUS=claude-3-opus-20240229

# Notion few-shot 동적 로드 (기본 OFF)
FEWSHOT_SOURCE_NOTION=false
NOTION_FEWSHOT_ENABLED=false
NOTION_FEWSHOT_DB_ID=
NOTION_FEWSHOT_PER_LEVEL=2
```

> `NOTION_API_KEY` / `NOTION_DB_ID` 중 하나라도 없으면 "Notion에 저장" 버튼이 숨겨집니다. Notion 없이도 앱은 정상 동작합니다.

---

## 7. 참고사항

### UI/UX 동결 규칙
이번 PR에서 레이아웃·색상·버튼 배치 등 시각적 요소는 변경하지 않았습니다. 디자이너가 설정한 화면 구조가 그대로 보존됩니다.

### UI 변경 이력
| 항목 | 변경 내용 | 이유 |
|------|-----------|------|
| F-04-2 복사 완료 피드백 | `st.toast` 대신 버튼 텍스트 2초 변경 | `st.toast`는 서버 코드라 JS 클릭 이벤트에서 호출 불가 (기술적 한계) |
| F-02-4 진단 원인 표시 범위 | 14점 이하 항목만 표시 → 전체 항목 표시 | 팀장님 피드백 반영, 회의 결정사항 (2026-04-09) |

---

## 8. 향후 업그레이드 계획

이번 PR 이후 차기 기능 후보입니다. 상세 설계는 `CLAUDE.md` 및 `ARCHITECTURE.md`를 참고하세요.

### F-09 고도화 — 자가개선 전략 확장
- 현재: 1차 루프(반복 개선 + 최고점 선택) 구현 완료
- 차기:
  - 다중 rewrite 전략 탐색(구조화 우선, 제약 강화 우선 등)
  - iteration별 성능 로그 분석/시각화
  - 루프 중간 품질 기준 기반 조기 종료 고도화

### Notion 기반 동적 학습/RAG 준비
- 현재: Notion few-shot 동적 로드(기본 OFF) + 레벨 균형 선별 구현
- 차기:
  - Notion 원천 데이터 임베딩 파이프라인
  - 레벨/점수/도메인 메타 필터 기반 retrieval
  - 사용자별 개인화 사례 추천

### 프롬프트 4단계 레벨 시스템
- 현재: 총점 기반 레벨 자동 판정(1~4) 및 학습 로그 저장 구현
- 차기:
  - 레벨별 개선 가이드/템플릿 자동 제시
  - 레벨별 취약 기준 분석 리포트
  - 누적 히스토리 패턴 기반 템플릿 자동 생성

> 상세 설계 및 구현 가이드: [`CLAUDE.md`](./CLAUDE.md) "향후 업그레이드 계획" 섹션, [`ARCHITECTURE.md`](./ARCHITECTURE.md) "F-09 자가개선 루프 연결 위치" 섹션 참고.

---

이 문서는 CLAUDE.md 기반으로 생성됐습니다.
