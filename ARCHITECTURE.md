# Prompt Clinic — 아키텍처 문서

> 개발자뿐 아니라 기획·디자인 팀원도 읽을 수 있도록 작성했습니다.
> "이 앱이 어떻게 생겼고, 데이터가 어떻게 흘러가는지"를 한눈에 보여줍니다.

---

## 1. 전체 파일 구조

```
prompt-clinic/
│
├── main.py                     # 앱의 시작점. UI 렌더링 + 세션 관리
│
├── chains/                     # AI 처리 로직 (LangChain 체인)
│   ├── pipeline.py             # 세 체인을 하나로 조립
│   ├── context_chain.py        # Chain 1 — 맥락 프로필 생성
│   ├── diagnosis_chain.py      # Chain 2 — 프롬프트 품질 진단
│   └── rewrite_chain.py        # Chain 3 — 개선 프롬프트 생성
│
├── utils/
│   └── notion.py               # Notion API 연동 (진단 결과 저장)
│
├── data/
│   └── fewshot_examples.json   # AI 진단 정확도를 높이는 예시 3건
│
├── .env                        # API 키 모음 (git에 올라가지 않음)
└── requirements.txt            # 설치 필요 라이브러리 목록
```

---

## 2. 사용자 입력 → 결과 출력 데이터 흐름

아래는 사용자가 "진단 시작" 버튼을 누른 순간부터 결과가 화면에 표시될 때까지의 흐름입니다.

```
[사용자]
  │
  │  ① 입력
  │    - 목적 (텍스트)
  │    - 출력 형식 (글/리스트/표/코드/JSON)
  │    - 개선 목적 (다중 선택)
  │    - 진단할 프롬프트 (텍스트)
  │
  ▼
[main.py — 입력 검증]
  │  최소 10자 미만이면 진단 차단, 오류 메시지 표시
  │
  ▼
[Chain 1: context_chain.py — 맥락 프로필 생성]
  │  입력: 목적 + 출력형식 + 개선목적
  │  출력: ContextProfile { purpose, output_format, improvement_goals, context_summary }
  │  (GPT-4o 호출 1회)
  │
  ▼
[Chain 2: diagnosis_chain.py — 품질 진단]
  │  입력: 맥락 프로필 + 원본 프롬프트
  │  출력: DiagnosisResult { clarity, constraint, output_format, context } 각 0~25점
  │  (GPT-4o 호출 1회, Few-shot 3개 참고)
  │
  ▼
[main.py — 가중치 계산]
  │  apply_goal_weights(): 개선 목적에 따라 관련 항목 +5점 (상한 25)
  │  종합 점수(0~100) → 등급 판정 (🟢 우수 / 🟡 보통 / 🔴 개선필요)
  │
  ▼
[Chain 3: rewrite_chain.py — 개선 프롬프트 생성]
  │  입력: 맥락 프로필 + 진단 결과 + 원본 프롬프트
  │  출력: RewriteResult { improved_prompt, changes[] }
  │  (GPT-4o 호출 1회)
  │
  ▼
[main.py — 결과 저장 & 렌더링]
  │  session_state.last_snapshot 에 저장
  │  session_state.history 에 append
  │  화면: Before/After 비교 + 점수 + 변경 이유
  │
  ▼
[사용자 선택]
  ├── 프롬프트 복사
  ├── 개선된 프롬프트로 재진단 (루프)
  ├── Notion에 저장 → utils/notion.py 호출
  └── 리포트 다운로드 (.md)
```

---

## 3. 세 체인의 연결 구조

세 체인은 **순서대로** 실행됩니다. 앞 체인의 출력이 다음 체인의 입력에 합쳐집니다.

```
사용자 입력 4가지
(purpose, output_format, improvement_goals, user_prompt)
         │
         ▼
┌─────────────────────────────┐
│  Chain 1: context_chain     │  → ContextProfile 생성
│  프롬프트 의도를 구조화      │    (목적·형식·맥락 한줄 요약)
└─────────────────────────────┘
         │  +context_profile 추가
         ▼
┌─────────────────────────────┐
│  Chain 2: diagnosis_chain   │  → DiagnosisResult 생성
│  4개 기준으로 점수 채점      │    (명확성·제약·형식·맥락 각 0~25)
│  Few-shot 예시 3개 참고      │
└─────────────────────────────┘
         │  +diagnosis 추가
         ▼
┌─────────────────────────────┐
│  Chain 3: rewrite_chain     │  → RewriteResult 생성
│  약점 개선 프롬프트 생성     │    (improved_prompt + changes[])
└─────────────────────────────┘
```

각 체인 내부 구조 (동일 패턴):

```
prep_*_input()          →  ChatPromptTemplate  →  ChatOpenAI  →  JsonOutputParser
(dict 전처리 함수)          (시스템+사용자 메시지)    (GPT-4o 호출)    (Pydantic 검증)
```

---

## 4. session_state 키 목록

Streamlit은 페이지를 새로 그릴 때마다 코드 전체를 재실행합니다.
`session_state`는 이 재실행 사이에 데이터를 유지하는 저장소입니다.

| 키 | 타입 | 역할 |
|----|------|------|
| `history` | `list[dict]` | 이번 세션의 전체 진단 이력. append-only (삭제·덮어쓰기 금지) |
| `last_snapshot` | `dict \| None` | 가장 최근 진단 결과 전체. 화면 렌더링에 사용 |
| `rediagnose_prompt` | `str` | "재진단" 버튼 클릭 시 프롬프트 입력창에 채울 텍스트 |
| `auto_diagnose` | `bool` | 재진단 자동 실행 플래그. True면 페이지 로드 즉시 진단 시작 |
| `lc_chat_history` | `InMemoryChatMessageHistory` | 원본·개선 프롬프트 쌍 저장. F-09 자가개선 루프에서 활용 예정 |
| `notion_save_status` | `None \| "success" \| "error"` | Notion 저장 결과. "error"면 다운로드 fallback 안내 표시 |

**snapshot 딕셔너리 구조** (history 각 항목 / last_snapshot):

```
{
  "ts":               datetime,       # 진단 시각
  "purpose":          str,            # 목적 입력값
  "output_format":    str,            # 출력 형식 선택값
  "improvement_goals": list[str],     # 개선 목적 선택값
  "user_prompt":      str,            # 원본 프롬프트
  "context_profile":  dict,           # Chain 1 출력
  "diagnosis_raw":    dict,           # Chain 2 출력 (가중치 전)
  "weighted":         dict,           # 가중치 적용 후 점수·등급
  "rewrite":          dict,           # Chain 3 출력 (improved_prompt + changes)
}
```

---

## 5. 환경변수 목록

`.env` 파일에 설정합니다. 앱 시작 시 자동으로 읽습니다.

| 변수명 | 필수 여부 | 기본값 | 역할 |
|--------|----------|--------|------|
| `OPENAI_API_KEY` | **필수** | 없음 | OpenAI GPT-4o 호출 인증키. 없으면 앱이 동작하지 않음 |
| `OPENAI_MODEL` | 선택 | `gpt-4o` | 사용할 OpenAI 모델명 |
| `OPENAI_TEMPERATURE` | 선택 | `0.2` | 응답 다양성 (0.0 = 일관성 최대, 1.0 = 창의성 최대) |
| `NOTION_API_KEY` | 선택 | 없음 | Notion Integration 토큰. 없으면 "Notion에 저장" 버튼 숨김 |
| `NOTION_DB_ID` | 선택 | 없음 | 저장 대상 Notion 데이터베이스 ID. API_KEY와 함께 필요 |

---

## 6. 향후 F-09 자가개선 루프 연결 위치

현재 구현된 흐름에 F-09가 어떻게 연결될지를 보여줍니다.
실선(─)은 현재 구현, 점선(╌)은 미래 계획입니다.

```
사용자 입력
    │
    ▼
Chain 1 → Chain 2 → Chain 3
                        │
                        │  현재: 결과를 화면에 표시하고 종료
                        │
                        ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
                        ╎  [F-09: 자가개선 루프 — 미착수]              ╎
                        ╎                                              ╎
                        ╎  개선 프롬프트를 다시 Chain 2에 투입          ╎
                        ╎       ↓                                      ╎
                        ╎  total_score 측정                            ╎
                        ╎       ↓                                      ╎
                        ╎  이전보다 점수 올랐으면 keep                  ╎
                        ╎  안 올랐으면 다른 전략으로 재시도             ╎
                        ╎       ↓                                      ╎
                        ╎  N회 반복 후 최고점 프롬프트 반환             ╎
                        ╎                                              ╎
                        ╎  활용 데이터: session_state.lc_chat_history  ╎
                        ╎  구현 위치: chains/self_improve_chain.py     ╎
                        ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
```

---

## 7. 주요 설계 원칙 (왜 이렇게 만들었나)

| 원칙 | 이유 |
|------|------|
| **체인 3개 분리** | 각 단계를 독립적으로 테스트하고 교체할 수 있음. 하나의 체인이 실패해도 다른 체인에 영향 없음 |
| **Pydantic 모델로 파싱** | AI가 잘못된 형식으로 출력해도 즉시 오류 감지. 타입이 보장된 데이터만 화면에 표시 |
| **invoke_with_retry 래핑** | API 오류 시 자동 재시도 (최대 2회). 네트워크 불안정에도 사용자 경험 보호 |
| **session_state append-only** | 히스토리 데이터를 절대 덮어쓰지 않음. 실수로 이전 진단 결과를 잃는 일 방지 |
| **환경변수로 기능 ON/OFF** | Notion 키가 없으면 관련 UI 자체가 숨겨짐. 미완성 기능이 사용자에게 노출되지 않음 |
| **utils/ 레이어 분리** | 외부 API 호출(Notion)은 반드시 utils/를 통해서만. UI 코드가 외부 시스템에 직접 의존하지 않음 |
