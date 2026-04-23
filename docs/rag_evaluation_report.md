# RAG 정량적 성능 지표 보고서 (F-16-4)

> 발표용 최종 초안. 자동 생성되는 집계 표는 `docs/rag_evaluation_stats.md`에 별도로 저장되며, 이 보고서(`rag_evaluation_report.md`)는 해석·내러티브 전용 수기 문서로 유지한다.

---

## 1. 실험 목적

1. Prompt Clinic에 도입된 **RAG(Retrieval-Augmented Generation)** 가 진단·재작성 품질에 실제로 얼마나 기여하는지 정량 검증한다.
2. 개선 목표 중 **"토큰 줄이기"** 가 현재 파이프라인에서 실제 토큰 절감으로 이어지는지, RAG 주입이 이를 어떻게 변화시키는지 확인한다.

---

## 2. 데이터 출처 및 조건

| 조건 | 소스 | 건수 | 수집 방식 |
|---|---|---|---|
| RAG ON (운영) | `data/prompt_runs.jsonl` (F-13-5 이후, `before_token_count > 0` 필터) | **18건** | 실사용자 진단 결과 — `.env`의 `RAG_ENABLED=true` 상태에서 누적 |
| RAG OFF (비교) | `data/rag_off_eval.jsonl` | **18건** | `scripts/eval_rag_off.py`로 **동일 `user_prompt` / `purpose` / `improvement_goals`** 를 `RAG_ENABLED=false`로 재실행 |

### 공통 조건
- 모델: `gpt-4o` (진단·재작성 동일) + `claude-3-opus-20240229` (Phase 2 자가개선)
- temperature: `0.2`
- `SELF_IMPROVE_ENABLED=true` (양쪽 동일) — **모든 레코드가 최대 6 iteration(Phase 1 OpenAI 3회 + Phase 2 Opus 3회)까지 수행**
- 점수는 모두 **개선결과 재진단 점수** (main.py와 동일 흐름)

### 표본 한계 (발표 시 명시)
- n=18 / "토큰 줄이기" 서브셋 n=7 — **통계적 유의성은 약함**. 경향 확인 수준으로 해석.
- LLM 비결정성(같은 프롬프트도 매 호출 결과가 달라짐) → 단일 재실행 결과는 ±오차를 가질 수 있음.
- RAG ON 데이터는 수집 기간 중 `RAG_ENABLED=true`였다는 전제에 기반. `rag_enabled` 메타 필드는 F-16-4 이후 버전부터 기록 예정.

---

## 3. 측정 지표

| 지표 | 정의 | 소스 필드 |
|---|---|---|
| 종합 점수 (total_score) | 4항목 가중 합산, 0~100 | `total_score` |
| 항목별 점수 | 명확성 / 제약조건 / 출력형식 / 맥락반영도 | `scores.clarity` 등 |
| 원본 토큰 수 | `user_prompt`의 tiktoken(gpt-4o) 토큰 | `before_token_count` |
| 개선 토큰 수 | `improved_prompt`의 tiktoken 토큰 | `after_token_count` |
| 토큰 절감율 | `(before − after) / before × 100 %` (음수 = 증가) | 계산값 |
| 전체 파이프라인 지연시간 | context + diagnosis + self-improve 루프 + 재진단 합계 (RAG OFF 재실행분 측정) | `latency_total_ms` |

---

## 4. 결과 1 — 전체 표본 RAG 전후 비교 (n=18)

| 조건 | n | total_score 평균 | 중앙 | clarity | constraint | output_format | context | before tok | after tok | 토큰 절감율 평균(%) | 평균 latency(ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RAG ON (운영 로그) | 18 | **96.9** | 97.5 | 25.0 | 23.9 | 24.7 | **23.3** | 24.8 | 1015.2 | −8812.0 | (측정 없음) |
| RAG OFF (재실행) | 18 | 96.1 | 96.5 | 24.6 | **24.2** | 24.7 | 22.6 | 24.8 | 978.1 | −7311.6 | ≈154,994 |
| **차이 (ON − OFF)** | — | **+0.8** | +1.0 | +0.4 | **−0.3** | 0.0 | **+0.7** | 0 | **+37.1** | −1500.4 | — |

### 관찰
1. **총점 차이가 0.8점에 불과**. 자가개선 루프가 최대 6 iteration까지 풀가동되면서 RAG의 기여분이 희석됨.
2. RAG가 가장 두드러지게 기여한 축은 **맥락반영도(+0.7)**. 이는 RAG의 설계 의도(도메인 지식·가이드 주입)와 일치.
3. **제약조건(constraint)은 RAG OFF가 오히려 +0.3 우세**. 외부 가이드가 제약 지시문에는 크게 기여하지 않고, 오히려 노이즈로 작용했을 가능성.
4. **토큰 팽창의 주범이 RAG가 아님**. 원본 24.8토큰 → 최종 978~1015토큰(**약 40배 증가**)인데, ON/OFF 차이는 37토큰(약 4%)에 불과. **진짜 주범은 자가개선 루프**가 각 iteration마다 제약·맥락·출력형식 조항을 누적 주입하는 구조.
5. 지연시간은 OFF만 측정 가능했는데 **건당 평균 2분 30초**. 모든 레코드가 6회 iteration까지 돌아간 탓.

---

## 5. 결과 2 — "토큰 줄이기" 서브셋 비교 (n=7)

| 조건 | n | total_score 평균 | 중앙 | clarity | constraint | output_format | context | before tok | after tok | 토큰 절감율 평균(%) | 평균 latency(ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RAG ON / 토큰 줄이기 | 7 | **95.7** | 95.0 | 25.0 | 25.0 | 24.3 | 21.4 | 41.9 | **375.9** | **−930.9** | (측정 없음) |
| RAG OFF / 토큰 줄이기 | 7 | 95.0 | 94.0 | 25.0 | 24.3 | **25.0** | 20.7 | 41.9 | 569.4 | −1587.6 | ≈118,448 |
| **차이 (ON − OFF)** | — | +0.7 | +1.0 | 0.0 | **+0.7** | **−0.7** | +0.7 | 0 | **−193.4** | **+656.7 pt** | — |

### 관찰 (중요한 반전)
1. 전체 표본에서는 무의미하던 RAG 효과가, **"토큰 줄이기" 서브셋에서는 토큰 측면에서 뚜렷하게 역전**됨:
   - after token: ON 375.9 vs OFF 569.4 → **RAG ON이 193토큰(약 34%) 덜 팽창**
   - 절감율 평균: ON −930.9% vs OFF −1587.6% → **RAG ON이 ~657%p 덜 팽창**
2. 즉 **"토큰 줄이기"를 사용자가 명시했을 때는, RAG 컨텍스트가 "불필요한 장황함"에 대한 레퍼런스로 작용해 오히려 토큰 팽창을 억제**하는 것으로 보임.
3. 그럼에도 원본 대비 여전히 8~10배 팽창 중. 따라서 **"토큰 줄이기" 개선목표 자체는 현 파이프라인에서 실질적으로 동작하지 않음**. RAG가 약간 완화시켜줄 뿐.

### 원인 가설
- `chains/rewrite_chain.py`의 시스템 프롬프트에 **토큰 절약을 지시하는 분기가 없음**. `improvement_goals`는 `self_improve_chain.GOAL_TO_CRITERIA`에서 채점 가중치로만 반영됨.
- 자가개선 루프가 매 iteration마다 "진단→개선" 싸이클을 반복하면서, 모델이 점수 상승을 위해 프롬프트에 제약·맥락 문구를 계속 덧붙이는 구조적 특성.
- 결과: 사용자가 "토큰 줄이기"를 선택해도 최종 출력은 원본의 8~40배 길이.

---

## 6. 정성 비교 (직접 확인 영역)

숫자만으로 포착되지 않는 품질 차이를 직접 확인하려면, `data/prompt_runs.jsonl`과 `data/rag_off_eval.jsonl`에서 동일 `paired_ts`를 가진 레코드 쌍을 찾아 `improved_prompt` 전문을 사이드바이사이드로 비교한다.

| 관점 | 확인 질문 |
|---|---|
| 도메인 정확성 | 의학/법률/코딩 등 학문축 프롬프트에서 RAG ON이 전문 용어·구조를 더 잘 반영하는가? |
| 원본 의도 보존 | `drift_score`(3축 보존도)가 RAG ON에서 더 높은가 낮은가? |
| 불필요한 장황함 | RAG ON에서 "RAG 참고" 표현이 그대로 유출되거나 반복되는 사례는 없는가? |
| 재작성 일관성 | 같은 프롬프트를 반복 실행 시 ON/OFF 어느 쪽이 결과 분산이 큰가? |

추천 확인 페어 (로그에서 점수 갭이 큰 순):
- `인스타그램에 올릴 시 써줘` (orig 32 → final 99, 10 → 1829 토큰 — RAG 유무 장황함 비교 포인트)
- `클로드야 게임 좀 만들어봐라` (orig 25 → 95, 8 → 1993 토큰 — 동일)
- `오목게임 만들어줘` (orig 25 → 95, 5 → 677 토큰 — 도메인=코딩, RAG 기술 지식 반영도 확인)

---

## 7. 결론

### 7-1. RAG의 실제 효과 (발표 1문장 버전)
> **RAG는 자가개선 루프가 적용된 환경에서 총점 기준 약 0.8점(평균 96.9 vs 96.1)의 한계적 기여에 그쳤다. 가장 명확한 효과는 맥락반영도(+0.7)에 국한되었고, 제약조건에서는 오히려 RAG 미적용 쪽이 +0.3 우세였다.**

### 7-2. "토큰 줄이기" 기능의 실효성
> **사용자가 "토큰 줄이기"를 선택해도 최종 프롬프트는 원본 대비 8~10배로 팽창한다. 다만 이 서브셋에서 RAG ON은 평균 193토큰(약 34%) 덜 팽창시키는 반대 방향의 효과를 보였다.**

### 7-3. 시사점 및 후속 작업 제안 (F-16-4 연계)
1. **RAG의 ROI 재검토 필요**: 자가개선 루프와 RAG가 둘 다 켜진 현 운영 조건에서는 RAG의 품질 기여가 매우 작음. 루프 OFF 환경에서 RAG 효과를 재측정하는 **별도 실험**이 유의미.
2. **토큰 줄이기 기능 실제 구현**: `chains/rewrite_chain.py`에 `improvement_goals` 분기를 추가해 **"원문 의도 보존 + 수사/중복 제거 + 최소 토큰"** 지시문을 시스템 프롬프트에 주입. 현재는 가중치만 있고 동작 없음.
3. **자가개선 루프 조기 종료 조건**: 모든 레코드가 최대 6 iteration까지 풀로 돌고 있음. 점수 임계 도달 시 조기 종료 로직 추가로 지연시간(평균 2분 30초)과 토큰 사용량 모두 개선 여지가 큼.
4. **운영 jsonl 필드 확장**: `rag_enabled`, `rewrite_latency_ms` 필드 추가해 이후 운영 데이터에서도 실시간 집계 가능하게.

---

## 8. 재현 절차

```bash
# 0) 사전 조건: .env에 OPENAI_API_KEY, ANTHROPIC_API_KEY, RAG_ENABLED=true, SELF_IMPROVE_ENABLED=true
#    chroma_db/ 인덱스 구축 완료 (utils/vector_store.build_index)

# 1) RAG OFF 재실행 (18건 전체, 자가개선 루프 포함 — 30~60분 소요)
python -m scripts.eval_rag_off

# 1-1) 중복 프롬프트 제외
python -m scripts.eval_rag_off --dedupe

# 1-2) 스모크 테스트 (앞 3건만)
python -m scripts.eval_rag_off --limit 3

# 2) 구버전 레코드(혹시 혼입) 정리
python -m scripts.cleanup_rag_off

# 3) 자동 집계 (이 보고서와 별도 파일로 출력 — 덮어쓰기 방지)
python -m scripts.summarize_rag_eval --markdown docs/rag_evaluation_stats.md
```

### 권장 실험 루틴
1. `.env` 원본 상태(`RAG_ENABLED=true`) 유지 — 스크립트가 프로세스 내부에서만 false로 덮어쓴다.
2. `scripts/eval_rag_off.py` 1차 실행으로 베이스라인 확보 (≈30~60분).
3. (선택) 핵심 7건(토큰 줄이기)만 `--limit 7` + 수동 반복 실행 3회 → 평균 내면 비결정성 보정.
4. `scripts/cleanup_rag_off` 으로 혹시 남은 구버전 레코드 제거.
5. `scripts/summarize_rag_eval --markdown docs/rag_evaluation_stats.md` 로 표 생성.
6. 본 보고서의 § 4·5 수치 갱신 시에는 `rag_evaluation_stats.md`의 값을 수동으로 가져와 반영 (스크립트는 이 보고서를 직접 덮어쓰지 못하도록 가드 설치됨).
