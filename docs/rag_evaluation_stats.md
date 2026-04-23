# RAG 정량 평가 집계 (자동 생성)

- RAG ON 데이터: `data/prompt_runs.jsonl` 중 `before_token_count > 0` 필터
- RAG OFF 데이터: `scripts/eval_rag_off.py` 재실행 결과 (`data/rag_off_eval.jsonl`)
- 집계 스크립트: `python -m scripts.summarize_rag_eval --markdown <path>`

### 1. 전체 표본 (RAG ON 18건 vs 동일 조건 RAG OFF 재실행)

| 조건 | n | total_score 평균 | total_score 중앙 | clarity | constraint | output_format | context | before tok | after tok | 토큰 절감율 평균(%) | 평균 latency(ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RAG ON (운영 로그) | 18 | 96.9 | 97.5 | 25.0 | 23.9 | 24.7 | 23.3 | 24.8 | 1015.2 | -8812.0 | 0 |
| RAG OFF (재실행) | 18 | 96.1 | 96.5 | 24.6 | 24.2 | 24.7 | 22.6 | 24.8 | 978.1 | -7311.6 | 154994 |

### 2. 토큰 줄이기 서브셋

| 조건 | n | total_score 평균 | total_score 중앙 | clarity | constraint | output_format | context | before tok | after tok | 토큰 절감율 평균(%) | 평균 latency(ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RAG ON / 토큰 줄이기 | 7 | 95.7 | 95.0 | 25.0 | 25.0 | 24.3 | 21.4 | 41.9 | 375.9 | -930.9 | 0 |
| RAG OFF / 토큰 줄이기 | 7 | 95.0 | 94.0 | 25.0 | 24.3 | 25.0 | 20.7 | 41.9 | 569.4 | -1587.6 | 118448 |

