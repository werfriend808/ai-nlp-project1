# KOSIS 검색 전략 4종 비교 실험 결과

평가셋: 골든셋 70건 (claim → 정답 table_id)


## 주요 지표

| Strategy | Recall@1 | Recall@10 | Recall@100 | MRR | NDCG@10 | Avg Cand | Avg Latency | P95 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Baseline | 4.3% | 25.7% | 55.0% | 0.128 | 0.147 | 100 | 265ms | 927ms |
| Item | 1.4% | 7.9% | 39.3% | 0.041 | 0.042 | 100 | 1092ms | 3773ms |
| Item → Axis | 1.4% | 8.6% | 39.3% | 0.050 | 0.053 | 100 | 146ms | 1088ms |
| Hybrid/RRF | 5.7% | 15.0% | 59.3% | 0.098 | 0.094 | 100 | 8535ms | 12349ms |

## 보조 지표

| Strategy | Hit@1 | Hit@10 | Hit@100 | NDCG@100 | 정답 평균 rank | 정답 중앙 rank | Gold miss |
|---|--:|--:|--:|--:|--:|--:|--:|
| Baseline | 4.3% | 25.7% | 55.7% | 0.214 | 15.7 | 11 | 44.3% |
| Item | 1.4% | 8.6% | 40.0% | 0.104 | 32.9 | 24 | 60.0% |
| Item → Axis | 1.4% | 8.6% | 40.0% | 0.110 | 41.0 | 26 | 60.0% |
| Hybrid/RRF | 5.7% | 15.7% | 60.0% | 0.189 | 25.3 | 16 | 40.0% |

## 후보 축소 (Candidate Reduction, 평균)

- **Baseline**: dense_tables 100
- **Item**: items 1000 → tables_from_items 657 → final 100
- **Item → Axis**: items 1000 → tables_from_items 657 → item_wide 395 → axis_passed 29 → final 100
- **Hybrid/RRF**: dense 100 → item 395 → lexical 49 → axis 24 → union 457 → final 100

## 언어별 분해 — 이번 실험의 진짜 병목

정답표 이름이 영어인지 한글인지로 나눠 보면 전략 차이보다 훨씬 큰 격차가 나온다.

| Strategy | 한글 표명 Recall@100 | 영어 표명 Recall@100 |
|---|--:|--:|
| Baseline | 12/13 = 92.3% | 27/57 = 47.4% |
| Item | 2/13 = 15.4% | 26/57 = 45.6% |
| Item → Axis | 2/13 = 15.4% | 26/57 = 45.6% |
| Hybrid/RRF | 11/13 = 84.6% | 31/57 = 54.4% |

- DB 전체 success 표 279,526건 중 영문 표명 **2,067건 (0.7%)**
- 그런데 골든셋 정답표 70건 중 영문은 **57건** — 뉴스가 인용하는 주요 국가통계표에 영문이 몰려 있다.
- 이 문제는 2026-08-18에 이미 발견됐고 `agent/kosis/patch_english_table_names.py`가 만들어졌으나, 그 스크립트는 구 파이프라인의 `data/vdb_pending.jsonl`을 고치는 것이라 이번 재구축 DB에는 적용되지 않았다.

