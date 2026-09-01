# KOSIS 검색 전략 4종 비교 실험 결과

평가셋: 골든셋 70건 (claim → 정답 table_id)


## 주요 지표

| Strategy | Recall@1 | Recall@10 | Recall@100 | MRR | NDCG@10 | Avg Cand | Avg Latency | P95 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Baseline | 5.7% | 32.1% | 57.9% | 0.152 | 0.185 | 100 | 284ms | 918ms |
| Item | 1.4% | 7.9% | 39.3% | 0.041 | 0.042 | 100 | 1399ms | 4667ms |
| Item → Axis | 1.4% | 8.6% | 39.3% | 0.050 | 0.053 | 100 | 209ms | 1072ms |
| Hybrid/RRF | 7.1% | 15.0% | 62.1% | 0.110 | 0.103 | 100 | 8681ms | 12353ms |

## 보조 지표

| Strategy | Hit@1 | Hit@10 | Hit@100 | NDCG@100 | 정답 평균 rank | 정답 중앙 rank | Gold miss |
|---|--:|--:|--:|--:|--:|--:|--:|
| Baseline | 5.7% | 32.9% | 58.6% | 0.237 | 16.0 | 7 | 41.4% |
| Item | 1.4% | 8.6% | 40.0% | 0.104 | 32.9 | 24 | 60.0% |
| Item → Axis | 1.4% | 8.6% | 40.0% | 0.110 | 41.0 | 26 | 60.0% |
| Hybrid/RRF | 7.1% | 15.7% | 62.9% | 0.202 | 28.3 | 18 | 37.1% |

## 후보 축소 (Candidate Reduction, 평균)

- **Baseline**: dense_tables 100
- **Item**: items 1000 → tables_from_items 657 → final 100
- **Item → Axis**: items 1000 → tables_from_items 657 → item_wide 395 → axis_passed 29 → final 100
- **Hybrid/RRF**: dense 100 → item 395 → lexical 52 → axis 24 → union 459 → final 100
