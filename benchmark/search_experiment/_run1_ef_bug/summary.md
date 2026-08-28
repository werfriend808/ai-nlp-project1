# KOSIS 검색 전략 4종 비교 실험 결과

평가셋: 골든셋 70건 (claim → 정답 table_id)


## 주요 지표

| Strategy | Recall@1 | Recall@10 | Recall@100 | MRR | NDCG@10 | Avg Cand | Avg Latency | P95 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| Baseline | 2.9% | 24.3% | 53.6% | 0.114 | 0.133 | 100 | 154ms | 664ms |
| Item | 1.4% | 7.9% | 35.0% | 0.041 | 0.042 | 80 | 167ms | 934ms |
| Item → Axis | 1.4% | 10.0% | 35.0% | 0.053 | 0.057 | 80 | 95ms | 1019ms |
| Hybrid/RRF | 5.7% | 13.6% | 56.4% | 0.096 | 0.090 | 100 | 8527ms | 12318ms |

## 보조 지표

| Strategy | Hit@1 | Hit@10 | Hit@100 | NDCG@100 | 정답 평균 rank | 정답 중앙 rank | Gold miss |
|---|--:|--:|--:|--:|--:|--:|--:|
| Baseline | 2.9% | 24.3% | 54.3% | 0.201 | 15.3 | 11 | 45.7% |
| Item | 1.4% | 8.6% | 35.7% | 0.096 | 32.4 | 19 | 64.3% |
| Item → Axis | 1.4% | 10.0% | 35.7% | 0.107 | 32.6 | 19 | 64.3% |
| Hybrid/RRF | 5.7% | 14.3% | 57.1% | 0.183 | 23.1 | 18 | 42.9% |

## 후보 축소 (Candidate Reduction, 평균)

- **Baseline**: dense_tables 100
- **Item**: items 131 → tables_from_items 95 → final 80
- **Item → Axis**: items 131 → tables_from_items 95 → item_wide 95 → axis_passed 8 → final 80
- **Hybrid/RRF**: dense 100 → item 95 → lexical 49 → axis 15 → union 178 → final 100
