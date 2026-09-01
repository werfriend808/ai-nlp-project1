# KOSIS Retrieval Recall 개선 실험 (실험 2차)

평가셋 70건 — dev 35 / test 35 (기사 단위 분할)


## 개별 검색기 단독 성능

| Retriever | R@1 | R@10 | R@50 | R@100 | R@200 | Avg Latency |
|---|--:|--:|--:|--:|--:|--:|
| `dense_ctxD2` | 7.1% | 47.9% | 72.1% | **75.0%** | 76.4% | 54ms |
| `dense_ctxD4` | 14.3% | 46.4% | 67.9% | **72.1%** | 76.4% | 15ms |
| `dense_expstruct` | 14.3% | 33.6% | 60.0% | **70.0%** | 75.7% | 20ms |
| `dense_struct` | 7.1% | 35.7% | 59.3% | **66.4%** | 70.7% | 82ms |
| `dense_ctxD3` | 2.9% | 42.1% | 59.3% | **66.4%** | 66.4% | 31ms |
| `dense_expanded` | 12.9% | 33.6% | 57.9% | **65.0%** | 72.1% | 23ms |
| `dense_ctxD5` | 10.0% | 44.3% | 60.7% | **60.7%** | 69.3% | 19ms |
| `dense_full` | 5.7% | 32.1% | 57.9% | **57.9%** | 63.6% | 378ms |
| `dense_measurement` | 4.3% | 27.1% | 49.3% | **53.6%** | 62.1% | 127ms |
| `bm25_expstruct` | 3.6% | 19.3% | 30.0% | **47.1%** | 50.0% | 2ms |
| `item_measurement` | 1.4% | 7.9% | 26.4% | **39.3%** | 42.1% | 1456ms |
| `dense_condition` | 2.9% | 11.4% | 29.3% | **35.0%** | 36.4% | 20ms |
| `dense_population` | 2.9% | 11.4% | 29.3% | **33.6%** | 35.0% | 226ms |
| `item_full` | 5.7% | 10.0% | 29.3% | **33.6%** | 37.9% | 1359ms |
| `bm25_struct` | 1.4% | 14.3% | 25.0% | **32.1%** | 32.1% | 2ms |
| `bm25_measurement` | 1.4% | 11.4% | 23.6% | **30.7%** | 30.7% | 1ms |
| `axis_full` | 4.3% | 11.4% | 17.9% | **22.1%** | 30.0% | 1448ms |
| `axis_condition` | 1.4% | 4.3% | 15.0% | **19.3%** | 20.7% | 1508ms |
| `bm25_full` | 0.0% | 2.9% | 17.9% | **17.9%** | 19.3% | 4ms |
| `bm25_ctxD4` | 0.0% | 4.3% | 12.1% | **12.1%** | 15.0% | 8ms |
| `trgm_struct` | 0.0% | 4.3% | 4.3% | **4.3%** | 4.3% | 5649ms |
| `dense_region` | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% | 3ms |

## 전략 종합 (전체 70건)

| Strategy | R@1 | R@10 | R@50 | R@100 | R@200 | MRR | NDCG@10 | Avg Latency | P95 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| COMBO-C (+struct+expansion) | 11.4% | 52.1% | 72.1% | **77.1%** | 80.0% | 0.249 | 0.307 | 1972ms | 1972ms |
| ALL (dense+bm25+item+axis+ctx) | 4.3% | 50.7% | 72.1% | **76.4%** | 77.9% | 0.174 | 0.242 | 3378ms | 3378ms |
| COMBO-A (dense+bm25+ctx) | 7.1% | 50.7% | 72.1% | **76.4%** | 78.6% | 0.224 | 0.283 | 414ms | 414ms |
| COMBO-B (dense+bm25+item+ctx) | 7.1% | 52.1% | 72.1% | **76.4%** | 77.9% | 0.196 | 0.261 | 1870ms | 1870ms |
| Context D2 (이전+현재) | 7.1% | 47.9% | 72.1% | **75.0%** | 76.4% | 0.200 | 0.258 | 54ms | 54ms |
| Dense + Context D4 | 13.6% | 42.1% | 70.7% | **73.6%** | 76.4% | 0.256 | 0.282 | 393ms | 393ms |
| Dense + Context D5 | 8.6% | 46.4% | 69.3% | **73.6%** | 79.3% | 0.220 | 0.269 | 397ms | 397ms |
| Dense + Ctx D4 + Ctx D5 | 17.1% | 47.9% | 66.4% | **73.6%** | 79.3% | 0.288 | 0.324 | 412ms | 412ms |
| Context D4 (이전+현재+다음) | 14.3% | 46.4% | 67.9% | **72.1%** | 76.4% | 0.253 | 0.295 | 15ms | 15ms |
| Dense + expansion | 7.1% | 47.9% | 70.0% | **71.4%** | 78.6% | 0.185 | 0.243 | 398ms | 398ms |
| Query expansion (구조화) | 14.3% | 33.6% | 60.0% | **70.0%** | 75.7% | 0.207 | 0.227 | 20ms | 20ms |
| Dense + Item | 5.0% | 25.0% | 53.6% | **69.3%** | 69.3% | 0.108 | 0.125 | 1834ms | 1834ms |
| Field (equal) | 8.6% | 37.1% | 57.9% | **69.3%** | 72.1% | 0.177 | 0.214 | 754ms | 754ms |
| Dense + BM25 | 2.9% | 25.7% | 53.6% | **68.6%** | 71.4% | 0.097 | 0.120 | 379ms | 379ms |
| Field (dense+measurement) | 11.4% | 33.6% | 65.0% | **67.9%** | 73.6% | 0.193 | 0.216 | 504ms | 504ms |
| Field (dense 중심) | 7.1% | 34.3% | 59.3% | **66.4%** | 75.0% | 0.169 | 0.199 | 754ms | 754ms |
| Field (measurement 중심) | 8.6% | 40.0% | 57.9% | **66.4%** | 70.7% | 0.197 | 0.239 | 754ms | 754ms |
| Context D3 (현재+다음) | 2.9% | 42.1% | 59.3% | **66.4%** | 66.4% | 0.155 | 0.211 | 31ms | 31ms |
| Query expansion (문장) | 12.9% | 33.6% | 57.9% | **65.0%** | 72.1% | 0.191 | 0.217 | 23ms | 23ms |
| Dense + Item + Axis | 7.1% | 20.7% | 46.4% | **63.6%** | 69.3% | 0.124 | 0.130 | 3342ms | 3342ms |
| Dense + Axis | 5.7% | 29.3% | 46.4% | **62.1%** | 63.6% | 0.135 | 0.164 | 1886ms | 1886ms |
| Context D5 (문단) | 10.0% | 44.3% | 60.7% | **60.7%** | 69.3% | 0.226 | 0.273 | 19ms | 19ms |
| Baseline (dense_full) | 5.7% | 32.1% | 57.9% | **57.9%** | 63.6% | 0.152 | 0.185 | 378ms | 378ms |
| Dense + trigram | 5.7% | 25.0% | 53.6% | **57.9%** | 63.6% | 0.121 | 0.140 | 6027ms | 6027ms |
| Dense + BM25(full+struct) | 1.4% | 26.4% | 46.4% | **57.9%** | 70.0% | 0.084 | 0.114 | 384ms | 384ms |
| BM25 only | 1.4% | 14.3% | 25.0% | **32.1%** | 32.1% | 0.060 | 0.075 | 2ms | 2ms |
| trigram only | 0.0% | 4.3% | 4.3% | **4.3%** | 4.3% | 0.017 | 0.023 | 5649ms | 5649ms |

## dev / test 분리 (과적합 점검)

| Strategy | dev R@100 | test R@100 | 차이 |
|---|--:|--:|--:|
| Dense + Context D5 | 100.0% | 47.1% | -52.9%p |
| Context D2 (이전+현재) | 97.1% | 52.9% | -44.3%p |
| Dense + Ctx D4 + Ctx D5 | 97.1% | 50.0% | -47.1%p |
| ALL (dense+bm25+item+axis+ctx) | 97.1% | 55.7% | -41.4%p |
| COMBO-A (dense+bm25+ctx) | 97.1% | 55.7% | -41.4%p |
| COMBO-B (dense+bm25+item+ctx) | 97.1% | 55.7% | -41.4%p |
| COMBO-C (+struct+expansion) | 97.1% | 57.1% | -40.0%p |
| Context D4 (이전+현재+다음) | 94.3% | 50.0% | -44.3%p |
| Context D5 (문단) | 94.3% | 27.1% | -67.1%p |
| Dense + Context D4 | 94.3% | 52.9% | -41.4%p |
| Context D3 (현재+다음) | 91.4% | 41.4% | -50.0%p |
| Dense + Item | 85.7% | 52.9% | -32.9%p |
| Dense + BM25 | 82.9% | 54.3% | -28.6%p |
| Field (equal) | 82.9% | 55.7% | -27.1%p |
| Dense + expansion | 82.9% | 60.0% | -22.9%p |
| Field (dense+measurement) | 80.0% | 55.7% | -24.3%p |
| Dense + Item + Axis | 77.1% | 50.0% | -27.1%p |
| Field (dense 중심) | 77.1% | 55.7% | -21.4%p |
| Field (measurement 중심) | 77.1% | 55.7% | -21.4%p |
| Query expansion (문장) | 77.1% | 52.9% | -24.3%p |
| Query expansion (구조화) | 77.1% | 62.9% | -14.3%p |
| Dense + BM25(full+struct) | 74.3% | 41.4% | -32.9%p |
| Dense + Axis | 71.4% | 52.9% | -18.6%p |
| Baseline (dense_full) | 68.6% | 47.1% | -21.4%p |
| Dense + trigram | 68.6% | 47.1% | -21.4%p |
| BM25 only | 25.7% | 38.6% | +12.9%p |
| trigram only | 0.0% | 8.6% | +8.6%p |

## Baseline 대비 win / loss (Recall@100 기준)

| Strategy | Baseline 실패 → 성공 | Baseline 성공 → 실패 | 순증 | Gold miss | 정답 중앙 rank |
|---|--:|--:|--:|--:|--:|
| COMBO-C (+struct+expansion) | 13 | 0 | +13 | 20.0% | 4 |
| ALL (dense+bm25+item+axis+ctx) | 13 | 0 | +13 | 21.4% | 6 |
| COMBO-A (dense+bm25+ctx) | 13 | 0 | +13 | 21.4% | 5 |
| COMBO-B (dense+bm25+item+ctx) | 13 | 0 | +13 | 21.4% | 7 |
| Context D2 (이전+현재) | 12 | 0 | +12 | 22.9% | 6 |
| Dense + Context D4 | 11 | 0 | +11 | 22.9% | 8 |
| Dense + Context D5 | 13 | 2 | +11 | 20.0% | 8 |
| Dense + Ctx D4 + Ctx D5 | 13 | 2 | +11 | 20.0% | 5 |
| Context D4 (이전+현재+다음) | 12 | 2 | +10 | 22.9% | 5 |
| Dense + expansion | 11 | 2 | +9 | 21.4% | 9 |
| Query expansion (구조화) | 13 | 5 | +8 | 24.3% | 15 |
| Dense + Item | 8 | 0 | +8 | 30.0% | 24 |
| Field (equal) | 11 | 3 | +8 | 27.1% | 10 |
| Dense + BM25 | 7 | 0 | +7 | 28.6% | 18 |
| Field (dense+measurement) | 9 | 2 | +7 | 25.7% | 16 |
| Field (dense 중심) | 6 | 0 | +6 | 24.3% | 12 |
| Field (measurement 중심) | 11 | 5 | +6 | 28.6% | 6 |
| Context D3 (현재+다음) | 9 | 3 | +6 | 32.9% | 6 |
| Query expansion (문장) | 6 | 1 | +5 | 27.1% | 16 |
| Dense + Item + Axis | 8 | 4 | +4 | 30.0% | 19 |
| Dense + Axis | 3 | 0 | +3 | 35.7% | 13 |
| Context D5 (문단) | 13 | 11 | +2 | 30.0% | 4 |
| Baseline (dense_full) | 0 | 0 | +0 | 35.7% | 9 |
| Dense + trigram | 0 | 0 | +0 | 35.7% | 15 |
| Dense + BM25(full+struct) | 7 | 7 | +0 | 30.0% | 14 |
| BM25 only | 9 | 27 | -18 | 67.1% | 16 |
| trigram only | 0 | 38 | -38 | 95.7% | 3 |

## Ablation — 최고 전략 `COMBO-C (+struct+expansion)` 에서 검색기 하나씩 제거

| 제거한 검색기 | R@100 | 변화 | 지연 |
|---|--:|--:|--:|
| (없음 — 전체) | 77.1% | — | 1972ms |
| −`dense_full` | 77.1% | +0.0%p | 1595ms |
| −`dense_struct` | 78.6% | +1.4%p | 1890ms |
| −`bm25_struct` | 76.4% | -0.7%p | 1970ms |
| −`item_measurement` | 78.6% | +1.4%p | 516ms |
| −`dense_ctxD4` | 77.1% | +0.0%p | 1957ms |
| −`dense_ctxD5` | 77.1% | +0.0%p | 1953ms |
| −`dense_expstruct` | 76.4% | -0.7%p | 1952ms |

## 실패 유형 — Baseline (Recall@100 미포함 29건)

| 유형 | 건수 |
|---|--:|
| 1. 지표명이 claim에 없음 | 9 |
| 2. 표명과 claim 표현이 다름 | 11 |
| 5. item 정보 부족 | 5 |
| 9. table embedding 정보 부족 | 4 |

<details><summary>개별 claim</summary>

| claim | 정답표 | 유형 | Baseline rank | 최고전략 rank | gold를 찾은 검색기 |
|---|---|---|--:|--:|---|
| 5-03 | DT_1DA7E06S | 2. 표명과 claim 표현이 다름 | miss | 4 | `dense_population`, `dense_condition`, `dense_struct`, `dense_ctxD2` |
| 5-07a | DT_1DA7024S | 1. 지표명이 claim에 없음 | 172 | 3 | `dense_measurement`, `dense_population`, `dense_struct`, `dense_ctxD2` |
| 5-07b | DT_1DA7024S | 1. 지표명이 claim에 없음 | 172 | 7 | `dense_measurement`, `dense_struct`, `dense_ctxD2`, `dense_ctxD3` |
| 5-14a | DT_1DA7147S | 9. table embedding 정보 부족 | miss | 1 | `dense_measurement`, `dense_struct`, `dense_ctxD2`, `dense_ctxD3` |
| 5-14b | DT_1DA7147S | 9. table embedding 정보 부족 | miss | 1 | `dense_measurement`, `dense_struct`, `dense_ctxD2`, `dense_ctxD3` |
| 5-15a | DT_1DA7147S | 9. table embedding 정보 부족 | miss | 2 | `dense_measurement`, `dense_struct`, `dense_ctxD2`, `dense_ctxD3` |
| 5-15b | DT_1DA7147S | 9. table embedding 정보 부족 | miss | 2 | `dense_measurement`, `dense_struct`, `dense_ctxD2`, `dense_ctxD3` |
| 19-01a | DT_1JH20202 | 2. 표명과 claim 표현이 다름 | miss | 111 | `dense_expstruct`, `bm25_struct`, `bm25_measurement`, `bm25_expstruct` |
| 19-07 | DT_1K41012 | 1. 지표명이 claim에 없음 | miss | 3 | `dense_measurement`, `dense_population`, `dense_condition`, `dense_struct` |
| 19-10a | DT_1JH20202 | 2. 표명과 claim 표현이 다름 | miss | miss | 없음 |
| 19-10b | DT_1JH20202 | 2. 표명과 claim 표현이 다름 | miss | miss | 없음 |
| 19-12 | DT_1C8015 | 2. 표명과 claim 표현이 다름 | miss | 3 | `dense_measurement`, `dense_population`, `dense_condition`, `dense_struct` |
| 19-13 | DT_1C8015 | 2. 표명과 claim 표현이 다름 | miss | 1 | `dense_measurement`, `dense_population`, `dense_condition`, `dense_struct` |
| 30-01 | DT_1B34E13 | 5. item 정보 부족 | miss | miss | 없음 |
| 30-02 | DT_1B34E13 | 5. item 정보 부족 | 169 | miss | `bm25_expstruct` |
| 30-03 | DT_1B34E13 | 5. item 정보 부족 | miss | miss | `bm25_expstruct` |
| 30-04 | DT_1B34E13 | 5. item 정보 부족 | miss | miss | 없음 |
| 30-05 | DT_1B34E13 | 5. item 정보 부족 | miss | miss | `bm25_expstruct` |
| 30-06a | DT_1B34E13 | 1. 지표명이 claim에 없음 | miss | miss | `bm25_expstruct` |
| 30-06b | DT_1B34E13 | 1. 지표명이 claim에 없음 | miss | miss | `bm25_expstruct` |
| 30-06c | DT_1B34E13 | 1. 지표명이 claim에 없음 | miss | miss | `bm25_expstruct` |
| 30-06d | DT_1B34E13 | 1. 지표명이 claim에 없음 | miss | miss | `bm25_expstruct` |
| 30-07e | DT_1B34E13 | 1. 지표명이 claim에 없음 | miss | miss | `bm25_expstruct` |
| 30-08a | DT_1B34E01 | 2. 표명과 claim 표현이 다름 | miss | miss | `dense_expanded` |
| 30-08b | DT_1B34E01 | 2. 표명과 claim 표현이 다름 | miss | miss | `dense_expanded`, `dense_expstruct` |
| 36-01 | DT_1NTA2002 | 2. 표명과 claim 표현이 다름 | 139 | 54 | `dense_ctxD2`, `dense_ctxD3`, `dense_ctxD4`, `dense_ctxD5` |
| 36-03 | DT_1NTA2002 | 1. 지표명이 claim에 없음 | miss | 38 | `dense_ctxD2`, `dense_ctxD3`, `dense_ctxD4`, `dense_ctxD5` |
| 36-07 | DT_1NTA2002 | 2. 표명과 claim 표현이 다름 | miss | 24 | `dense_measurement`, `dense_population`, `dense_condition`, `dense_struct` |
| 36-10 | DT_1NTA2002 | 2. 표명과 claim 표현이 다름 | miss | 184 | `dense_ctxD5` |

</details>


## 실패 유형 — 최고 전략 (COMBO-C (+struct+expansion)) (Recall@100 미포함 16건)

| 유형 | 건수 |
|---|--:|
| 1. 지표명이 claim에 없음 | 5 |
| 2. 표명과 claim 표현이 다름 | 6 |
| 5. item 정보 부족 | 5 |

<details><summary>개별 claim</summary>

| claim | 정답표 | 유형 | Baseline rank | 최고전략 rank | gold를 찾은 검색기 |
|---|---|---|--:|--:|---|
| 19-01a | DT_1JH20202 | 2. 표명과 claim 표현이 다름 | miss | 111 | `dense_expstruct`, `bm25_struct`, `bm25_measurement`, `bm25_expstruct` |
| 19-10a | DT_1JH20202 | 2. 표명과 claim 표현이 다름 | miss | miss | 없음 |
| 19-10b | DT_1JH20202 | 2. 표명과 claim 표현이 다름 | miss | miss | 없음 |
| 30-01 | DT_1B34E13 | 5. item 정보 부족 | miss | miss | 없음 |
| 30-02 | DT_1B34E13 | 5. item 정보 부족 | 169 | miss | `bm25_expstruct` |
| 30-03 | DT_1B34E13 | 5. item 정보 부족 | miss | miss | `bm25_expstruct` |
| 30-04 | DT_1B34E13 | 5. item 정보 부족 | miss | miss | 없음 |
| 30-05 | DT_1B34E13 | 5. item 정보 부족 | miss | miss | `bm25_expstruct` |
| 30-06a | DT_1B34E13 | 1. 지표명이 claim에 없음 | miss | miss | `bm25_expstruct` |
| 30-06b | DT_1B34E13 | 1. 지표명이 claim에 없음 | miss | miss | `bm25_expstruct` |
| 30-06c | DT_1B34E13 | 1. 지표명이 claim에 없음 | miss | miss | `bm25_expstruct` |
| 30-06d | DT_1B34E13 | 1. 지표명이 claim에 없음 | miss | miss | `bm25_expstruct` |
| 30-07e | DT_1B34E13 | 1. 지표명이 claim에 없음 | miss | miss | `bm25_expstruct` |
| 30-08a | DT_1B34E01 | 2. 표명과 claim 표현이 다름 | miss | miss | `dense_expanded` |
| 30-08b | DT_1B34E01 | 2. 표명과 claim 표현이 다름 | miss | miss | `dense_expanded`, `dense_expstruct` |
| 36-10 | DT_1NTA2002 | 2. 표명과 claim 표현이 다름 | miss | 184 | `dense_ctxD5` |

</details>


## 상한선

- 이번에 만든 **모든 검색기의 top-100 합집합**에 정답이 들어있는 claim: **66/70 = 94.3%**
- 융합 방식을 아무리 잘 골라도 이 값을 넘을 수 없다. 이 선을 올리려면 검색기 자체나 표 임베딩을 바꿔야 한다.


---

## [정정 2026-08-27] 지연(latency) 수치 무효 — 캐시 워밍 아티팩트

위 표들의 **dense 계열 검색기 지연은 신뢰할 수 없다.** run.py가 검색기를 순차 실행해서,
먼저 돈 검색기일수록 HNSW 인덱스(2.1GB)가 콜드였다. 실행 순서대로 지연이 단조 감소한다:

```
1번째  dense_full     378ms      6번째  dense_struct  82ms
2번째  dense_meas.    127ms      7번째  dense_ctxD2   54ms
3번째  dense_pop.     226ms     10번째  dense_ctxD4   15ms
```

이들은 전부 같은 테이블·같은 인덱스·같은 LIMIT에 벡터 내용만 다른 동일 쿼리다.
질의 벡터가 다르다고 HNSW 탐색 비용이 25배 달라지는 메커니즘은 없다.

워밍업 후 A/B 교차 실행으로 재측정한 결과(각 210회, `latency_ab.py`):

| | DB 검색 평균 | 중앙 | p95 | 인코딩 | 종단 합계 |
|---|--:|--:|--:|--:|--:|
| Baseline (full, 56자) | 51.0ms | 4.0ms | 412ms | 43ms | **94ms** |
| Context D2 (100자) | 36.6ms | 3.4ms | 318ms | 46ms | **83ms** |

- "Context D2가 Baseline보다 7배 빠르다"는 **틀렸다**. 실제로는 12% 차이이고 중앙값은 거의 같다.
- 문맥을 붙이면 텍스트가 길어져 **인코딩은 오히려 느리다**(43 → 46ms).
- dense 검색의 지연은 DB(중앙 4ms)가 아니라 **임베딩 인코딩(43~46ms)이 지배**한다.

여전히 유효한 것: 구조적으로 느린 검색기의 비교. `trgm_struct` 5,649ms(전수 스캔),
`item_measurement` 1,066ms(671k행 × ef=1000)는 캐시와 무관한 실제 비용이며,
**BM25 2ms vs trigram 5,649ms**도 유효하다.

Recall 수치는 이 정정의 영향을 받지 않는다.

---

## [추가 2026-08-27] 실험 E6 — item 필드 제외 검증

**가설**: E1→E2에서 `항목`을 넣자 R@100이 −1.4%p였고, 융합 ablation에서도 `item_measurement`
제거가 +1.4%p였다. 최종 포맷에서도 item이 해로우면 빼는 게 낫다.

**사전 확정 판정 기준**(실행 전에 못박음): E5 대비 +5%p 이상이면 실제 효과로 보고 전체
재임베딩(약 8시간)을 검토한다. ±5%p 이내면 70건 표본의 잡음으로 보고 현행 포맷을 유지한다.

| 구성 | 평균 길이 | R@10 | R@100 | MRR |
|---|--:|--:|--:|--:|
| E1 `table_name` | 23자 | 4.3% | 18.6% | 0.036 |
| E2 `+item` | 53자 | 2.9% | 17.1% | 0.035 |
| E3 `+axis` | 69자 | 4.3% | 26.4% | 0.037 |
| E4 `+axis_value` | 297자 | 15.0% | 46.4% | 0.074 |
| **E5 `+organization` (현행 운영 포맷)** | 311자 | **35.0%** | **62.1%** | **0.148** |
| E6 `E5 minus item` | 281자 | 26.4% | 55.0% | 0.130 |

**결과: E6 − E5 = R@100 −7.1%p, R@10 −8.6%p (70건 환산 −5.0건).**

가설은 반증됐다. item은 최종 포맷에서 해롭지 않고 **오히려 크게 기여한다**. E1→E2의 −1.4%p는
23자짜리 텍스트에서만 나타난 현상이었고, 311자 텍스트에서의 효과를 예측하지 못했다.
융합 ablation의 `item_measurement` −1.4%p 역시 **검색기 조합의 문제이지 문서 텍스트 구성의
문제가 아니었다** — 둘을 같은 증거로 묶은 것이 잘못이었다.

**결론: 현행 포맷(`item_axis_value_capped` = E5) 유지 확정. 재임베딩 불필요.**
