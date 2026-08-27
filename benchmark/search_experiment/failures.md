# 실패/성공 사례 분석


## baseline 실패 → item_axis 성공 — 8건

**[5-07a]** 한편 40대도 5만 5000명, 50대도 5만3000명 줄었다.
- 정답표: DT_1DA7024S(성/연령별 취업자)
- 조건어: ['40대']
- gold rank: Baseline=miss, Item=68, Item → Axis=68, Hybrid/RRF=miss

**[5-07b]** 한편 40대도 5만 5000명, 50대도 5만3000명 줄었다.
- 정답표: DT_1DA7024S(성/연령별 취업자)
- 조건어: ['50대']
- gold rank: Baseline=miss, Item=68, Item → Axis=68, Hybrid/RRF=miss

**[5-14a]** 지난달 쉬었음 인구는 243만4000명으로 1년 전 대비 6만명 늘었다.
- 정답표: DT_1DA7147S(연령/활동상태별(쉬었음) 비경제활동인구)
- 조건어: []
- gold rank: Baseline=miss, Item=24, Item → Axis=24, Hybrid/RRF=52

**[5-14b]** 지난달 쉬었음 인구는 243만4000명으로 1년 전 대비 6만명 늘었다.
- 정답표: DT_1DA7147S(연령/활동상태별(쉬었음) 비경제활동인구)
- 조건어: []
- gold rank: Baseline=miss, Item=24, Item → Axis=24, Hybrid/RRF=52

**[5-15a]** 특히 20대 쉬었음 인구는 39만6000명으로 전년 동월 대비 2000명 늘었다.
- 정답표: DT_1DA7147S(연령/활동상태별(쉬었음) 비경제활동인구)
- 조건어: ['20대']
- gold rank: Baseline=miss, Item=24, Item → Axis=26, Hybrid/RRF=57

**[5-15b]** 특히 20대 쉬었음 인구는 39만6000명으로 전년 동월 대비 2000명 늘었다.
- 정답표: DT_1DA7147S(연령/활동상태별(쉬었음) 비경제활동인구)
- 조건어: ['20대']
- gold rank: Baseline=miss, Item=24, Item → Axis=26, Hybrid/RRF=57

**[19-12]** 이 지표는 작년 5월(99.7)부터 13개월 연속으로 100을 밑돌았다.
- 정답표: DT_1C8015(경기종합지수(2020=100)(10차))
- 조건어: ['동행종합지수 순환변동치']
- gold rank: Baseline=miss, Item=23, Item → Axis=25, Hybrid/RRF=61

**[19-13]** 이런 경우는 코로나로 이 지표가 21개월 연속 100 미만이었던 2020년 2월~2021년 10월 이후 처음이다.
- 정답표: DT_1C8015(경기종합지수(2020=100)(10차))
- 조건어: ['동행종합지수 순환변동치']
- gold rank: Baseline=miss, Item=23, Item → Axis=25, Hybrid/RRF=62


## item 성공 → item_axis 실패 — 0건


## hybrid_rrf 에서만 성공 — 0건


## 모든 전략 실패 — 21건

**[5-03]** 작년 7월 이후 12개월 연속으로 감소세를 보이고 있다.
- 정답표: DT_1DA7E06S(산업별 취업자)
- 조건어: ['제조업 취업자']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[19-01a]** 5월 생산과 설비투자가 2개월 연속 동반 감소했다.
- 정답표: DT_1JH20202(전산업생산지수(계절조정지수))
- 조건어: ['산업생산', '설비투자']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[19-07]** 지난 3·4월 2개월 연속 줄었는데, 5월 1일 국회를 통과한 1차 추경 집행에도 반등에 실패했다.
- 정답표: DT_1K41012(재별 및 상품군별 소매판매액지수(2020=100.0))
- 조건어: ['소매판매']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[19-10a]** 건설 경기가 얼어붙으면서 건축·토목 분야 건설 실적을 뜻하는 '건설기성'도 전달보다 3.9% 줄어 3개월 연속 마이너스(-)를 기록했다.
- 정답표: DT_1JH20202(전산업생산지수(계절조정지수))
- 조건어: ['건설기성']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[19-10b]** 건설 경기가 얼어붙으면서 건축·토목 분야 건설 실적을 뜻하는 '건설기성'도 전달보다 3.9% 줄어 3개월 연속 마이너스(-)를 기록했다.
- 정답표: DT_1JH20202(전산업생산지수(계절조정지수))
- 조건어: ['건설기성']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[30-01]** 지난해 '인구 감소 지역' 4곳 중 3곳의 자살률이 전국 평균보다 높은 것으로 나타났다.
- 정답표: DT_1B34E13(시군구/사망원인(50항목)/성/ 사망자수, 사망률, 연령표준화 사망률(1998~))
- 조건어: ['인구 감소 지역']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[30-02]** 인구 감소 지역으로 지정된 전국 89개 시군구의 지난해 자살률은 인구 10만명당 36.3명이었다.
- 정답표: DT_1B34E13(시군구/사망원인(50항목)/성/ 사망자수, 사망률, 연령표준화 사망률(1998~))
- 조건어: ['인구 감소 지역']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[30-03]** 한국 전체 자살률 29.1명보다 25%가량 높은 수치다.
- 정답표: DT_1B34E13(시군구/사망원인(50항목)/성/ 사망자수, 사망률, 연령표준화 사망률(1998~))
- 조건어: ['인구 감소 지역']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[30-04]** 인구 감소 지역 89곳 중 전국 평균보다 자살률이 높은 곳은 67곳(75.3%)에 달했다.
- 정답표: DT_1B34E13(시군구/사망원인(50항목)/성/ 사망자수, 사망률, 연령표준화 사망률(1998~))
- 조건어: ['인구 감소 지역']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[30-05]** 특히 지난해 자살률 상위 10개 지역은 모두 인구 감소 지역이었다.
- 정답표: DT_1B34E13(시군구/사망원인(50항목)/성/ 사망자수, 사망률, 연령표준화 사망률(1998~))
- 조건어: ['자살률 상위 지역']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss


## 근접 중복 후보 분석 (Baseline top-10)

정답표 이름의 핵심어를 공유하는 상위 후보가 몇 개인지 — 중복 경쟁의 크기.

| claim | gold rank | top10 중 유사표 | 예시 |
|---|--:|--:|---|
| 5-01a | 2 | 8/10 | DT_1DA7A64 종사자규모별 취업자(시계열 보정 前 자료) |
| 5-01b | 2 | 8/10 | DT_1DA7A64 종사자규모별 취업자(시계열 보정 前 자료) |
| 5-02a | 1 | 5/10 | DT_1DA7E06S_NEW 산업별 취업자 |
| 5-02b | 1 | 5/10 | DT_1DA7E06S_NEW 산업별 취업자 |
| 5-03 | miss | 0/10 |  |
| 5-04 | 32 | 4/10 | DT_1FB0011 한국에서의 동일직업 근무기간별 취업자 |
| 5-05a | 9 | 5/10 | DT_1DA9003S 산업별 계절조정 취업자 |
| 5-05b | 9 | 5/10 | DT_1DA9003S 산업별 계절조정 취업자 |
| 5-06a | 2 | 7/10 | DT_201004_O030004 연령별 취업자 |
| 5-06b | 2 | 7/10 | DT_201004_O030004 연령별 취업자 |
| 5-06c | 2 | 7/10 | DT_201004_O030004 연령별 취업자 |
| 5-07a | miss | 0/10 |  |
| 5-07b | miss | 0/10 |  |
| 5-11 | 24 | 3/10 | DT_1DA9003S 산업별 계절조정 취업자 |
| 5-12a | 3 | 5/10 | DT_1DE9046 연령별 경제활동상태 |

**Baseline 평균 근접중복 비율: 18.7%** (top-10 중 정답표와 표명 핵심어를 공유하는 다른 표)


### 전략별 근접중복 비율 (top-10)

| Strategy | 평균 중복 비율 |
|---|--:|
| Baseline | 18.7% |
| Item | 24.6% |
| Item → Axis | 28.3% |
| Hybrid/RRF | 25.3% |
