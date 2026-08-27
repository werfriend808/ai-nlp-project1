# 실패/성공 사례 분석


## baseline 실패 → item_axis 성공 — 5건

**[5-14a]** 지난달 쉬었음 인구는 243만4000명으로 1년 전 대비 6만명 늘었다.
- 정답표: DT_1DA7147S(Economically inactive pop.(Rested) by age group/ activity type)
- 조건어: []
- gold rank: Baseline=miss, Item=24, Item → Axis=24, Hybrid/RRF=52

**[5-14b]** 지난달 쉬었음 인구는 243만4000명으로 1년 전 대비 6만명 늘었다.
- 정답표: DT_1DA7147S(Economically inactive pop.(Rested) by age group/ activity type)
- 조건어: []
- gold rank: Baseline=miss, Item=24, Item → Axis=24, Hybrid/RRF=52

**[5-15a]** 특히 20대 쉬었음 인구는 39만6000명으로 전년 동월 대비 2000명 늘었다.
- 정답표: DT_1DA7147S(Economically inactive pop.(Rested) by age group/ activity type)
- 조건어: ['20대']
- gold rank: Baseline=miss, Item=24, Item → Axis=26, Hybrid/RRF=52

**[5-15b]** 특히 20대 쉬었음 인구는 39만6000명으로 전년 동월 대비 2000명 늘었다.
- 정답표: DT_1DA7147S(Economically inactive pop.(Rested) by age group/ activity type)
- 조건어: ['20대']
- gold rank: Baseline=miss, Item=24, Item → Axis=26, Hybrid/RRF=52

**[19-13]** 이런 경우는 코로나로 이 지표가 21개월 연속 100 미만이었던 2020년 2월~2021년 10월 이후 처음이다.
- 정답표: DT_1C8015(Composite Index of Business Indicators(2020=100))
- 조건어: ['동행종합지수 순환변동치']
- gold rank: Baseline=miss, Item=23, Item → Axis=25, Hybrid/RRF=61


## item 성공 → item_axis 실패 — 0건


## hybrid_rrf 에서만 성공 — 0건


## 모든 전략 실패 — 26건

**[5-03]** 작년 7월 이후 12개월 연속으로 감소세를 보이고 있다.
- 정답표: DT_1DA7E06S(산업별 취업자)
- 조건어: ['제조업 취업자']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[19-01a]** 5월 생산과 설비투자가 2개월 연속 동반 감소했다.
- 정답표: DT_1JH20202(Index of All Industry Production(Seasonally Adjusted Index)	)
- 조건어: ['산업생산', '설비투자']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[19-06]** 5월 소매 판매는 전달과 같은 수준을 유지했다.
- 정답표: DT_1K41012(Sales index by product group(2020=100))
- 조건어: ['소매판매']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[19-07]** 지난 3·4월 2개월 연속 줄었는데, 5월 1일 국회를 통과한 1차 추경 집행에도 반등에 실패했다.
- 정답표: DT_1K41012(Sales index by product group(2020=100))
- 조건어: ['소매판매']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[19-10a]** 건설 경기가 얼어붙으면서 건축·토목 분야 건설 실적을 뜻하는 '건설기성'도 전달보다 3.9% 줄어 3개월 연속 마이너스(-)를 기록했다.
- 정답표: DT_1JH20202(Index of All Industry Production(Seasonally Adjusted Index)	)
- 조건어: ['건설기성']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[19-10b]** 건설 경기가 얼어붙으면서 건축·토목 분야 건설 실적을 뜻하는 '건설기성'도 전달보다 3.9% 줄어 3개월 연속 마이너스(-)를 기록했다.
- 정답표: DT_1JH20202(Index of All Industry Production(Seasonally Adjusted Index)	)
- 조건어: ['건설기성']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[30-01]** 지난해 '인구 감소 지역' 4곳 중 3곳의 자살률이 전국 평균보다 높은 것으로 나타났다.
- 정답표: DT_1B34E13(Deaths, Death rates, Age-standardized death rates by cause(50 item) and sex: Si, Gun, and Gu )
- 조건어: ['인구 감소 지역']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[30-02]** 인구 감소 지역으로 지정된 전국 89개 시군구의 지난해 자살률은 인구 10만명당 36.3명이었다.
- 정답표: DT_1B34E13(Deaths, Death rates, Age-standardized death rates by cause(50 item) and sex: Si, Gun, and Gu )
- 조건어: ['인구 감소 지역']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[30-03]** 한국 전체 자살률 29.1명보다 25%가량 높은 수치다.
- 정답표: DT_1B34E13(Deaths, Death rates, Age-standardized death rates by cause(50 item) and sex: Si, Gun, and Gu )
- 조건어: ['인구 감소 지역']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss

**[30-04]** 인구 감소 지역 89곳 중 전국 평균보다 자살률이 높은 곳은 67곳(75.3%)에 달했다.
- 정답표: DT_1B34E13(Deaths, Death rates, Age-standardized death rates by cause(50 item) and sex: Si, Gun, and Gu )
- 조건어: ['인구 감소 지역']
- gold rank: Baseline=miss, Item=miss, Item → Axis=miss, Hybrid/RRF=miss


## 근접 중복 후보 분석 (Baseline top-10)

정답표 이름의 핵심어를 공유하는 상위 후보가 몇 개인지 — 중복 경쟁의 크기.

| claim | gold rank | top10 중 유사표 | 예시 |
|---|--:|--:|---|
| 5-01a | 4 | 2/10 | DT_1DA7A64S Employed persons by size of em |
| 5-01b | 4 | 2/10 | DT_1DA7A64S Employed persons by size of em |
| 5-02a | 2 | 3/10 | DT_1DA7E06S_NEW 산업별 취업자 |
| 5-02b | 2 | 3/10 | DT_1DA7E06S_NEW 산업별 취업자 |
| 5-03 | miss | 0/10 |  |
| 5-04 | 36 | 4/10 | DT_1FB0011 한국에서의 동일직업 근무기간별 취업자 |
| 5-05a | 11 | 3/10 | DT_21603_C001004 산업별 취업자 |
| 5-05b | 11 | 3/10 | DT_21603_C001004 산업별 취업자 |
| 5-06a | 3 | 2/10 | DT_11732S0125 Employment Population and Empl |
| 5-06b | 3 | 2/10 | DT_11732S0125 Employment Population and Empl |
| 5-06c | 3 | 2/10 | DT_11732S0125 Employment Population and Empl |
| 5-07a | 42 | 1/10 | DT_1BD0002 Number of workers by gender &  |
| 5-07b | 42 | 1/10 | DT_1BD0002 Number of workers by gender &  |
| 5-11 | 24 | 1/10 | DT_20503_C001004 산업별 취업자 |
| 5-12a | 4 | 3/10 | DT_1DE9046S Economically active pop. by ag |

**Baseline 평균 근접중복 비율: 16.7%** (top-10 중 정답표와 표명 핵심어를 공유하는 다른 표)


### 전략별 근접중복 비율 (top-10)

| Strategy | 평균 중복 비율 |
|---|--:|
| Baseline | 16.7% |
| Item | 18.0% |
| Item → Axis | 16.0% |
| Hybrid/RRF | 21.1% |
