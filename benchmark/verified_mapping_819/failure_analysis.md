# failure_analysis.md — 819건 Verified Mapping 실험 실패/UNKNOWN 사례 정성 분석

검증 가능(verifiable=True)했지만 confidence != HIGH인 claim: 1971건 (전체 claim 2464건 중).

## confidence별 건수
- MEDIUM: 1178건
- LOW: 793건

## 사례 샘플 (최대 30건, confidence_reason별 대표)
- [row17-c1] confidence=MEDIUM reason='실제 KOSIS 수치까지 조회했으나 judge()가 판단불가(주제 불일치/모호 등)' sentence='주력 수출 품목인 반도체 수출액이 1419억달러로 역대 최대치를 고쳐 쓰면서 수출 증가세를 주도했다.'
- [row19-c0] confidence=LOW reason='최상위 후보가 RRF 기준으로 신뢰도 낮음(keyword 미발견+리랭커 비신뢰)' sentence='경북 칠곡에 있는 자동차 부품 가공 업체 화신정공은 2016년에 처음 산업용 로봇을 두 대 도입한 이후 현재 로봇 27대를 운용하는 곳이다.'
- [row19-c1] confidence=MEDIUM reason='후보+슬롯은 있으나 KOSIS 실측값 조회 실패/미지원(표 구조 불일치 등)' sentence='경남 창원에 있는 한 도금업체 대표는 "주 52시간 근로제로 인해 직원들에게 야근이나 추가 근무를 시킬 수 없게 되면서 로봇을 전체 공정에 투입'
- [row22-c2] confidence=LOW reason='구조적으로는 후보가 그럴듯하나 실제 KOSIS 수치와 불일치(mapping conflict형)' sentence='올해 취업자 수 증가폭은 작년(17만명·정부 전망치)보다 5만명 줄어든 것으로, 코로나로 취업자 수가 감소세로 돌아선 2021년 2월(-47만3'
- [row23-c0] confidence=MEDIUM reason='후보+슬롯은 있으나 KOSIS 실측값 조회 실패/미지원(표 구조 불일치 등) (예상외 예외 UndefinedTable: relation "kosis_table_detail_cache" does not exist\nLINE 1: select * from kosis_table_detail_cache where tbl_id = \'DT_2I...\n                      ^\n)' sentence='지난해 이 같은 일시 대출에 따라 정부가 부담한 이자는 2092억원에 달한 것으로 집계됐다.'
- [row240-c4] confidence=MEDIUM reason='슬롯 채우기 예외: HTTPError: 429 Client Error: Too Many Requests for url: https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-DASH-002' sentence='홍콩 항셍 지수는 -2.37%, 중국 상해종합 지수는 3.28%였다.'
- [row337-c1] confidence=MEDIUM reason='judge() 실패(JudgeError: 응답에서 JSON 객체를 찾지 못했습니다: \'입력하신 기사 주장과 통계 계산값은 서로 다른 지표를 다루고 있습니다.\\n따라서 topic_match는 false이며, verdict는 "판단불가"입니다.\\ngap_type은 null입니다.\\n이유는 다음과 같습니다: 이 통계표는 코스피 지수를 다루는데, 기사 주장은 정치 테마주를 다루고 있어 같은 지표가 아닙니다.\'), 실측값은 조회됨' sentence='정치 테마주가 단기 급등한 게 투자주의 종목 증가를 이끌었다.'
- [row1588-c1] confidence=MEDIUM reason='judge() 실패(JudgeError: 응답에서 JSON 객체를 찾지 못했습니다: "입력하신 기사 주장과 통계 계산값은 다음과 같습니다.\\n\\n- 기사 주장: 작년 초 물가 상승의 주범이던 농산물 가격도 지난달 1.5% 하락했는데, 사과(-5.7%)와 참외(-16.5%), 파(-20.8%) 등의 하락 폭이 컸다. (claim_type=규모, period=2024년 4월, unit=%, population=농산물 가격)\\n- 통계 계산값: 단순조회 116.38 2020=100 (2025년 4월 기준)\\n\\n이를 바탕으로 판단한 결과는 다음과 같습니다.\\n\\n- topic_match: true\\n- verdict : 판단불가\\n- gap_type : 기간\\n- reason : 기사는 작년 4월(2024년 4월)을 말하는데, 통계는 현재 달(2025년 4월)이어서 시점이 다르다. 또, 기사 주장에는 \'%\'변화율만 나와있고 원래 지수값이 없어서, 통계의 원본값(116.38)과 직접 비교할 수 없다."), 실측값은 조회됨' sentence='작년 초 물가 상승의 주범이던 농산물 가격도 지난달 1.5% 하락했는데, 사과(-5.7%)와 참외(-16.5%), 파(-20.8%) 등의 하락 '
