# calc_type 라우팅 작업 테스트 기록 (2026-08-05)

- 배경: claim_type(2단계)과 calc_type(6단계) 사이에 매핑 로직이 없던 갭을 메우는 작업.
  자세한 배경/근거는 개인 폴더의 `CALC_TYPE_ROUTING_DESIGN.md` 참고.
- 대상: `agent/kosis/calculator.py`(compute_min_check), `agent/preprocessing/claim_extractor.py`
  (null→"None" 버그 수정, comparison_operator 정규화, 스키마 밖 claim 필터링),
  `agent/orchestrator/calc_type_router.py`(신규), `agent/pipeline/batch_runner.py`("전망" 조기 차단)
- 결과: **아래 5개 테스트 파일, 총 36/36 PASS** (전부 API 호출 없는 순수 로직 테스트,
  batch_runner 테스트만 1·2단계와 DB 저장을 mock으로 대체). 검증 완료 후 이 5개 스크립트
  자체는 저장소에서 삭제하고, 무엇을 확인했는지만 이 기록으로 남긴다.

## 1. compute_max_check / compute_min_check (calculator.py)

`tests/test_calculator_extremes.py` — **7/7 PASS**. 두 메서드 다 이전까지 테스트가 하나도
없었음.

| # | 확인 내용 | 결과 |
|---|---|---|
| 1 | current가 최댓값이 아니어도 historical 전체에서 진짜 최댓값을 찾는지 | ✅ |
| 2 | historical에 current가 빠져 있어도 자동으로 합쳐서 비교하는지 | ✅ |
| 3 | 단위(unit)가 다른 값들을 넘기면 CalculationError가 나는지 (최댓값) | ✅ |
| 4 | 진짜 최솟값을 찾는지 (compute_min_check) | ✅ |
| 5 | historical에 current가 빠져 있어도 최솟값 비교에 포함되는지 | ✅ |
| 6 | 단위가 다른 값들을 넘기면 CalculationError가 나는지 (최솟값) | ✅ |
| 7 | historical 없이 current 하나만 있어도 그 값 자체가 최솟값으로 나오는지 | ✅ |

## 2. claim_extractor.py 파싱 (_item_to_claim)

`tests/test_claim_extractor_parsing.py` — **9/9 PASS**. P0(null→"None" 버그)와 P3
(comparison_operator "하락"→"감소") 수정 검증.

| # | 확인 내용 | 결과 |
|---|---|---|
| 1 | HCX가 claim_type: null을 주면 문자열 "None"이 아니라 진짜 None이 되는지 (버그 재현 케이스) | ✅ |
| 2 | claim_type 키 자체가 없어도 KeyError 없이 None으로 처리되는지 | ✅ |
| 3 | 정상 4종(규모/증감률/비교/전망)은 그대로 통과하는지 | ✅ |
| 4 | "배경"/"규정"/"요약"/빈 문자열처럼 스키마 밖 값이 None으로 정규화되는지 | ✅ |
| 5 | `_normalize_claim_type()` 직접 호출 (None/정상값/스키마밖/숫자 타입) | ✅ |
| 6 | comparison_operator "하락"이 "감소"로 정규화되는지 | ✅ |
| 7 | 정상 5종(증가/감소/동일/초과/미만)은 그대로 통과하는지 | ✅ |
| 8 | "혼합"/"완화"/"약화"/"회복"/"2년 연속"은 (의도적으로) 정규화 없이 원문 그대로인지 | ✅ |
| 9 | comparison_operator: null은 None으로 유지되는지 | ✅ |

**부수 발견**: `agent/verdict/judge.py`의 `_apply_direction()`은 `comparison_operator=="감소"`
일 때만 명시적으로 부호를 뒤집고, 그 외 non-None 값은 전부 "부호 없음"으로 취급한다. 즉
"하락"이 정규화 없이 그대로 들어왔다면 실제로는 감소인데 부호가 안 뒤집히는 버그가 있었을
것으로 추정됨 — 이번 정규화로 같이 해소됨.

## 3. claim_extractor.py 스키마 밖 필터링 (_parse_claims / _salvage_claims)

`tests/test_claim_extractor_schema_filter.py` — **4/4 PASS**. P2("배경"/"규정" 필터링) 검증.

| # | 확인 내용 | 결과 |
|---|---|---|
| 1 | `_parse_claims`가 null claim_type 항목을 결과 리스트에서 제외하는지 | ✅ |
| 2 | "배경"/"규정" 같은 스키마 밖 문자열 항목을 제외하는지 | ✅ |
| 3 | 전부 스키마 밖이면 빈 리스트를 반환하는지 | ✅ |
| 4 | 배열이 중간에 끊긴 salvage 경로에서도 동일하게 필터링되는지 | ✅ |

## 4. calc_type_router.py (신규 라우터)

`tests/test_calc_type_router.py` — **14/14 PASS**. `route_calc_type()`과
`detect_extreme_value_claim()` 둘 다 신규 모듈이라 테스트가 없었음. 극값 판별 케이스 중
다수는 뉴스기사 100건 실측 조사(개인 폴더 원시 데이터)에서 실제로 나온 문장을 그대로
재현 케이스로 사용.

| # | 확인 내용 | 결과 |
|---|---|---|
| 1 | claim_type="증감률" → "증감률" | ✅ |
| 2 | claim_type="비교" → "증감" | ✅ |
| 3 | claim_type="전망" → None(판단불가 신호) | ✅ |
| 4 | claim_type=None/스키마밖("규정") → None | ✅ |
| 5 | claim_type="규모" + 극값 패턴 없음 → "단순조회" | ✅ |
| 6 | "규모" + "역대 최대" → "최댓값검증" | ✅ |
| 7 | "규모" + "12년 만에 최저치"(실측 문장) → "최솟값검증" | ✅ |
| 8 | "규모" + "9년 만에 최고치"(실측 문장) → "최댓값검증" | ✅ |
| 9 | "규모" + "코로나 이후 최고" → "최댓값검증" | ✅ |
| 10 | claim_type="증감률"은 극값 패턴이 있어도 항상 "증감률"로 라우팅되는지(규칙표 확인) | ✅ |
| 11 | 극값 패턴 없는 문장은 is_extreme=False | ✅ |
| 12 | "역대 최고"류 → direction="max" | ✅ |
| 13 | "역대 최저"류 → direction="min" | ✅ |
| 14 | "역대"만 있고 방향 단어가 없는 드문 경우 → 기본값 "max" | ✅ |

## 5. batch_runner.py "전망" 조기 차단

`tests/test_batch_runner_forecast_shortcut.py` — **2/2 PASS**. 1·2단계(classify/
extract_claims)와 `insert_verification`(DB 저장)을 mock으로 대체해서 실제 HCX/KOSIS
API·DB 호출 없이 확인.

| # | 확인 내용 | 결과 |
|---|---|---|
| 1 | claim_type="전망" claim은 `search_and_rerank`(3단계)가 아예 호출되지 않고, verdict=판단불가로 즉시 1건 기록되는지 | ✅ |
| 2 | 회귀 확인: claim_type="규모" claim은 여전히 3단계를 정상적으로 타는지 | ✅ |

## 커버 안 된 것 (다음에 필요하면 추가)

- `calc_type_router.route_calc_type()`은 아직 `batch_runner.py`/`pipeline_1_4.py` 어디에도
  실제로 배선되지 않은 독립 모듈. 배선 시 통합 테스트 별도 필요.
- 극값 시계열을 실제로 fetch해서 `compute_max_check`/`compute_min_check`에 넘기는
  end-to-end 경로(감지 → `agent_chat.py`의 `resolve_max_*_responses` → 실제 계산)는
  이번 범위에서 다루지 않음 — 개별 조각(감지 함수, calculator 메서드)만 단위 검증함.
