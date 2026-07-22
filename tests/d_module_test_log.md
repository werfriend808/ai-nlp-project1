# D 담당(4단계) 모듈 단위 테스트 기록

- 대상: `agent/orchestrator/slot_filler.py` (`fill_slots`)
- 결과: **10/12 PASS** (아래 "발견된 이슈" 참고)
- 참고: `agent/orchestrator/clarify.py`는 아직 `clarify()` 함수가 구현되어 있지 않음(주석 한 줄만 있는 상태). 그래서 `clarify()`까지 같이 부르는 `agent/orchestrator/test_integration_manual.py`는 현재 `ImportError`로 아예 실행이 안 됨. 이번 테스트는 `slot_filler.py`만 떼어서 검증.

## 실행 방법

프로젝트 루트에서 실행 (실제 HCX-DASH-002 API를 호출하므로 `.env`에 `HCI__API_KEY` 필요):

```
python -m tests.test_slot_filler_module
```

## 케이스 목록

| # | 이름 | 내용 | 결과 |
|---|---|---|---|
| 1 | case_01_region_only | 지역만 있는 발화 ("서울 통계 알려줘") | ✅ |
| 2 | case_02_period_only_absolute_year | 절대 연도만 있는 발화 ("2024년 통계 알려줘") | ✅ |
| 3 | case_03_calc_type_only | 계산종류만 있는 발화 ("평균 알려줘") | ❌ (아래 이슈 1 참고) |
| 4 | case_04_nothing_in_utterance | 아무 슬롯 정보 없는 발화 | ✅ |
| 5 | case_05_all_slots_at_once | 세 슬롯 모두 있는 발화 ("작년 서울 증감률 알려줘") | ✅ |
| 6 | case_06_relative_last_year | "작년"만 있는 발화 ("작년 통계 줘") | ❌ (아래 이슈 2 참고) |
| 7 | case_07_relative_this_year | "올해"만 있는 발화 | ✅ |
| 8 | case_08_relative_two_years_ago | "재작년"만 있는 발화 | ✅ |
| 9 | case_09_ambiguous_relative_time_delegated_to_llm | "지난달"(월 단위, HCX-003 위임) + 지역 | ✅ |
| 10 | case_10_existing_slots_preserved_on_partial_update | 기존 슬롯 있는 상태에서 지역만 보완 (오염 방지 회귀) | ✅ |
| 11 | case_11_unrelated_utterance_does_not_pollute_existing | 통계와 무관한 발화가 기존 슬롯을 안 건드리는지 | ✅ |
| 12 | case_12_empty_string_utterance_should_not_crash | 빈 문자열 입력 시 크래시 없는지 | ✅ |

## 발견된 이슈

두 이슈 모두 **코드 버그 아님** — `fill_slots`/`normalize_time_expressions`의 로직은 정상이고, HCX-DASH-002의 슬롯 추출(LLM) 단계에서만 발생. 여러 번 재실행해도 동일하게 재현됨(랜덤성 아님).

### 1. calc_type / region 단독 언급 시 추출 실패
- **증상**: "평균 알려줘", "증감률 알려줘", "합계 알려줘", "순위 알려줘", "서울 알려줘" → 전부 `{"period": null, "region": null, "calc_type": null}`
- **대조**: "서울 **통계** 알려줘"(case_01)는 성공 → "통계"라는 단어가 같이 있어야 트리거되는 것으로 보임
- **원인 추정**: `build_extraction_prompt`의 few-shot 예시가 전부 "슬롯 2개 이상 동시 등장" 형태라, 슬롯 1개뿐인 짧은 발화에서 모델이 추출을 포기
- **제안**: 슬롯 1개짜리 예시(예: `"평균 알려줘"` → `{"calc_type": "평균", ...}`)를 few-shot에 추가 후 재검증. 프롬프트 수정 여부는 팀 상의 필요

### 2. "작년"만 단독으로 오면 인식 실패 ("올해"/"재작년"은 정상)
- **증상**: "작년 통계 줘", "작년 통계 알려줘", "작년 것 알려줘", "작년꺼 알려줘" → 전부 `period: null`
- **대조**: 동일 문장 구조의 "올해 통계 줘"(`period: "올해"`), "재작년 통계 줘"(`period: "재작년"`)는 정상 추출
- **비고**: `RELATIVE_YEAR_OFFSET`에 세 단어가 동등하게 정의돼 있고, 프롬프트 예시에도 "작년 증감률 알려줘"가 있는데도 실패 — 원인 불명
- **제안**: 이슈 1과 함께 few-shot 보강 후 재테스트

## 커버 안 된 것 (다음에 필요하면 추가)
- `clarify.py`의 `clarify()` 함수 자체가 미구현 상태라 되묻기(clarify) 로직 및 `test_integration_manual.py` 통합 테스트는 이번 범위 밖. 구현 후 별도 테스트 필요.
- 여러 지역이 동시에 언급되는 경우("서울 대비 전국" 같은 비교 발화)의 region 추출 우선순위는 다루지 않음.
- HCX API 타임아웃/5xx 등 인프라성 실패 케이스는 다루지 않음.
