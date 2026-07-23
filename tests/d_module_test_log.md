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

---

# D 담당(7단계) 모듈 단위 테스트 기록

- 대상: `agent/verdict/judge.py` (`judge`, `judge_complex`, `needs_hybrid_reasoning`, `judge_all`)
- 결과: **4/4 스모크 케이스 PASS** — 규칙 기반 1차 필터(일치/불일치 즉시 확정) 2건, HCX-003 애매 경계 판정 1건, HCX-007 복합 케이스 승격 조건 확인 1건. 전부 실제 API 호출로 검증.

## 실행 방법

프로젝트 루트에서 실행 (`.env`에 `HCX_API_KEY` 필요 — HCX-003/HCX-007 실제 호출 포함):

```
python -m agent.verdict.judge
```

## 설계 메모 (interfaces.py에 없는 부분이라 이 모듈에서 임의로 정한 것)

- `interfaces.py`의 `Claim`은 `sentence`(원문 텍스트)만 있고 파싱된 숫자 필드가 없음. 규칙 기반
  1차 필터를 돌리려면 숫자가 필요해서, `_extract_claim_number()`가 정규식으로 문장에서 숫자를
  뽑아냄 (단위가 있으면 그 단위 앞 숫자 우선, 없으면 문장 마지막 숫자 사용, "2024년" 같은 연도
  표기는 제외). 100% 정확할 수 없는 heuristic이라 실패하면 규칙 필터를 건너뛰고 바로 LLM에
  원문 문장을 그대로 넘김 — 애매한 걸 코드가 잘못 확정 판정하는 것보다 보수적인 쪽을 택함.
- claim_type="증감률" 문장은 숫자에 부호가 없고 "감소/증가" 같은 단어로만 방향이 표현됨
  (`calculator.py`의 `compute_change_rate`는 감소를 음수로 냄). `_apply_direction()`에서
  감소 단어("감소", "하락", "줄어" 등) 발견 시 부호를 붙여서 계산값과 같은 척도로 비교.
- 오차 허용 기준: %류 주장은 %p(절대 차이), 그 외 단위는 상대오차(%)로 통일 (`NUMERIC_TOLERANCE
  = 0.1`, `CLEAR_GAP_MULTIPLIER = 5` → 0.5%p/0.5% 넘게 벌어지면 규칙만으로 "불일치" 확정).
- 기간(period) 단위(연/월) 불일치 감지: `_period_granularity()`가 "월" 포함 여부와 자릿수(6자리=월,
  4자리=년)로 best-effort 추정. 이는 Day2 파이프라인 연결 테스트에서 실제로 발견된 문제(기사는
  "전년 동월 대비"인데 KOSIS는 연간 평균 기준으로 비교되던 것)를 규칙 단계에서 미리 걸러
  LLM에 위임하기 위해 넣음.
- 복합 케이스 승격 조건(`needs_hybrid_reasoning`): Claim이 2개 이상이고 그중 `claim_type="비교"`가
  있거나, ComputedResult가 2개 이상 필요한 경우 HCX-007로 승격. 단일 Claim-ComputedResult 쌍으로
  안 끝나는 경우를 다루기 위함.

## 케이스 목록

| # | 이름 | 내용 | 경로 | 결과 |
|---|---|---|---|---|
| 1 | 규칙 기반 일치 | "1.1% 감소" 주장 vs 계산값 -1.15% (오차 0.05%p, 방향 부호 처리 확인) | 규칙(LLM 미호출) | ✅ |
| 2 | 규칙 기반 불일치 | "10%까지 치솟았다" 주장 vs 계산값 7.2% (오차 2.8%p, 기획서 예시 그대로) | 규칙(LLM 미호출) | ✅ |
| 3 | 애매 경계 → HCX-003 | "전년 동월 대비 2.2%" 주장 vs 계산값 2.3%(연간 평균 기준, 기간 단위 월≠년) | HCX-003 | ✅ |
| 4 | 복합 케이스 승격 조건 | Claim 2개(하나는 claim_type="비교") + ComputedResult 2개 | `needs_hybrid_reasoning` | ✅ |

`judge_complex()`(HCX-007)도 케이스 4의 입력으로 별도 실행해서 확인함 (아래 이슈 1 참고 후 정상 동작).

## 발견된 이슈

### 1. (수정 완료) HCX-007이 `maxTokens` 파라미터 자체를 거부함
- **증상**: `agent/preprocessing/hcx_client.py`의 `call_hcx()`가 항상 body에 `maxTokens`를 넣는데,
  HCX-007로 호출하면 값과 무관하게(100/1024/2048/4096/8192/32768 전부 동일) `40001 Invalid
  parameter: maxTokens`로 거부됨. `maxTokens` 필드 자체를 빼면 200 OK (추론 단계 때문에 모델이
  길이를 자체 결정하는 것으로 보임).
- **영향 범위**: HCX-007을 쓰는 모든 곳(7단계 `judge_complex`뿐 아니라 8단계 설명 생성도 HCX-007을
  씀 — `interfaces.py` 8단계 담당 모델 참고). A/D 공통 파일이라 `hcx_client.py`에서 고침.
- **수정**: `NO_MAX_TOKENS_MODELS = {"HCX-007"}`을 두고, 해당 모델이면 `call_hcx()`가 body에
  `maxTokens`를 아예 넣지 않도록 수정. 다른 모델(HCX-DASH-002, HCX-003 등) 동작은 그대로.

## 커버 안 된 것 (다음에 필요하면 추가)
- `_extract_claim_number()`의 정규식 기반 숫자 추출은 배수 표현("두 배", "절반")처럼 숫자가 아닌
  경우 아예 추출 실패로 처리되고 전부 LLM에 위임됨 — 이런 표현이 실제로 얼마나 자주 나오는지,
  그때 LLM 판정 품질이 괜찮은지는 실기사 샘플로 추가 검증 필요.
- 모집단(population) 불일치는 규칙 단계에서 구조적으로 검증 못 함 — `ComputedResult`에
  population 관련 필드가 없어서(interfaces.py 참고), LLM이 `claim.population` 텍스트만 보고
  판단함. 필요하면 팀 상의 후 `ComputedResult`에 필드 추가 검토.
- `judge_all()`은 아직 실제 오케스트레이터(8단계 연결)에서 호출해본 적 없음 — Day2
  `batch_runner.py`처럼 1~7단계 전체를 잇는 연결 테스트는 이번 범위 밖.
