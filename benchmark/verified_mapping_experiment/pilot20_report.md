# Verified Mapping + Retrieval Cascade — 20건 Pilot 결과 보고

실행일: 2026-08-30 / 실행 스크립트: `pilot20_run.py` / 산출물: `pilot20_articles.json`,
`pilot20_claims.jsonl`(77건), `pilot20_candidates.jsonl`, `pilot20_verified_mappings.jsonl`(4건),
`pilot20_failure_cases.jsonl`(68건), `pilot20_run_stats.json`

**최종 판정: NEEDS_FIX** (근거는 10번 참고 — 핵심 검증 엔진은 정상 동작하나, mapping override
로직에 재현 가능한 구체적 결함 1건을 발견함. 좁은 범위의 수정 후 재pilot 권장.)

---

## 1. 20건 선정 결과

`select_pilot20_articles()`(정규식 기반 6버킷, 무작위 없음)로 선정. Discovery 10 /
Evaluation 10 정확히 분리됨.

| 버킷 | 목표 | 실제 |
|---|---|---|
| qualitative_no_number | 4 | 4 |
| comparative | 4 | 4 |
| numeric | 3 | 3 |
| threshold | 3 | 3 |
| superlative | 3 | 3 |
| trend | 3 | 3 |

Discovery: row1, row3, row4, row5, row12, row25, row45, row50, row62, row76
Evaluation: row0, row2, row6, row10, row15, row17, row21, row27, row60, row65

관찰: row4(discovery)·row15(evaluation)는 같은 사건("2024년 수출 6838억달러 역대 최대")을
다룬 근접중복 기사 — methodology.md 4번에 미리 명시해둔 대로, 이 쌍이 아래 6번 Mapping
Reuse Test 사례 대부분(6건 중 6건)의 출처다. **즉 이번 20건의 A/B 비교 신호는 사실상 "우연히
같은 사건을 다룬 기사 쌍 1개"에서만 나왔다** — 결과 해석 시 반드시 감안해야 하는 가장 중요한
제약이다(10번에서 재논의).

## 2. Claim 추출 결과

- 20건 중 18건이 관련 기사로 분류됨(2건 무관 판정).
- 총 77건 claim 추출(HCX 최적화 프롬프트 + workers=2 + 조건부 recovery 1라운드).
- extended_claim_type 실측 분포(사전 선정 버킷이 아니라 추출 후 실제 재분류):
  numeric=42, comparative=27, superlative=6, trend=2, qualitative(전망/해외)=0.
- **숫자없는 통계주장(claim.value=None) 특별 점검(grounding 2번 요청)**: 77건 중 **3건(3.9%)**
  만 명시적 수치값이 없었다(row3-c0, row4-c2, row17-c1) — `qualitative_no_number` 버킷으로
  기사 4건을 일부러 골랐음에도 실제 추출된 claim 중 "정말 숫자가 없는" 건은 소수였다. 이는
  `prompt_optimization_experiment`가 이미 보고한 "숫자없는 통계claim 추출이 약함(72.5%)"과
  일관된 관찰이다 — 기사에 통계적 사실(예: "역대 최대")과 숫자가 같은 문단에 같이 나오면
  추출기가 숫자 쪽을 claim의 핵심으로 붙이는 경향이 있다.
  - 다만 **성공 사례 1건은 확보됨**: `row4-c2`("...역대 최대 수출 실적과 무역수지 흑자를
    동시에 기록했다", 숫자 없음)가 **HIGH**로 검증됐다 — 2023/2024 KOSIS 실측값을 각각
    조회해 실제로 2024년이 최대인지 비교 검증에 성공한 사례(judge_reason: "기사 주장과 통계
    계산값 모두 역대 최대 수출입 실적을 말하고 있어 일치한다"). 숫자 없는 주장도 **검증
    자체는 가능하다**는 걸 최소 1건으로 확인함(일반화하기엔 표본이 너무 작음).

## 3. Confidence 분포 (전체 77건)

| Confidence | 건수 | 비율 |
|---|---|---|
| HIGH | 9 | 11.7% |
| MEDIUM | 28 | 36.4% |
| LOW | 40 | 51.9% |
| UNKNOWN | 0 | 0% |

Discovery(34건): HIGH 4 / MEDIUM 12 / LOW 18. Evaluation baseline(43건): HIGH 5 / MEDIUM 16 /
LOW 22. **HIGH가 관대하게 과다생성되지 않음** — LOW가 과반(52%)으로, Verification Engine이
"애매하면 HIGH 안 준다" 원칙을 실제로 지키고 있다는 신호(judge()가 "판단불가"를 반환한 경우
전부 MEDIUM으로 정확히 떨어짐, `mapping_conflict`형은 LOW로 정확히 분류됨).

## 4. Discovery: HIGH mapping 4건, 실제검증가능비율

| claim_id | table_id | table_name |
|---|---|---|
| row4-c1 | DT_134001_001 | 수출입총괄 |
| row4-c2 | DT_1R11006_FRM101 | 국가별 수출액, 수입액 |
| row4-c5 | DT_1R11001_FRM101 | 품목별 수출액, 수입액 |
| row4-c11 | DT_1R11001_FRM101 | 품목별 수출액, 수입액 |

**실제검증가능비율 = 4/4 = 100%(구조상 당연)** — `classify_confidence()` 설계상 HIGH는
`judge()`가 실제 KOSIS 조회값과 대조해 "일치"를 반환한 경우만 나온다(6번 methodology.md
참고) — HIGH이면서 검증 안 된 경우는 애초에 존재할 수 없다. 다만 **4건 전부 discovery 10건
중 단 1개 기사(row4)에서만 나왔다** — 나머지 discovery 9건은 HIGH mapping을 하나도 못
만들었다. 10개 기사 중 1개만 mapping을 생산했다는 건 "Discovery 단계 자체의 mapping 생산율이
낮다"는 뜻이고, 이번 20건 규모에서는 재사용 가능한 mapping 풀이 사실상 1개 사건에 묶여 있다는
뜻이다(위 1번 관찰과 동일한 제약).

## 5. Mapping Reuse Test — Evaluation(43건) A(baseline) vs B(mapping-assisted)

- gold 확보(scored): **6건**(43건 중 13.9%) — 나머지 37건(86.1%)은 baseline도 mapping도
  독립 재검증에서 HIGH를 못 받아 gold 자체가 없음(no_gold, R@k 분모에서 제외).
- mapping이 실제로 "적용 시도"된 건: **6건**(no_mapping_applicable=37, 나머지 전부 row4
  mapping이 baseline top-100 안에 있었던 row15의 6개 claim). 즉 mapping 후보가 있었던 경우는
  전부 위 1번에서 언급한 그 근접중복 기사 쌍뿐이다.

| Recall@k | Baseline(A) | Mapping-assisted(B) |
|---|---|---|
| R@1 | 0.8333 (5/6) | 0.8333 (5/6) |
| R@10 | 1.0 | 1.0 |
| R@50 | 1.0 | 1.0 |
| R@100 | 1.0 | 1.0 |

**집계 숫자만 보면 baseline과 assisted가 완전히 동일하다 — 그러나 이건 "차이가 없다"는
뜻이 아니라 개선 1건과 악화 1건이 우연히 상쇄된 것이다.** 6건의 outcome 분해:

| ab_outcome | 건수 | claim_id |
|---|---|---|
| mapping_rescued_claim_improvement | 1 | row15-c4 |
| mapping_overrode_correct_baseline_possible_harm | 1 | row15-c0 |
| mapping_confirms_baseline | 1 | row15-c3 |
| mapping_rejected_conflict | 3 | row15-c1, row15-c2, row15-c6 |

**개선 사례(row15-c4)**: "수출 증가율은 작년 8월(10.9%)부터 11월까지 꺾이다가 지난달 들어
반도체 수출이 31.5% 불어나며 오름세로 돌아섰다" — baseline이 엉뚱한 표(`DT_092_115_2009_S023`,
judge="판단불가", R@1=0)를 골랐는데, discovery mapping 후보(`DT_134001_001`, 수출입총괄)를
재검증하니 HIGH로 확인돼 승격(R@1=1). **의도한 대로 작동한 사례.**

**악화 가능 사례(row15-c0, 가장 중요)**: "작년 한 해 전체 수출액이 6838억달러로 2023년에
비해 8.2% 증가했다..." — baseline이 이미 `DT_1R11006_FRM101`로 HIGH 검증(judge="일치",
R@1=1)했는데, mapping 후보(`DT_134001_001`)도 **독립적으로 재검증하니 별도로 HIGH**가
나와서(두 표 다 이 claim의 숫자와 부합할 수 있는 유사 통계) 승격 로직이 baseline의 원래
정답을 다른 표로 덮어썼다(R@1 1→0). **두 표 다 "검증 통과"라는 이유만으로, 이미 맞는 답을
있는지도 몰랐던 다른 답으로 바꿔버린 것** — `find_applicable_mapping`/승격 로직에 "baseline이
이미 HIGH면 애초에 mapping 재검증을 시도하지 않는다"는 가드가 없어서 발생했다(6번에서 정확한
수정안 제시).

**거부 사례 3건**은 설계대로 정확히 동작함 — mapping 후보를 재검증했지만 LOW/MEDIUM이 나와서
baseline을 그대로 유지했다(`mapping_reject_reason`에 사유 기록됨, 예: "judge=불일치").

## 6. 비용/시간

- 총 처리시간: **1,135.0초(18.9분)** — 사전 추정(20~35분) 범위 내, 하한에 가까움.
  - claim 추출(article-level workers=2): 604.6초(10.1분)
  - Discovery+Evaluation retrieval+검증(순차): 530.4초(8.8분)
- HCX 호출: **총 185회** — classify 20회(20.0s평균×20 없음, 평균 3.4s), claim_extract 84회
  (평균 13.2s, 최적화 프롬프트+11-shot이라 개별 호출이 길지만 총 호출수는 적음), slot_filler
  81회(평균 1.5s). 에러 0건.
- KOSIS 호출: **총 63회**, 평균 1.1초, 에러 0건(timeout 25s/retry 1회 안전망 적용 후 정체
  없이 전부 정상 응답).
- 21분 정체 재발 없음 — 두 안전망(slot_filler `requests.post` timeout, `HF_HUB_OFFLINE`)이
  모두 실측으로 검증됨(GPU 모델 로딩 23.5초, KOSIS 호출 전부 1.1초 평균).

## 7. 실패사례 (68건, `pilot20_failure_cases.jsonl`)

LOW 40건 + MEDIUM 28건. 대표 패턴:
- **untrusted_top1(LOW, 다수)**: keyword 채널이 못 찾고 리랭커도 신뢰 안 하는 경우 — RRF
  신뢰도 게이트가 정상 작동해 애초에 4단계(슬롯 채우기)까지 안 가고 조기 종료.
- **judge 판단불가(MEDIUM, 다수)**: KOSIS 실측값은 조회됐지만 주제/기간이 애매해 judge()가
  "판단불가" 반환 — 억지로 HIGH를 주지 않고 정확히 MEDIUM으로 떨어짐(spec 5번 요구사항
  "검증불가시 억지로 HIGH 주지 말고 UNKNOWN/MEDIUM" 충족).
- **mapping_conflict(LOW)**: 후보 표는 그럴듯했지만 실제 KOSIS 수치가 claim과 불일치 —
  judge="불일치"로 정확히 LOW.
- 부수 관찰(조사 범위 아님, 결과에 영향 없음): KOSIS 조회 단계에서 `pandas` 계열의
  `Undefined`류 예외가 일부 claim(약 10여 건)에서 발생했으나 전부 generic exception
  handler가 잡아 MEDIUM으로 정상 다운그레이드됐다(크래시 없음, 결과 왜곡 없음) — 정확한
  발생 지점은 이번 pilot 스코프 밖이라 조사하지 않음, 다음 pilot에서 빈도가 늘어나면
  살펴볼 항목으로만 남겨둔다.

## 8. 알려진 한계 (결과 해석 시 반드시 감안)

1. **표본 크기**: Evaluation 43건 중 gold가 확보된 건 6건뿐이고, mapping이 실제 개입한 건
   6건뿐이며, 그 6건이 전부 근접중복 기사 1쌍(row4/row15)에서 나왔다. R@k 숫자(특히 R@1
   0.8333)는 **20건 규모에서 통계적 의미를 가질 수 없다** — 방향성 참고용일 뿐 최종 성능
   주장은 절대 불가(spec 10번 요구사항 그대로 준수).
2. **R@k 지표의 구조적 편향**: gold_table_id 자체가 baseline top-1 또는 mapping 후보 중
   하나에서 나오므로, "그 후보가 자기 자신의 리스트 안에 있는지"를 재는 셈이라 R@10/50/100이
   1.0으로 수렴하는 경향이 있다(methodology.md 6번에 사전 명시한 한계) — 100~200건 확장
   시엔 gold 산정 방식을 더 독립적으로(예: 더 많은 후보 검증, 또는 일부 사람 라벨링 혼합)
   보강해야 R@k가 의미를 가진다.
3. **숫자없는 통계주장의 희소성**: 의도적으로 골랐음에도 실제 "값 없는" claim은 3/77(3.9%)뿐
   — 이 유형을 늘리려면 claim 추출 프롬프트 자체를 손봐야 하는데, 이번 pilot 스코프 밖(HCX
   프롬프트는 이미 최적화된 것을 그대로 쓰기로 grounding에서 명시).
4. **mapping 매칭 게이트(organization 2글자 substring)는 이번 20건에서 실질적으로 검증되지
   않음** — mapping 후보가 나온 유일한 케이스(row15)는 organization이 모두 "산업통상자원부"로
   일치해 이 게이트가 실제로 뭔가를 걸러내는 걸 관찰하지 못했다. 더 다양한 기관이 섞인
   100~200건에서 이 게이트의 오탐/누락률을 봐야 한다.

## 9. 절대 원칙 준수 확인
production 코드 파일 미수정(전부 import + 런타임 monkeypatch만, 이번에 추가된 안전망 2건도
동일 원칙), production DB는 SELECT만, 로컬 SQLite(`data/verifications.db`) 미사용, 70건
골든셋 gold 미사용(discovery mapping 생성에 전혀 관여 안 함, 애초에 로드하지 않음), retrieval
Top-1을 자동 정답으로 쓴 적 없음(전부 judge() 실제 수치 검증 통과가 HIGH 조건), Evaluation의
gold나 결과로 새 mapping 생성 안 함(Discovery mapping만 재사용), 전체 2,706건/재임베딩/
새 synonym dictionary/특정 table_id 하드코딩 전부 없음.

## 10. 최종 판정: **NEEDS_FIX**

**PIPELINE_VALID로 보기엔 이른 이유**: mapping override 로직에 재현 가능한 구체적 결함을
1건 발견했다(5번 "악화 가능 사례") — baseline이 이미 독립적으로 HIGH 검증한 답을, mapping
후보가 "그것과는 다른 표인데도 마찬가지로 HIGH 검증됐다"는 이유만으로 덮어쓸 수 있다. 이건
가설이 아니라 이번 20건 안에서 실제로 관측된 동작이다. Verified Mapping 접근의 핵심 약속이
"기존 baseline보다 나빠지지 않는다"인데, n=6이라는 작은 표본에서조차 이 약속이 깨지는 사례가
나왔다는 건 100~200건으로 그대로 확대하면 같은 패턴이 몇 배로 늘어날 가능성이 크다는 뜻이다.

**REJECT로 보기엔 과한 이유**: 같은 6건 안에 mapping이 의도대로 baseline의 실패를 정확히
구제한 사례(row15-c4)도 있고, mapping이 스스로 판단해 부적절한 승격을 3번 정확히 거부한
사례(judge 재검증 기반)도 있다. 즉 "mapping 후보 재검증 → 통과 못 하면 무시" 안전장치
자체는 설계대로 작동했다 — 문제는 딱 한 곳, "baseline이 이미 HIGH일 때도 mapping 재검증을
시도해서 승격까지 허용한다"는 좁은 로직 구멍이다. Discovery 단계(추출·검색·판정 엔진)
자체는 9건의 HIGH 전부가 실제 KOSIS 수치와 독립 검증됐고, 확장 없이도 "숫자없는 통계주장"을
검증한 사례까지 1건 확보했다 — 엔진 자체의 근본적 결함은 보이지 않는다.

**권장 수정(좁은 범위, 다음 pilot 전에 적용 권장)**: `process_evaluation_claim()`에서
`baseline_result["confidence"] == "HIGH"`이면 애초에 mapping 후보 재검증(probe)을 시도하지
않도록(mapping은 baseline이 HIGH가 아닐 때만 "구제 시도" 용도로 쓰고, baseline이 이미 HIGH인
claim은 무조건 그대로 유지) 가드를 추가한다 — 이 pilot 스크립트 레벨(실험 코드)의 정책
변경이라 production 코드에는 영향 없다. 이 한 가지 수정만으로 5번의 악화 사례는 구조적으로
재발하지 않는다(같은 로직을 유지한 채 표본만 키우면 동일 패턴이 반복될 위험이 있음).

**다음 단계**: 위 수정 적용 후 100~200건 재pilot 권장 — 그때는 (a) mapping이 실제 적용되는
케이스가 최소 두 자릿수 이상 나오도록(이번엔 사실상 1개 사건에서만 나옴), (b) R@k gold 산정
방식을 더 독립적으로 보강, (c) 숫자없는 통계주장 비율이 여전히 낮은지 재확인하는 것을 목표로
삼는다.
