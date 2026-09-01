# Verified Mapping + Retrieval Cascade — 실행 전 필수 보고 (20건 Pilot, Discovery 10 / Evaluation 10)

작성일: 2026-08-30 / 스코프: 이번 위임은 **20건 pilot까지만**. 목표는 2,706건 전체 실행이
아니라 "Verified Mapping + Dense/BM25/RRF Retrieval Cascade"가 구조적으로 성립하는지 20건으로
검증하는 것. 이 문서는 원래 50건 pilot용으로 작성됐던 버전을 20건 Discovery/Evaluation
분리 스코프로 갱신한 것이다(50건 버전 자산 — `pilot_run.py` — 은 그대로 재사용, 아래 3·9번
참고). 2,706건 전체(discovery/validation/evaluation 60/20/20 분리 포함)는 이 문서/실행에
포함하지 않는다 — pilot 결과와 최종 판정만 `pilot20_report.md`에 보고하고 여기서 멈춘다.

## 0-1. 21분 정체 원인 후보 2개 — 둘 다 안전망 설치(실행 전 스모크 테스트로 재확인 완료)
1. `agent/orchestrator/slot_filler.py:49`의 `requests.post(URL, headers=HEADERS, json=payload)`가
   timeout 인자 없음(코드 정적 분석으로 발견) — 전역 `requests.post`에 timeout=25s 기본값
   안전망 설치(위 pilot20_run.py 참고).
2. **2건 스모크 테스트로 실측 발견한 두 번째 원인**: `SentenceTransformer("Qwen/Qwen3-
   Embedding-4B", ...)`가 모델이 이미 로컬에 완전히 캐시(7.6GB)돼 있는데도
   huggingface_hub이 파일마다 원격 etag를 재검증하려다 이 sandbox 네트워크 환경에서
   580초 타임아웃까지 응답 없이 멈췄다("Loading weights: 100%"는 찍히고 그다음 줄이 안
   나옴 — GPU 연산이 아니라 HTTP 검증 단계에서 멈춘 것). `HF_HUB_OFFLINE=1`+
   `TRANSFORMERS_OFFLINE=1`로 재현 테스트하니 5.1초에 로드 완료 — 이게 지난 21분 정체의
   실제 원인이었을 가능성이 1번보다 오히려 크다(1번은 KOSIS/HCX 응답을 실제로 기다리는
   경로라 21분씩 걸리려면 서버가 응답을 아예 안 줘야 하는데, 2번은 이 sandbox에서 매번
   재현 가능한 확정적 지연이었음). `pilot20_run.py`에 두 안전망 모두 설치, production
   파일은 둘 다 미수정(환경변수만).

## 0. prd_se(시점 코드) 버그 수정 이력 — 실행 재개 배경
직전 시도에서 pilot을 짜던 중 4단계(슬롯 채우기) 검증이 "시점 주기(prd_se) 선택" 버그의
영향을 받는다는 게 발견돼 pilot 실행을 한 차례 중단했다. production에서 별도로 수정·검증
완료됨(2026-08-30, 이번 위임 밖에서 진행):
- `agent/pipeline/batch_runner.py::_infer_desired_granularity`에 일(D)/반기(H)/다년(F) 인식
  추가(기존엔 분기/월만 인식) + `_build_dynamic_kosis_slots`가 64개 카탈로그 표 경로가 이미
  쓰던 `_select_prd_se(supported, desired)`를 VDB 전용 표 경로에서도 재사용하도록 수정
  (예전엔 표에서 처음 찾은 주기 하나를 claim이 뭘 원하든 그대로 썼음).
- `agent/kosis/enrich_objl.py::fetch_table_detail`이 D/M/Q/H/Y/F/IR 전체를 확인해
  `prd_se_list`를 반환하도록 수정, `agent/kosis/detail_cache.py`가 리스트 전체를 캐시.
- 실제 KOSIS API(`DT_1DA7024S`)로 검증 완료, 기존 테스트 회귀 없음 확인됨(coordinator 보고).
- 이 pilot의 `verify_table_for_claim()`은 `run_stage_4`/`run_stage_5_6`을 그대로 호출하므로
  수정된 로직을 자동으로 사용한다(이 스크립트 자체는 수정 불필요, 시그니처 불변 확인함).

## 1. 70건 골든셋 위치 (2번째 grounding 재확인)
`benchmark/search_experiment/eval_set.json` — 70개 claim, 각
`{claim_id, sentence, claim_type, period, unit, gold: [table_id], gold_status, gold_missing}`.
article_url/article_title 필드 없음 — claim.sentence 텍스트만 있음. 이번 20건 pilot에서는
**discovery mapping 생성에 절대 사용하지 않는다**(순수 external anchor로만) — 아래 5번에서
"이번 20건은 leakage 체크 자체를 생략한다"는 결정과 사유를 명시한다.

## 2. 2,706건 schema
`data/data_set.csv` — 컬럼: `기사제목, 작성일, URL, 기사 본문 전체, 검색 구분 레이블`.
`검색 구분 레이블 == "true"`인 행 2,507건(`agent/pipeline/batch_runner.py::_load_csv_rows()`가
이 필터를 적용). article_id 없음 → 이번 실험도 50건 pilot과 동일하게 원본 CSV 행 인덱스
기반 `row{N}` 형식을 그대로 쓴다.

## 3. 기존 retrieval pipeline — 50건 pilot과 동일(재확인)
`agent/mapping/reranker.py::search_and_rerank(claim, keyword_fn, embedding_fn, vdb_fn, bm25_fn)`
진입점, Dense(`agent.kosis.query_vdb.batch_query_vdb`, Qwen3-Embedding-4B 2560차원)+
BM25(`bm25_query_vdb`)+Keyword(64개 카탈로그)+Embedding(64개 카탈로그) → `_merge_candidates`
→ `_rrf_fuse_no_ce`(8채널 RRF, **`_DISABLE_RERANKER=True` 기본값이라 CE 없음** — 2026-08-30
production 변경). 질의 텍스트는 Context D2(`prev_sentence + sentence`). **이 함수들은 전부
수정 없이 import해서 그대로 호출한다.**

검증(4~8단계)도 `agent.orchestrator.calc_type_router.route_calc_type`,
`agent.pipeline.batch_runner.run_stage_4`/`run_stage_5_6`, `agent.kosis.api_client.KosisApiClient`,
`agent.kosis.calculator.KosisCalculator`, `agent.verdict.judge.judge`를 그대로 재사용(파일
수정 없음, import만). `db.store.insert_verification()`은 호출하지 않는다(로컬 SQLite에
실험 데이터를 쓰지 않음, 아래 9번).

**R@100 측정을 위한 유일한 조정**: `agent/kosis/query_vdb.py`의 `VDB_TOP_K`/`LEXICAL_TOP_K`는
각각 환경변수 `DENSE_TOP_K`(기본 10)/`BM25_TOP_K`(기본 30)를 import 시점에 읽는 모듈 상수라,
이 두 값이 작으면 RRF 융합 후보군이 100위까지 못 채워진다. `pilot20_run.py`는 이 두 환경변수를
**production import 전에** 100으로 설정한다 — 코드 변경이 아니라 이미 있는 설정 파라미터를
이번 실행 프로세스에서만 넓게 쓰는 것(운영 DB/코드에는 전혀 영향 없음). `search_and_rerank`의
`top_k` 인자도 20(50건 pilot 기본값)이 아니라 100으로 호출해 fused 리스트 전체를 후보로 저장한다.

## 4. Discovery(10)/Evaluation(10) split 방식 — 이번 20건에 실제 적용
50건 pilot 문서는 "2,706건 확장 시 계획(참고용)"만 적어뒀는데, 이번 20건은 사용자가 명시적으로
분리를 요구했으므로 실제로 적용한다.

**20건 선정 — 단순 랜덤 금지, claim 유형 다양성 확보(`pilot20_run.py::select_pilot20_articles`)**:
- HCX로 실제 claim을 뽑기 전에는 정확한 claim 유형을 알 수 없으므로, 기사 **본문 표층
  신호(정규식)** 로 6개 버킷(qualitative_no_number/threshold/trend/superlative/comparative/
  numeric)을 미리 추정해 버킷별 쿼터(4/3/3/3/4/3=20)를 원본 CSV 행 순서대로 채운다 —
  무작위 요소 전혀 없음(같은 CSV면 항상 같은 20건). 사전 실측(`검색 구분 레이블=True`
  2,507건 대상 버킷 카운트): numeric 1021 / qualitative_no_number 386 / trend 312 /
  comparative 267 / threshold 238 / superlative 159 / 미분류 124 — 쿼터를 채우기에 충분함
  확인.
- **이 버킷은 "선정 단계의 사전 필터"일 뿐, 실제 claim 유형이 아니다.** 실제 유형은 HCX
  추출 후 production `claim_type` + `route_calc_type()` 결과 + 원문 정규식 신호를 조합한
  `extended_claim_type()`(50건 pilot과 동일 로직, 아래 6번)으로 재측정해서
  `pilot20_report.md`에 실측 분포로 보고한다 — 사전 필터의 정확도를 결과로 주장하지 않는다.
- **split 배정**: 버킷 우선순위(qualitative_no_number > threshold > trend > superlative >
  comparative > numeric) + CSV 순서로 만든 "선정 순서" 전체를 관통하는 전역 인덱스 기준으로
  짝수 번째는 discovery, 홀수 번째는 evaluation에 배정한다(버킷 내부가 아니라 전체 선정
  순서 기준 — 정확히 10/10이 되도록 보장, 각 버킷도 대체로 양쪽에 걸침).
- **실측 선정 결과**(`pilot20_articles.json` 참고): discovery 10건(row1/3/4/5/12/25/45/50/
  62/76), evaluation 10건(row0/2/6/10/15/17/21/27/60/65). 버킷 분포 comparative=4,
  qualitative_no_number=4, numeric=3, threshold=3, superlative=3, trend=3 — 목표 쿼터 정확히
  달성.
- **관찰(제약 아님, 해석 시 유의사항)**: row4(discovery, "2024년 수출 6838억달러 역대 최대")와
  row15(evaluation, "반도체 덕에… 지난해 수출 6838억달러 역대 최대")는 서로 다른 기사지만
  같은 사건(2024년 수출 실적)을 다룬 근접중복이다. 2,706건 전체 확장 시엔 근접중복 클러스터
  단위로 split을 배정할 계획(50건 pilot 문서 4번에 이미 명시)이지만, 이번 20건 pilot은
  "파이프라인이 구조적으로 작동하는지"만 보는 스코프라 이 필터를 적용하지 않았다 — 대신
  Mapping Reuse Test 결과 해석 시(`pilot20_report.md`) 이 쌍이 mapping 재사용 성공에
  기여했다면 "우연히 쉬운 사례"일 수 있다는 점을 별도로 표시한다.

Discovery 10건에서만 claim 추출→retrieval→검증을 돌려 HIGH mapping을 생성한다(50건 pilot과
동일한 `process_discovery_claim` 로직, 재사용 가능한 기존 mapping이 없으므로 매 claim이
"No Mapping → fallback retrieval → 최초 검증" 경로). Evaluation 10건은 Discovery mapping만
재사용 가능하고, Evaluation의 gold나 결과로 새 mapping을 생성하지 않는다(아래 6번).

## 5. Near-duplicate / Golden Set 리크 방지
70건 골든셋(`eval_set.json`) leakage 체크는 **이번 20건 pilot에서 생략한다** — 사용자 grounding이
명시적으로 "이번 20건 discovery mapping 생성에 gold_table_id를 사용 안 하면 충분, leakage
체크까지 할 필요는 없다(2,706건 본실행 때 항목)"고 지시했다. 대신 gold_table_id는 애초에
로드조차 하지 않는다 — discovery/evaluation 어느 단계에서도 `eval_set.json`을 읽는 코드가
`pilot20_run.py`에 없다(50건 pilot의 `check_leakage()`도 이번 스크립트엔 없음 — grep으로
`eval_set` 문자열이 `pilot20_run.py`에 전혀 등장하지 않음을 확인 가능).

## 6. Mapping Reuse Test (A vs B) — gold 산정 방식 (신규, 이번 20건 핵심)
Evaluation claim마다 **두 후보**만 KOSIS로 독립 재검증한다(Top-100 후보 전체를 검증하지
않음 — KOSIS 호출 폭증 방지, 50건 pilot의 "top-1만 검증" 원칙을 그대로 확장):
1. **baseline top-1**(A) — Dense+BM25+RRF만으로 나온 순수 결과.
2. **mapping 후보**(B, 있는 경우만) — `find_applicable_mapping()`이 고른 Discovery HIGH
   mapping 표.

**mapping "적용 가능" 판정(새 synonym dictionary 없음)**: mapping 표의 table_id가 이 claim
자신의 baseline top-100 후보 리스트 안에 이미 있는지(기존 8채널 RRF 신호 자체를 게이트로
씀 — 새 사전 없이 검색이 이미 준 신뢰도만 재사용) + organization이 둘 다 있으면 느슨한
substring 비교(2글자 겹침)만 확인한다. **이 조건을 만족해도 "적용 가능 후보"일 뿐 즉시
승격하지 않는다** — 반드시 `verify_table_for_claim()`으로 그 표를 claim의 현재 period/
region/slots 기준으로 KOSIS 재조회 + `judge()` 재검증을 통과해야만(confidence=HIGH) 실제
top-1로 승격한다. 통과 못 하면(충돌/검증불가) mapping을 버리고 baseline을 그대로 쓴다
(`mapping_reject_reason`에 사유 기록) — "mapping 있으면 무조건 top-1 고정" 금지 원칙 준수.

**gold 정의**: baseline이 HIGH면 baseline 표가 gold(`gold_source=baseline_verified`).
baseline이 HIGH가 아니고 mapping 후보가 독립 재검증으로 HIGH가 되면 mapping 표가 gold
(`gold_source=mapping_verified` — "mapping이 baseline이 놓친 걸 찾아낸" 사례). 둘 다 HIGH가
아니면 이 claim은 **gold 없음**으로 분류하고 R@k 분모에서 제외한다(`pilot20_report.md`에
"gold 없는 claim 수"로 별도 보고 — 억지로 gold를 만들지 않는다).

**R@k 계산**: baseline 후보 리스트(최대 100위)에서 gold table_id의 순위를 그대로 쓴다
(baseline R@k). assisted 리스트는 mapping이 검증 통과해서 승격됐을 때만 그 표를 1위로
올리고 나머지는 baseline 순서를 유지한 리스트(assisted R@k) — mapping이 없거나 거부되면
baseline과 완전히 동일해 두 지표가 같아진다(중립).

**알려진 비대칭성(한계, 결과 해석 시 명시)**: gold가 `baseline_verified`인 claim은 정의상
baseline R@1이 항상 1이다(baseline 스스로 찾은 답을 다시 baseline이 맞혔다고 세는 구조) —
이 부분만 놓고 보면 baseline이 유리해 보일 수 있다. 반대로 `mapping_verified`인 claim은
mapping이 없으면 애초에 gold 자체가 안 생겼을 claim이라 baseline R@1이 구조적으로 낮게
나온다. `pilot20_report.md`는 이 두 gold_source를 나눠서 보고해 이 비대칭을 감추지 않는다.

**"Mapping이 실제 개선/악화" 판정**: 정확한 정의는 `pilot20_run.py::process_evaluation_claim`의
`ab_outcome` 계산부(`no_mapping_applicable`/`mapping_rejected_conflict`/
`mapping_confirms_baseline`/`mapping_overrode_correct_baseline_possible_harm`/
`mapping_rescued_claim_improvement`/`mapping_applied_no_change`)에 고정돼 있다. "harm"은
mapping이 검증 통과한 표로 승격했는데 그 표가 baseline의 원래 HIGH 표와 다를 때만 잡는다
(둘 다 독립적으로 HIGH인 서로 다른 표 — 실제로는 하나만 진짜 정답일 수 있는 애매한 경우,
"harm 가능성"으로 보수적으로 센다).

## 7. claim 추출 — 최적화 프롬프트 (grounding 2번 재확인, 이번 20건 기본값)
`benchmark/prompt_optimization_experiment/optimized_prompt.txt`(11-shot) +
`_MAX_RECOVERY_ROUNDS=1`(조건부 게이트, `find_missed_candidates()`가 아무것도 못 찾으면
recovery 스킵) + article-level `workers=2`(4/8은 429 에러 다발로 탈락, 2가 실측 최적치).
`prompt_optimization_experiment/run_experiment.py`의 Config B 오케스트레이션을
`pilot20_run.py::extract_claims_optimized`/`run_extraction_stage`에 그대로 복제(그 스크립트
자체는 계측 코드가 섞여 있어 import하지 않음). 50건 A/B 실측: HCX calls/article
4.84→3.14(-35%), Total time 2,288.7s→1,051.9s(-54%), 품질은 numeric/comparative/
superlative/trend/threshold 전부 88~94% 유지, **"숫자없는 통계claim"만 72.5%로 상대적으로
약함** — 이번 20건(qualitative_no_number 버킷 4건 포함)에서 이 유형이 실제로 몇 개나
뽑히는지 `pilot20_report.md`에서 특별히 확인한다.

## 8. Verification Engine 독립성 — 50건 pilot과 동일 근거, Evaluation 단계 재확인
Engine(코드 규칙): `judge._rule_based_verdict` + LLM 폴백 — production 코드 그대로. Discovery
단계는 50건 pilot과 완전히 동일(재사용 가능한 mapping이 아직 없음, 모든 claim이 최초 검증
경로). **Evaluation 단계에서 처음으로 "mapping 재사용" 경로가 실제로 생긴다** — 여기서도
순환 논리가 아닌 이유: mapping 후보가 승격되려면 반드시 그 claim 고유의 period/region/
slots로 KOSIS를 다시 조회하고 `judge()`가 실제 수치를 다시 비교해야 한다(Discovery 때
검증했던 것과 같은 표라도, Evaluation claim의 시점/조건이 다르면 다른 실제값과 비교됨) —
"발견 로직=평가 로직 재사용"이 아니라 "발견된 표 후보=평가 로직이 매번 새로 검증"하는
구조다. Confidence 산정 규칙(`classify_confidence()`)은 50건 pilot에서 고정한 그대로
재조정 없이 사용한다(2,706건 확장 시에도 유지 원칙, 50건 문서 6번과 동일).

**알려진 한계(중단 사유 아님, 결과 해석 전제로 명시)**: (a) trend의 monotonic 검증, threshold의
기준값 검증은 production에 전용 계산 경로가 없어 이번 pilot도 라벨링만 하고 전용 로직을
새로 만들지 않음(50건 pilot 문서 8번과 동일 결정, 재사용 아닌 신규 개발 방지). (b) claim당
검증하는 후보는 최대 2개(baseline top-1 + mapping 후보 1개)뿐, Top-100 전체를 검증하지
않는다(KOSIS 호출 폭증 방지) — 따라서 R@k는 "이 2개 후보 중 gold가 있었는지"의 근사치이고,
baseline이나 mapping 둘 다 놓친 진짜 정답이 top-100 어딘가에 더 있을 가능성은 이번 pilot
스코프에서 확인하지 않는다.

## 9. 산출물 스코프 (20건 pilot)
```
benchmark/verified_mapping_experiment/
    methodology.md               <- 이 문서 (20건 스코프로 갱신)
    progress.md                  <- 갱신
    pilot_run.py                 <- 50건 pilot 스크립트(원본, 수정 없이 import만 당함)
    pilot20_run.py                <- 이번 20건 실행 스크립트(신규, pilot_run.py를 재사용)
    pilot20_articles.json
    pilot20_claims.jsonl
    pilot20_candidates.jsonl
    pilot20_verified_mappings.jsonl
    pilot20_failure_cases.jsonl
    pilot20_run_stats.json
    pilot20_report.md             <- 이번 위임의 최종 산출물, 8~10번 항목 + 최종 판정
```
50건 pilot의 산출물(`pilot_claims.jsonl` 등, 전부 빈 파일로 남아있음 — 실제 50건 실행은
이 20건 pilot으로 대체됨)은 그대로 두고 덮어쓰지 않는다.

## 10. 절대 금지 재확인
production DB write 금지(SELECT만, `SUPABASE_DB_URL`), production retrieval/reranker/
api_client/calculator/judge/batch_runner 코드 파일 수정 금지(전부 import만 — 이번 20건
스크립트가 새로 하는 유일한 "코드 레벨 개입"은 자기 프로세스 안에서의 런타임 monkeypatch
2건: ① `requests.post` 기본 timeout 안전망, ② `claim_extractor._load_prompt_template`
교체 — 둘 다 원본 파일은 바이트 하나도 안 바뀜), 모델 fine-tuning 금지, 새 synonym
dictionary 하드코딩 금지(mapping 적용 게이트는 기존 retrieval 신호 재사용, 위 6번),
특정 table_id/article_id 예외처리 금지, 70건 골든셋 gold를 discovery mapping 생성에 사용
금지, retrieval Top-1 자동 정답 간주 금지(judge() 실제 수치 검증을 거쳐야만 HIGH), 애매하면
UNKNOWN, Evaluation의 gold나 결과로 mapping 신규 생성 금지, 전체 2,706건/corpus 재임베딩
금지. 로컬 SQLite(`data/verifications.db`)에도 실험 레코드를 쓰지 않는다(production
insert_verification 미호출).
