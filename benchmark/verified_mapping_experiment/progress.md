# Progress Log — Verified Mapping + Retrieval Cascade (50건 Pilot)

## 스코프
사용자 지시: 이번 위임은 **50건 pilot까지만**. 완료 후 결과 + 2,706건 외삽 수치만 보고하고
2,706건 전체(또는 discovery/validation/evaluation 분할) 실행은 하지 않는다.

## 진행 순서
1. [완료] 저장소 구조 확인 — `data/data_set.csv`(2,706행, `검색 구분 레이블=True` 2,507행),
   `benchmark/search_experiment/eval_set.json`(70건, URL/제목 없음, sentence만),
   production retrieval(`agent/mapping/reranker.py` `search_and_rerank`, 2026-08-30부로
   `_DISABLE_RERANKER=True` 기본값 확인), KOSIS API client/calculator/judge 구조 확인.
2. [완료] `methodology.md` 작성 — 실행 전 필수 보고 10개 항목.
3. [완료] `pilot_run.py` 작성 — production 코드를 그대로 import해서 재사용(수정 없음),
   로컬 SQLite(`data/verifications.db`)에는 쓰지 않도록 `insert_verification()`을 아예
   호출하지 않는 방식으로 설계(개별 stage 함수만 직접 호출).
4. [완료] import/구문 스모크 테스트, `load_pilot_articles`/`check_leakage` 단독 테스트(5건) —
   정상 동작 확인.
5. [진행] PILOT_N=2로 엔드투엔드 스모크 실행(GPU 로딩+HCX+KOSIS 실제 호출 확인) 후, 문제
   없으면 PILOT_N=50 본 실행.
6. [대기] 본 실행 결과로 `pilot_claims.jsonl`/`pilot_candidates.jsonl`/
   `pilot_verified_mappings.jsonl`/`pilot_run_stats.json` 생성.
7. [대기] `pilot_report.md` 작성 (7개 항목 + 외삽 수치) — 이 문서가 이번 위임의 최종 산출물.

## 설계상 주요 결정 (요약, 근거는 methodology.md)
- Verification Engine: production `judge()`/`run_stage_4`/`run_stage_5_6`/
  `route_calc_type()`을 그대로 재사용(수정 없음). 이번 pilot엔 재사용 가능한 기존 mapping이
  없으므로 모든 claim이 "No Mapping → fallback retrieval → 최초 검증" 경로.
  Mapping Hit/Mapping+Independent Verification Hit 재사용 경로는 이번 pilot 범위 밖(2,706건
  확장 이후에나 발생).
- Confidence 산정 규칙은 `pilot_run.py::classify_confidence()`에 고정 — 결과가 마음에 안
  든다고 사후 조정하지 않는다(spec의 "Validation에서 확정 후 재조정 금지" 원칙을 이 단일
  규칙에도 적용).
- 확장 taxonomy(numeric/comparative/superlative/trend/threshold/qualitative)는 **새 LLM
  프롬프트가 아니라, production claim_type + route_calc_type() + 원문 정규식 신호를 조합한
  규칙 기반 재분류**로 구현(스코프 축소, methodology.md 3·8번에 근거 명시). trend의
  monotonic 검증, threshold의 기준값 검증은 production에 전용 계산 경로가 없어 이번 pilot도
  구현하지 않음 — 라벨링만 하고 알려진 한계로 보고.
- 70건 골든셋 leakage 체크는 URL/제목이 아니라 sentence 부분 문자열 포함 여부로 수행
  (eval_set.json에 URL/제목 필드 자체가 없음, methodology.md 5번).

## 실행 로그
(본 실행 완료 후 실제 수치로 갱신 — pilot_report.md에 최종 수치 정리)

---

# [추가] 20건 Pilot (Discovery 10 / Evaluation 10) — 2026-08-30, 중단 상태

## 상태: **중단(HOLD)** — production 실행 보류, 코드만 작성 완료, 실행은 아직 안 함

조정자(coordinator) 지시로 중단: "pilot의 숫자검증(4단계) 단계가 별도로 발견된 prd_se 버그에
영향받는다는 게 확인돼서, 그 버그를 먼저 production 코드에서 고친 뒤에 pilot을 다시 시작하기로
했다." — **`pilot20_run.py`는 아직 한 번도 실행하지 않았다**(GPU 로딩도, HCX/KOSIS 호출도 전혀
발생하지 않음). 실행 중이던 백그라운드 프로세스 없음(`ps aux` 확인 완료, 정리할 것 없음).

## 이번 위임에서 완료한 것 (코드만, 미실행)
1. 저장소/기존 자산 재확인: `benchmark/verified_mapping_experiment/methodology.md`,
   `pilot_run.py`(50건 pilot, 그대로 재사용 대상), `benchmark/prompt_optimization_experiment/`
   (optimized_prompt.txt, run_experiment.py의 Config B 몽키패치 기법), `agent/mapping/reranker.py`
   (`_DISABLE_RERANKER=True` 확인), `data/data_set.csv`(2,507건 True 라벨).
2. **21분 정체 재발 원인 특정**: `agent/kosis/api_client.py`(KOSIS_TIMEOUT=10s 기본값 이미 있음)와
   `agent/preprocessing/hcx_client.py`(DEFAULT_HARD_TIMEOUT_SECONDS=120s 이미 있음)는 원인이
   아닐 가능성이 크고, **`agent/orchestrator/slot_filler.py:49`의
   `requests.post(URL, headers=HEADERS, json=payload)`가 timeout 인자를 전혀 안 넘기는 게
   유일하게 timeout 없는 런타임 경로**임을 `agent/` 전체 grep으로 확인(다른 requests 호출은
   전부 명시적 timeout 있음, 유지보수 스크립트 제외). `pilot20_run.py`에 전역 `requests.post`
   safety-net(값 없을 때만 timeout=25s 채움, production 파일 미수정)을 설치해뒀음 —
   **이 발견이 이번에 코드에는 반영했지만, 아직 실행/검증은 안 됐다.**
3. `benchmark/verified_mapping_experiment/pilot20_run.py` 작성 완료(신규 파일, 미실행):
   - `pilot_run.py`를 `import pilot_run as pr`로 재사용(CallCounter, classify_confidence,
     extended_claim_type, CountingKosisApiClient, build_vdb_bm25_fns 등 전부 재사용, 새로
     안 만듦).
   - 20건 deterministic 선정(`select_pilot20_articles`): 정규식 기반 6버킷
     (qualitative_no_number/threshold/trend/superlative/comparative/numeric) 쿼터를 원본
     CSV 순서대로 채움, 무작위 없음. 사전 실측(`data_set.csv` 2,507건 대상): 버킷별 후보
     수 numeric 1021 / qualitative_no_number 386 / trend 312 / comparative 267 /
     threshold 238 / superlative 159 / 미분류 124 — 쿼터(4/3/3/3/4/3=20) 충분히 채울 수
     있음 확인.
   - claim extraction: `optimized_prompt.txt` + 조건부 recovery(1라운드) + workers=2
     (`run_extraction_stage`, `prompt_optimization_experiment/run_experiment.py`의 Config B
     오케스트레이션 복제).
   - `verify_table_for_claim()`: `pilot_run.process_claim`의 stage4 이후 로직을 임의
     table_id에 대해 쓸 수 있게 일반화(baseline top1 검증과 mapping 후보 검증에 공용으로 씀).
   - Discovery(`process_discovery_claim`)/Evaluation(`process_evaluation_claim`) 분리,
     `find_applicable_mapping()`(새 synonym dictionary 없이, mapping 표가 claim 자신의
     baseline top-100 후보 안에 있는지 + organization 느슨한 substring 비교만으로 게이트,
     최종 승격은 반드시 KOSIS 재조회+judge() 재검증 통과해야 함), R@1/10/50/100 계산 로직
     (gold = baseline 또는 mapping 후보 중 judge()로 독립 검증된 쪽, 둘 다 실패하면 gold
     없음으로 분모 제외) 전부 구현 완료.
   - `DENSE_TOP_K=100`/`BM25_TOP_K=100` env로 R@100 측정 가능하게 채널 후보 폭을 넓힘
     (production 코드 미수정, env-only).

## 다음 세션에서 이어서 할 것
1. **prd_se 버그가 production에서 수정됐는지 먼저 확인**(어느 파일/함수인지는 이번 위임
   범위 밖 — coordinator가 별도로 고친다고 함). 수정 확인 전에는 `pilot20_run.py` 절대
   실행하지 말 것.
2. 수정 확인 후: `PILOT_N`류 축소 없이 바로 스모크(예: articles 앞 2~4건만) 먼저 돌려서
   slot_filler timeout 안전망 + 최적화 프롬프트 경로 + Discovery/Evaluation 분기가 실제로
   동작하는지 확인 후 20건 본 실행.
3. 본 실행 후 `pilot20_report.md` 작성(8~10번 항목 + 최종 판정) — 아직 시작 안 함.
4. methodology.md는 아직 20건 스코프로 갱신 안 함(50건 pilot 내용 그대로) — 다음 세션에서
   위 20건 설계(버킷 선정 방식, mapping 적용/gold 산정 규칙, prd_se 버그로 인한 재시작 이력)를
   반영해 갱신 필요.

## 산출물 현재 상태 (2026-08-30 최종 업데이트)

## 상태: **완료**

prd_se 버그가 production에서 수정된 뒤(coordinator 보고) 재개해서 끝까지 완료했다.

1. `methodology.md` — 20건 Discovery/Evaluation 스코프로 전면 갱신 완료(prd_se 수정 이력,
   21분 정체 원인 2건 모두 문서화, 버킷 선정 방식, Mapping Reuse Test gold 산정 방식 전부
   반영).
2. 스모크 테스트 2회 진행 중 **새 정체 원인을 하나 더 실측 발견**: `SentenceTransformer`가
   모델이 이미 로컬에 완전히 캐시돼 있어도 huggingface_hub이 원격 etag를 재검증하려다
   이 sandbox에서 580초+ 멈춤(1차 스모크가 이걸로 timeout 킬됨) — `HF_HUB_OFFLINE=1`+
   `TRANSFORMERS_OFFLINE=1`로 해결(재현 테스트 5.1초). `pilot20_run.py`에 안전망 추가,
   2차 스모크(4건)부터 정상 완료 확인.
3. **20건 본 실행 완료**(1,135.0초=18.9분, HCX 185회/KOSIS 63회, 에러 0건). 산출물 전부 생성:
   `pilot20_articles.json`/`pilot20_claims.jsonl`(77건)/`pilot20_candidates.jsonl`/
   `pilot20_verified_mappings.jsonl`(4건)/`pilot20_failure_cases.jsonl`(68건)/
   `pilot20_run_stats.json`.
4. `pilot20_report.md` 작성 완료 — 8~10번 항목 전부 + 최종 판정.

## 최종 판정: **NEEDS_FIX**
Discovery 단계(추출·검색·판정 엔진)는 정상 동작(HIGH 9건 전부 실제 KOSIS 수치와 독립 검증,
"숫자없는 통계주장" 검증 성공 사례도 1건 확보). 그러나 Mapping Reuse Test(Evaluation)에서
**재현 가능한 구체적 결함 1건** 발견: baseline이 이미 HIGH로 검증한 답을, mapping 후보가
"별도로 HIGH"라는 이유만으로 덮어써 R@1을 1→0으로 악화시킨 사례(row15-c0). 권장 수정(좁은
범위): baseline이 이미 HIGH면 mapping 재검증 자체를 시도하지 않도록 가드 추가 후 재pilot.
자세한 근거는 `pilot20_report.md` 10번 참고.

## 산출물 최종 상태
- `pilot20_run.py`: 작성+실행 완료.
- `pilot20_articles.json`/`pilot20_claims.jsonl`/`pilot20_candidates.jsonl`/
  `pilot20_verified_mappings.jsonl`/`pilot20_failure_cases.jsonl`/`pilot20_run_stats.json`/
  `pilot20_report.md`: 전부 생성 완료.
- `methodology.md`: 20건 스코프로 갱신 완료.
- `pilot20_smoke2_*`: 4건 스모크 테스트 산출물(검증용, 참고 자료로 보존).
