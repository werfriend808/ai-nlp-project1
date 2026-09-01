# methodology.md — Verified Mapping + Retrieval Cascade, 819건 전체 실행

`benchmark/verified_mapping_experiment/pilot20_report.md`(20건 pilot, baseline-보호 회귀
수정 포함)의 후속. 파이프라인 로직 자체는 새로 만들지 않고 `pilot20_run.py`/`pilot_run.py`를
그대로 import해서 재사용한다(`benchmark/verified_mapping_819/run819.py`). 이 문서는 819건
스케일에서 **새로 추가/결정한 부분만** 다룬다.

## 1. 대상 819건과 row_index의 의미

대상 파일: `data/article_priority_institution_mentioned.json` (819건, 기관 매칭 우선순위
기사). 각 항목의 `row_index`는 **`data/data_set.csv`를 필터링 없이 원본 순서 그대로
`csv.DictReader`로 읽었을 때의 인덱스**다(20건 pilot/50건 pilot이 쓰던
`_load_csv_rows()`의 인덱스는 `'검색 구분 레이블'==True`로 필터링된 리스트 기준이라
서로 다르다 — 혼동하기 쉬워서 명시).

실측 검증(코드 실행으로 확인, 추측 아님):
- `row_index=15` -> 원본 CSV 15번째 행 제목 "이창용 "崔대행, 경제 고려한 결정..." (JSON과 일치)
- `row_index=17` -> "반도체 덕에… 지난해 수출 6838억달러 역대 최대" (JSON과 일치)
- 819건 전부 `row_index < len(원본 2706행)`, 전부 `'검색 구분 레이블'=='true'`, url 전부 일치,
  title은 817/819 일치(2건은 원본 CSV 쪽에 파일 중간 BOM 잔재 mojibake가 섞여 있을 뿐 —
  `_row_to_article`은 title 문자열 매칭이 아니라 직접 인덱싱이라 결과에 영향 없음).

따라서 `run819.py::_load_raw_csv_rows()`는 `_load_csv_rows()`와 동일한 BOM 정리 로직을
쓰되 `'검색 구분 레이블'` 필터를 걸지 않고, `row_index`로 직접 인덱싱해 production 함수
`pr._row_to_article()`에 넘긴다. `article_id`는 20건 pilot과 동일한 컨벤션인
`f"row{row_index}"` 포맷을 그대로 쓰되, 값은 819 대상 파일이 이미 부여한 `row_index`를
그대로 쓴다(새 ID 스킴을 만들지 않음).

## 2. Discovery / Evaluation 70:30 split

- `SEED_819 = 20260830819` (고정 상수, `pilot20`의 `SEED=20260830`/`pilot_run`의
  `SEED=20260830`과 겹치지 않음).
- `random.Random(SEED_819).shuffle(articles)` 후 앞 70%를 discovery, 뒤 30%를 evaluation으로
  배정(반올림으로 discovery 개수 결정, 나머지가 evaluation). 819건 기준 discovery≈573건,
  evaluation≈246건.
- 셔플을 **먼저** 하기 때문에 원본 파일의 기사 발행일/기관 우선순위 등 어떤 순서 신호와도
  상관되지 않는다.
- claim-type 버킷 다양성(20건 pilot이 쓴 `BUCKET_QUOTAS`)은 819 규모에서는 표본 크기 자체가
  이미 충분히 다양한 claim 유형을 자연스럽게 포함하므로 과설계로 판단해 적용하지 않는다
  (spec 명시적 지시). 이로 인해 `claims_out`의 `selection_bucket` 필드는 819 실행에서는
  항상 `null`이다(20건 pilot과 스키마 호환을 위해 필드 자체는 유지).

## 3. Discovery/Evaluation loop — pilot20 로직 그대로 재사용

- claim extraction: `pr.install_hcx_instrumentation()` + `p20.install_optimized_extraction()`
  (prompt_optimization_experiment B설정 — 조건부 recovery 1라운드), article-level
  `ThreadPoolExecutor(max_workers=2)` (`p20.run_extraction_stage` 그대로 재사용 — 4/8 workers는
  기존 실험에서 429 폭주로 탈락한 전례가 있어 2 유지).
- Discovery: `p20.process_discovery_claim()` 그대로 — retrieval top-100(RRF-only, CE 비활성)
  → top1 후보만 KOSIS 재조회+judge() 검증 → HIGH만 `discovery_mappings`에 누적.
- Evaluation: `p20.process_evaluation_claim()` 그대로 — **baseline-보호 규칙 포함**
  (2026-08-30 회귀 수정: baseline이 이미 `confidence=HIGH`면 mapping 후보를 재검증조차
  하지 않고 무시 — `mapping_skipped_baseline_protected` outcome). gold 산정 규칙, R@k 계산
  방식도 20건 pilot과 완전히 동일.
- retrieval `top_k=100`(`TOP_K_RETRIEVAL`)으로 R@1/10/50/100 전부 측정.

## 4. A / B / C 정의 (사후 계산, 별도 파이프라인 재실행 없음)

같은 파이프라인을 여러 번 돌리는 대신, evaluation claim 1건당 이미 기록되는 필드에서
`run819.py::write_final_outputs()`가 실행 종료 후 한 번에 계산한다.

- **A (baseline)** = evaluation claim의 `baseline_recall` 평균 (mapping을 전혀 쓰지 않고
  retrieval top1을 그대로 검증했을 때의 R@k). 모집단: `ab_status=="scored"`인 evaluation
  claim 전체(= gold_table_id가 확정된 claim, baseline이 HIGH이거나 mapping이 rescue한 경우).
- **C (mapping-assisted)** = 같은 모집단에서 `assisted_recall` 평균 — baseline-보호 규칙이
  이미 반영된 최종 파이프라인 결과(mapping이 baseline을 절대 덮어쓰지 않고, baseline이
  HIGH가 아닐 때만 mapping이 기여할 수 있음).
- **B (mapping-only)** = **`mapping_hit_table_id`가 존재하는 evaluation claim만의 부분집합**에서
  `assisted_recall` 평균. 즉 "discovery mapping이 애초에 후보로조차 안 걸린 claim"은 B의
  분모에서 제외(그런 claim에 대해 mapping만으로 답하는 것 자체가 정의 불가능하므로 기권
  처리). mapping이 걸렸지만 baseline과 동일 표를 가리켜 사실상 baseline과 같은 경우
  (`ab_outcome=="mapping_confirms_baseline"`)도 B에는 포함하되(assisted_recall == baseline
  일 것), `results.csv`/`evaluation.json`의 outcome 분포에서 이 경우가 몇 건인지 별도로
  드러나므로 "mapping이 순수하게 baseline과 달랐던 기여분"을 구분해서 읽을 수 있다.
  이 정의는 spec에서 애매할 경우 명시하라고 지시한 대로, 추측 대신 여기 명문화한다.

## 5. 체크포인트 / 재개

- discovery+evaluation을 합쳐 **5개 기사 처리할 때마다** `checkpoint.json`을
  임시파일 작성 후 `os.replace()`로 원자적으로 교체 저장한다(중간에 프로세스가 죽어도
  파일이 깨지지 않음). 저장 내용: `completed_article_ids`, `claims_out`, `candidates_out`,
  `mappings_out`, `verification_pairs`, `discovery_mappings`, `current_phase`
  (`discovery`|`evaluation`|`done`), `last_updated`(UTC ISO).
- 같은 체크포인트 시점에 `claims.jsonl`/`candidates.jsonl`/`verified_mappings.jsonl`도
  전체 스냅샷으로 다시 쓰고(원자적 교체), `run_status.json`도 갱신한다(phase, 처리 건수,
  이번 프로세스 인스턴스 기준 건당 평균 처리시간·ETA).
- 재시작 시 `checkpoint.json`이 있으면 그 상태를 그대로 로드하고,
  `completed_article_ids`에 없는 기사부터 이어서 처리한다(같은 claim_id 중복 저장 없음 —
  기사 단위로 완료 여부를 추적하므로 부분적으로 claim이 저장된 기사가 다시 처리될 일도
  없다: 한 기사의 모든 claim을 처리한 뒤에만 그 기사를 completed에 추가).
- **phase 경계 보존**: discovery 단계가 전부 끝나야(`discovery_articles` 전원이
  `completed_article_ids`에 포함) `current_phase`가 `"evaluation"`으로 전환된다. 즉
  discovery 재개 도중 죽었다 재시작해도 evaluation이 먼저 시작되는 일은 없다.
- 819건 전체(discovery+evaluation)를 처리하는 claim extraction 단계도 "이미 완료된 기사는
  제외한 나머지"만 매 (재)시작 시 다시 계산한다(전체를 캐시하지 않음 — extraction은
  GPU가 아니라 HCX API만 쓰므로 GPU OOM 크래시의 원인이 아니고, 남은 기사만 다시
  추출하는 비용이 크지 않기 때문. 반면 retrieval/judge 단계 결과는 claim 단위로 이미
  `claims_out`/`candidates_out`에 영구 저장되므로 재추출로 인해 이미 완료된 claim이 다시
  계산되는 일은 없다).

## 6. 스모크 테스트

`RUN819_SMOKE=1` (+ 선택적으로 `RUN819_SMOKE_DISCOVERY`, `RUN819_SMOKE_EVAL`, 기본
discovery 2건/evaluation 1건)로 실행하면 `checkpoint.json`/산출물 전부 `smoke_` 접두사
경로에 별도로 쓰여 본 실행 상태와 절대 섞이지 않는다. 스모크 성공 확인 후
`smoke_*` 파일은 정리(삭제)한다.

## 7. 절대 원칙 (20건 pilot과 동일, 재확인)

production DB(`SUPABASE_DB_URL`)는 SELECT만. production 코드 파일(`agent/` 전체)은
import만 하고 수정하지 않는다. `agent/mapping/table_catalog.json`,
`agent/kosis/table_params.json`은 절대 수정/자동 승격하지 않는다. retrieval Top-1을 자동
정답으로 간주하지 않는다 — KOSIS 실측치 재조회 + `judge()` 통과해야만 HIGH. baseline이
이미 HIGH면 mapping이 절대 덮어쓰지 않는다(pilot20_run.py에 이미 반영된 규칙, 이 파일은
이를 재사용할 뿐 변경하지 않는다). 특정 table_id/claim/기사 하드코딩, 새 synonym 사전,
모델 파인튜닝 없음.
