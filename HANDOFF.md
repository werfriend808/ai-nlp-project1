# KOSIS 재임베딩·검색 개선 인수인계 (2026-08-26 ~ 08-28)

브랜치 `feature/kosis-reembedding`.
서버 7-1(172.31.26.58, GPU — 이 문서를 쓴 서버) / 7-2(172.31.25.70) / Colab A100.

> **2026-08-28 갱신.** 검색 실험 결과를 운영 코드에 반영했다(문맥 보강 + BM25 교체).
> 그 과정에서 **운영에서 죽어 있던 필드 3개**를 발견했다 — 6장 참고. 이게 지금까지의
> 여러 측정 전제를 무너뜨리므로 먼저 읽을 것.
>
> 7-2는 SSH 키가 이 서버에 없어 접속 불가, PostgreSQL도 안 떠 있다(5432 refused).
> "서버 2대 병렬"은 아직 시작 전이다.

---

## 1. 지금 DB에 뭐가 있나

PostgreSQL 16 + pgvector 0.8.0, `172.31.26.58:5432/kosis_db`.
데이터 디렉터리 `/home/ubuntu/pgdata_ebs` → **EBS 루트 볼륨 `vol-03af1131e580d6b72`**.

| 테이블 | 행 수 | 임베딩 | HNSW 인덱스 |
|---|--:|---|---|
| `kosis_vdb_tables_qwen` | 287,498 | 전건 (2560d) | `..._hnsw` 2,155 MB |
| `kosis_vdb_items_qwen` | 671,145 | 전건 | `..._hnsw` 4,179 MB |
| `kosis_vdb_axes_qwen` | 508,359 | 전건 | `..._hnsw` 2,732 MB |
| `kosis_vdb_axis_values_qwen` | 10,027,486 | 없음 (어휘 매칭용) | trgm GIN |

표 상태: success 279,526 / excluded_too_large 7,483 / error_other 360 / error_no_data 129.

- 모델 `Qwen/Qwen3-Embedding-4B`, `truncate_dim=2560`, `normalize_embeddings=True`
- 임베딩 텍스트 포맷: `item_axis_value_capped` (기관명 / 통계표명 / 항목 / 분류축 / 분류값 50개 cap)
  → 규칙 원본은 `agent/kosis/embedding_text.py`
- HNSW는 `(embedding::halfvec(2560)) halfvec_cosine_ops`, `m=16, ef_construction=64`.
  **질의도 반드시 `embedding::halfvec(2560) <=> %s::halfvec(2560)` 형태여야 인덱스를 탄다.**
  `::vector`로 쓰면 전수 스캔(5.2초)으로 떨어진다.

### 되돌릴 수 있는 백업
- `kosis_english_name_backup` (2,832행) — 영문 표명 한글화 이전의 table_name / embedding_text / embedding 원본.

---

## 2. 이번에 한 일 (커밋 순)

| 커밋 | 내용 |
|---|---|
| `21dfe73` | 28.7만 표 재임베딩 워커·스키마·검증 스크립트 |
| `90ffa70` | 전체 재구축·복구 인프라 (v2 워커, 3단계 복구 폴백, 품질 게이트, 백업/터널) |
| `6c6600f` | 영문 표명 2,824건 한글화 + 해당 표만 재임베딩 |
| `4ae5127` | 운영 검색 경로를 재구축 DB에 맞춤 (2560차원, halfvec, ef_search 자동) |
| `26e7418` | 검색 전략 비교 실험 (Recall@100 57.9% → 77.1%) |
| *(미커밋)* | **문맥 보강 배관 + trigram→BM25 교체** — 아래 2-1 |

### 2-1. 2026-08-28 작업 (미커밋 상태)

**(1) 문맥 보강 배관** — claim 앞 문장이 검색 질의까지 전달되게 함.
- `agent/interfaces.py` — `Claim`에 `context_before` / `sentence_index` 추가
- `agent/preprocessing/claim_extractor.py` — `attach_sentence_context()` 신설.
  `extract_claims()`/`recover_missed_claims()`가 정제 본문을 들고 있는 그 자리에서 채운다
  (사후 문자열 매칭은 dedupe 때문에 어긋나므로 추출 시점이 유일하게 정확한 지점).
- `agent/mapping/reranker.py` — `build_retrieval_query()` 신설(dense용 질의 조립 단일화)
- `batch_runner.py` / `rerank_local.py` / `api/server.py` — 이 함수를 쓰도록 교체

검증: 골든셋 70건에서 실험 `ctx_D2`와 질의가 **70/70 글자 단위 일치**. dedupe 적용 후에도
위치 탐색률 동일(64/70). `KOSIS_QUERY_CONTEXT=0`이면 도입 이전과 완전히 같아진다.

**(2) trigram → BM25 교체** — 어휘 검색기 자리를 갈아끼움.
- `agent/kosis/bm25_search.py` (신규) — 디스크 캐시 희소 인덱스, `bm25_query_vdb()`
- `agent/kosis/build_bm25_index.py` (신규) — 인덱스 생성기
- `agent/mapping/reranker.py` — `build_lexical_query()` 신설(BM25 전용 짧은 구조화 질의)
- `batch_runner.py` / `api/server.py` — `lexical_query_vdb` → `bm25_query_vdb`

`query_vdb.lexical_query_vdb`(trigram)는 노트북·벤치마크가 아직 임포트하므로 **남겨뒀다**.
운영 경로에서만 빠졌다.

### 운영 코드 변경분 (`4ae5127`)
- `agent/kosis/query_vdb.py` — 테이블명 `kosis_vdb_tables_qwen`, halfvec 질의,
  커넥션 생성 시 `hnsw.ef_search` 자동 적용(`HNSW_EF_SEARCH=100`)
- `agent/pipeline/rerank_local.py` / `batch_runner.py` / `agent/api/server.py` — 2560차원
- `agent/kosis/enrich_objl.py`, `build_vdb_index.py` — `[DEPRECATED]` 표기

---

## 3. 검색 실험 결론 (`benchmark/search_experiment2/REPORT.md`)

평가셋: 골든셋 70건 (정답표가 있는 claim). **표본이 작아 1~2%p 차이는 무의미.**

| 전략 | R@1 | R@10 | R@100 | R@200 |
|---|--:|--:|--:|--:|
| Baseline (claim 문장 → 표 dense) | 5.7% | 32.1% | 57.9% | 63.6% |
| **Context D2 (claim + 이전 문장)** | 7.1% | **47.9%** | **75.0%** | 76.4% |
| COMBO-A (dense+BM25+ctx) | 8.6% | 50.7% | 76.4% | 78.6% |
| COMBO-C (+struct+expansion) | 10.0% | 52.1% | **77.1%** | 80.0% |

**검색기 22종 top-100 합집합 상한선 = 94.3%.** 융합만으로는 이 위로 못 간다.

### 확정된 사실
1. **문맥 보강이 최대 기여.** claim 앞 문장 한 줄만 붙여도 Recall@100 +17.1%p,
   손해 보는 claim 0건. 검색기 추가도 융합도 필요 없다.
2. **BM25가 trigram을 완전히 대체.** hit@100 31.4% vs 0%, 지연 2ms vs 6,893ms.
   기존 Hybrid의 8.7초 지연은 전부 trigram 탓이었다. 인덱스는 운영 DB가 아니라
   Python 희소행렬(`benchmark/search_experiment2/bm25_index/`, 336MB, 빌드 2분).
3. **표 임베딩 포맷은 현행 유지 확정.** item 제외(E6) 시 −7.1%p로 가설 반증.
   기여도 순: 기관명 +15.7%p, 분류값 +20.0%p, 분류축 +9.3%p, 항목 +7.1%p.
4. **검색 개선은 최종 top-1까지 전달된다.** 리랭커 통과 후 R@1 5.7% → 14.3%.
   리랭커는 후보를 버리지 않는다(순열, 70/70건 확인) — R@100은 전후 동일.
5. **리랭커 절대 성능은 낮다.** 후보 100개 안에 정답이 있는 53건 중 10건(18.9%)만
   1등으로 올린다. 개선 여지는 검색(19%p)보다 리랭커(56%p)가 크지만 비용도 크다.

### 실패한 가설 (다시 시도하지 말 것)
- **Item → Axis 계층 검색**: R@100 39.3%로 Baseline보다 15.7%p 낮음. item 임베딩이
  "취업자" 같은 짧은 단어라 변별력이 없다. 앞단에서 정답이 탈락하면 복구 불가.
- **item 필드 제외**: 위 3번. 짧은 텍스트(23자)에서의 −1.4%p는 311자에서 예측력이 없었다.

### 주의 — 신뢰할 수 없는 수치
- **`REPORT.md`의 dense 계열 지연은 무효.** 검색기를 순차 실행해 먼저 돈 것일수록
  HNSW 인덱스가 콜드였다(실행 순서대로 378ms → 15ms 단조 감소).
  워밍 후 재측정: Baseline 94ms vs Context D2 83ms (인코딩 포함). **"7배 빠름"은 오류.**
  dense 검색 지연은 DB(중앙 4ms)가 아니라 **임베딩 인코딩(43~46ms)이 지배**한다.
  재측정 스크립트: `benchmark/search_experiment2/latency_ab.py`
- **dev/test 분리 실패.** 같은 검색기가 dev 97.1% / test 52.8%로 갈렸다. 기사별 난이도
  편차 때문이며 70건으로는 유의미한 분할이 불가능. 모든 수치는 전체 70건 기준.

---

## 4. 다음에 할 일 (우선순위)

> 1·2순위였던 문맥 보강과 BM25 교체는 2026-08-28에 완료했다(2-1장). 아래는 그 이후.

### 1순위 — 죽은 필드 3개 복구
6장 참고. `age`/`gender`/`search_query`가 운영에서 한 번도 채워진 적이 없다.
이게 살아나야 (a) BM25 질의를 지금의 슬롯 조합 대신 설계 의도대로 줄 수 있고,
(b) 층2의 성별 가점이 처음으로 동작하며, (c) "문맥을 문장에 붙일지 요약어에 붙일지"를
비로소 실측으로 답할 수 있다.

### 2순위 — 골든셋 확대
70건으로는 dev/test 분리가 불가능하고 1~2%p 판단도 못 한다. 리랭커 개선의 선행 조건이다
(리랭커의 개선폭은 검색보다 크지만, 70건으로는 개선 여부를 판별할 수 없다).

### 3순위 — 리랭커
`BAAI/bge-reranker-v2-m3`. 후보 내 정답의 81%를 1등으로 못 올린다.
`_rrf_fuse`가 검색 순위와 리랭커 순위를 50:50으로 섞는데, 실측상 이 구조 자체는
문제가 없었다(두 신호가 보강 작용). 모델 쪽 문제.

### 상시 과제 — 골든셋 확대
70건으로는 dev/test 분리가 불가능하고 1~2%p 판단도 못 한다. 이게 가장 큰 제약.

### BM25 운영 요건 (교체하면서 새로 생긴 것)
- **인덱스가 DB에 없다.** `data/bm25_index/`(319MB, gitignore됨). 표가 추가·변경되면
  `python -m agent.kosis.build_bm25_index`로 다시 만들어야 한다(28.7만 표 기준 약 2분).
  **재빌드를 안 하면 새 표가 어휘 검색에 영원히 안 잡힌다** — 자동화 안 돼 있음.
- **메모리 +1.6GB.** 첫 질의 때 lazy 로딩된다(로딩 5.9초). API 서버는 Qwen 임베딩·리랭커와
  같은 프로세스라 이 몫을 감안해야 한다(현재 서버 RAM 15GB).
- 인덱스가 없으면 `VdbUnavailableError`를 올리고 호출부가 잡아서 dense/keyword만으로
  계속 진행한다(trigram 시절과 동일한 실패 처리).

### 미완료
- **excluded_too_large 7,483건 복구 중단 상태.** 국가데이터처(org 101) 4,011건이
  남아 있었고 7-1/7-2로 반씩 나눠 돌리다 멈췄다. 재개하려면
  `agent/kosis/recover_excluded.py --ids-file <목록> --concurrency 3 --api-keys "..."`.
  성공률 약 83%, 시간당 213표.
- **EBS 스냅샷 못 찍음.** 인스턴스에 AWS 자격증명(IAM 역할·`~/.aws`·env) 이 전혀 없고
  사용자도 학원 계정이라 콘솔 접근 불가. 인스턴스 컨트롤 패널에는 시작/중지/재시작만
  있고 terminate 버튼이 없어 실수로 볼륨이 삭제될 경로는 없다.

---

## 5. 환경 메모

- **인스턴스 스토어 주의**: `/home/ubuntu/data`(`nvme1n1`, 232GB)는 인스턴스 스토어라
  **중지(stop)만 해도 초기화된다.** 현재 HF 모델 캐시 7.6GB만 들어 있어 재다운로드로
  끝나지만, DB나 산출물을 절대 여기 두지 말 것. DB는 EBS 위에 있다(확인 완료).
- **HF Hub 오프라인**: 네트워크가 막히면 모델 로딩이 443 연결에서 멈춘다.
  `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` 로 우회.
- **pgvector 함정**: `hnsw.ef_search`(기본 40, 상한 1000)보다 많은 행을 반환하지 않는다.
  `LIMIT`을 크게 주려면 `ef_search`를 먼저 올려야 한다. 실험 1차를 이것 때문에 폐기했다.
- **플래너 함정**: `kosis_vdb_axes_qwen`에서 플래너가 병렬 seq scan을 골라 질의당 40초가
  걸렸다. `set enable_seqscan = off`로 해결(0.9초).
- **pkill 주의**: `pkill -f run_experiment.py` 같은 패턴은 자기 셸 명령줄까지 매칭해
  스스로를 죽인다. 패턴을 좁힐 것.

---

## 6. [2026-08-28] 운영에서 죽어 있던 필드 3개 — 먼저 읽을 것

**`age` / `gender` / `search_query`가 한 번도 채워진 적이 없다.**

HCX-DASH-002가 `claim_extractor_prompt.txt`의 출력 스키마에서 **마지막 세 필드를 아예
출력하지 않는다.** null이 아니라 키 자체가 없다. 프롬프트에는 20줄에 걸쳐 정의돼 있는데도
모델이 `source_report`까지만 쓰고 객체를 닫는다. 골든셋 기사로 교차 확인:

| | age | gender | search_query | source_org |
|---|--:|--:|--:|--:|
| 기사 11 (claim 6건) | 0/6 | 0/6 | **0/6** | 6/6 |
| 골든셋 매칭 27건 | — | — | **0/27** | 4/27 |

`source_org`·`statistic_expression`은 정상이다. 스키마 뒤쪽 세 개만 잘린다.

### 이것이 무너뜨리는 전제들

1. **운영은 `search_query`로 검색한 적이 없다.** `base = claim.search_query or claim.sentence`가
   항상 문장 원문으로 폴백했다. 즉 운영의 실제 baseline은 실험의 `dense_full`(58.6%)이었다.
2. **`reranker.py`에 기록된 "trigram Recall@30 2.4%→41.5%(짧은 search_query 기준)"의
   전제가 성립한 적이 없다.** trigram에는 계속 긴 문장이 들어갔고, 골든셋 재측정 결과
   **운영 질의 기준 hit@100이 정확히 0.0%**였다(어제 실험의 4.3%는 구조화 질의 기준).
3. **층2의 `_apply_gender_signal`은 한 번도 발동한 적이 없다.** `claim.gender`가 항상
   None이라 함수 첫 줄에서 즉시 반환한다. `age` 가점도 마찬가지.

### 고칠 때 참고
프롬프트 출력 스키마에서 세 필드를 앞쪽으로 옮기거나, 별도 호출로 분리하거나, 모델을
바꾸는 방향. 고친 뒤에는 `build_lexical_query()`의 첫 줄이 이미 `claim.search_query`를
우선 쓰도록 준비돼 있으므로 질의는 자동으로 설계 의도대로 돌아간다.

---

## 7. [2026-08-28] 반영 후 실측치

골든셋 70건. dense는 문맥 보강 적용(`ctx_D2`), 융합은 RRF(k=60).

### 어휘 검색기 단독 — 운영이 실제로 보내던 질의 기준

| | hit@10 | hit@30 | hit@100 | 평균 지연 |
|---|--:|--:|--:|--:|
| trigram (교체 전) | 0.0% | 0.0% | **0.0%** | 7,667ms |
| BM25 (교체 후) | 20.0% | 28.6% | **47.1%** | 2~10ms |

### dense와 융합했을 때

| | R@10 | R@100 | R@200 |
|---|--:|--:|--:|
| dense 단독 | 48.6% | 75.7% | 77.1% |
| dense + trigram *(교체 전)* | **31.4%** | 75.7% | 77.1% |
| dense + BM25 *(교체 후)* | 48.6% | **78.6%** | **90.0%** |

**trigram은 이득이 없는 정도가 아니라 해가 되고 있었다.** 무관한 후보 30개를 RRF에
밀어넣어 dense 단독보다 R@10을 17.2%p 끌어내렸다. 교체로 그 피해가 사라지고
R@200이 77.1% → 90.0%가 됐다. 손해 본 claim 1건(`30-09`), 순증 +2.

### 문맥 보강 단독 효과 (dense)
Recall@100 **58.6% → 75.7%**, 손해 보는 claim 0건(+12/−0). 어제 실험의 +17.1%p가
운영 조건에서 그대로 재현됐다 — 위 6장 1번 때문에 운영 baseline이 실험 baseline과
같았기 때문이다.

### dense와 lexical은 원하는 질의가 정반대다
`build_retrieval_query()`(dense)와 `build_lexical_query()`(BM25)를 나눠 놓은 근거:

| dense | | BM25 | |
|---|--:|---|--:|
| 문맥+문장 | 75.0% | 구조화+확장 | 47.1% |
| 문장 전체 | 57.9% | 구조화 | 32.1% |
| | | 문장 전체 | 17.9% |
| | | 문맥+문장 | **12.1%** |

**BM25에 문맥을 붙이면 오히려 나빠진다.** 두 검색기에 같은 질의를 주면 안 된다.

### 측정 스크립트
- `benchmark/search_experiment2/ctx_prod_ab.py` — 문맥 보강 A/B(4개 조합)
- `benchmark/search_experiment2/bm25_swap_ab.py` — trigram/BM25 교체 전후
- `benchmark/search_experiment2/build_golden_search_query.py` — 골든셋 `search_query`
  생성 시도. **6장 때문에 0/70으로 실패한다** — 필드가 복구되면 그때 다시 쓸 것.
