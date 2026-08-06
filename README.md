# AI 기반 뉴스 사실검증 시스템

뉴스 기사에서 수치 기반 주장을 자동으로 탐지하고, KOSIS(국가데이터처) 공식 통계와 비교해 **일치 / 불일치 / 판단불가**로 판정한 뒤 근거를 설명까지 생성하는 End-to-End 검증 파이프라인입니다.

멋사 NLP 5기 종합 프로젝트(클라비 기업 연계)로 진행 중이며, 실전1(주장 추출·EDA) → 실전2(대화형 통계 조회 에이전트) → 종합(전체 자동화)의 누적 구조 중 종합 단계에 해당합니다.

## 프로젝트 배경

뉴스 및 온라인 콘텐츠에서 왜곡된 통계 인용, 맥락 없는 수치 사용, 시점·단위 혼동 등이 반복적으로 발생하지만 현재는 수작업 검증 비중이 높습니다. 본 프로젝트는 공공데이터(KOSIS Open API)를 기준으로 뉴스의 수치 주장을 자동 검증하는 PoC를 구축하는 것을 목표로 합니다.

## 검증 프로세스 (8단계)

```
기사 원문
  → [1] 분류(classifier)          관련 기사인지(수치 기반 주장 포함 여부) 판별
  → [2] 주장 추출(claim_extractor) 검증 가능한 수치 주장 문장을 구조화해서 추출
  → [3] 통계표 매핑              keyword_search + embedding_search + reranker로 대응 KOSIS 표 탐색
  → [4] 슬롯필링/되묻기           시점·지역·계산종류 등 조회에 필요한 값 채우기
  → [5] KOSIS API 조회            공식 통계 데이터 실제 호출
  → [6] 계산                     합계/비율/증감/증감률/최댓값·최솟값검증 등 코드 연산
  → [7] 판정                     기사 수치 vs 계산값 비교 → 일치/불일치/판단불가
  → [8] 설명 생성                 근거·계산방식·판정이유·한계를 포함한 설명문 생성
```

판정 결과 기준:

| 판정 | 조건 |
|---|---|
| 일치 | 기사 수치가 공식 통계와 동일하거나 시점·단위가 일치 |
| 불일치 | 기사 수치가 공식 데이터와 다르거나 시점·단위·모집단 해석이 잘못됨 |
| 판단불가 | 지표·시점이 불명확하거나 대응하는 공식 데이터가 없음 |

## 폴더 구조

```
agent/
  preprocessing/   1~2단계 — classifier.py, claim_extractor.py, hcx_client.py
  mapping/         3단계   — keyword_search.py, embedding_search.py, reranker.py, table_catalog.json
  orchestrator/    4단계   — slot_filler.py, clarify.py, clarify_rules.py, calc_type_router.py
  kosis/           5~6단계 — api_client.py, calculator.py, table_params.json
  verdict/         7단계   — judge.py
  explain/         8단계   — explainer.py
  shared/          여러 단계 공용 — extreme_value_patterns.py (역대/N년만에/코로나이후 등 극값 표현 감지)
  pipeline/        전체 연결 — batch_runner.py(하드코딩 시나리오), csv_batch_runner.py(실제 CSV)
  interfaces.py    팀 공통 계약(단계별 입출력 타입) — 임의 변경 금지, 수정 시 팀 합의 필요
data/              데이터셋(data_set.csv, 조선일보 기사, .gitignore 대상) + verifications.db
db/                store.py — 검증 결과 SQLite 저장소
prompts/           단계별 LLM 프롬프트(few-shot 포함)
docs/              설계 문서, PENDING_TABLES.md(KOSIS 매핑 실패 항목 기록)
tests/             단위 테스트 + 모듈별 검증 로그(*_test_log.md)
notebooks/         모델 비교, 골든셋 관련 노트북/스크립트
```

## 시작하기

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경변수 설정

리포지토리 루트에 `.env` 파일을 만들고 아래 두 키를 채웁니다.

```
KOSIS_API_KEY=...
HCX_API_KEY=...
```

- `HCX_API_KEY`: 1·2·4·7·8단계(LLM 호출)에 필요
- `KOSIS_API_KEY`: 5단계(공식 통계 조회)에 필요

### 3. 실행

전체 파이프라인(하드코딩된 10개 시나리오, 알려진 이슈 재현용):

```bash
python -m agent.pipeline.batch_runner
```

실제 데이터셋(`data/data_set.csv`) 샘플로 실행:

```bash
python -m agent.pipeline.batch_runner --csv
```

1~3단계만 실제 CSV 전체(또는 `--limit`)로 안정성 확인:

```bash
python -m agent.pipeline.csv_batch_runner --limit 25
```

### 4. 테스트

각 모듈은 `python -m tests.<파일명>` 형태로 개별 실행합니다(예: `python -m tests.test_kosis_module`). 실행 조건(API 키 필요 여부)은 각 테스트 파일 상단 docstring에 명시돼 있습니다.

## 현재 진행 상황 (골든셋 기준)

| 단계 | 지표 | 결과 |
|---|---|---|
| 1단계 분류 | 골든셋 정확도 | 95% 이상 |
| 2단계 주장 추출 | recall | 28.8% → 80.8% |
| 3단계 통계표 매핑 | 카탈로그 규모 | 56개 표 |
| 7단계 판정 | 골든셋 정확도 | 64% → 92.3% |
| 8단계 설명 생성 | - | 검증 예정 |

## 알려진 한계

- 개별 단계는 골든셋 기준 성능이 검증됐지만, 전체 단계를 연결해서 실행하면 성능이 저하되는 현상이 있어 단계별로 연결하며 원인을 파악 중입니다.
- KOSIS Open API로 데이터 자체가 열려있지 않아 검증 불가로 결론난 통계 항목이 있습니다(연체율, 한국은행 기준금리 등 — 상세는 `docs/PENDING_TABLES.md` 참고). 원출처가 KOSIS가 아닌 ECOS/FISIS 등인 경우로 추정됩니다.
- 검증자(사람) 리뷰 단계는 DB 스키마에 컬럼만 준비돼 있고 UI/로직은 아직 없습니다.
- End-to-End 서비스/API 서버(FastAPI 등)는 아직 구현되지 않았습니다.
- 데이터셋 수집 기간이 2025년 1~7월에 집중되어 있고 8월은 공백, 9~12월은 매우 적습니다.

## 팀 역할 구분 (파이프라인 단계 기준)

- **A** — 1~2단계 전처리 (classifier, claim_extractor)
- **B** — 3단계 통계표 매핑 (keyword_search, embedding_search, reranker)
- **C** — 5~6단계 KOSIS 연동 (api_client, calculator)
- **D** — 4·7·8단계 오케스트레이션/판정/설명 (slot_filler, clarify, judge, explainer)
