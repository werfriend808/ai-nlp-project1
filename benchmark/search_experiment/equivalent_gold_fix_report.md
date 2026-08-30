# 골든셋 "동등 표" 채점 오류 수정 — 결과 보고

C 트랙 인수인계 문서 "06 다음에 할 일 · 1. 동등 표를 정답으로 인정하도록 채점 고치기"(미착수) 작업.

작성 2026-08-30 · 브랜치 `fix/golden-eval-equivalent-tables` (base: `feature/kosis-reembedding`)

## 문제

채점이 "표 ID가 정확히 일치"만 정답으로 쳤는데, 실제로는 같은 통계가 tblId를 갈아치우며
재발행되는 경우(예: `DT_1DA7E06S` 구판 vs `DT_1DA7E06S_NEW` 신판)가 있어서, claim이
요구하는 기간을 둘 다 커버하면 어느 쪽을 찾아도 정답이어야 하는데 하나만 정답으로 인정되고
있었다.

## 한 일

### 1) 이미 검증된 수정 동기화 (수동)

8/26 `71a28c5`(브랜치 `origin/experiment/pipeline-redesign`, 현재 팀이 실제 쓰는
`feature/kosis-reembedding`엔 병합 안 돼 있음)에서 신영님이 `DT_1DA7E06S` 관련 골든셋
라벨을 이미 검증·수정했었다(구판 라벨인데 실제로는 2025년 데이터가 필요해 신판이어야
하는 6~7건). 그런데 이 수정은 `골든셋_통합.xlsx`에만 반영됐고, C 트랙이 실제로 쓰는
`benchmark/search_experiment/eval_set.json`(8/27 생성, xlsx 수정 다음 날인데도 반영 안 됨)엔
빠져 있었다 — 두 골든셋 파일 사이 동기화 누락.

`eval_set.json`에서 `gold: ["DT_1DA7E06S"]`였던 7건(`5-02a`, `5-02b`, `5-03`, `5-04`,
`5-05a`, `5-05b`, `5-11` — 전부 `period: "2025-06"`)을 `["DT_1DA7E06S_NEW"]`로 교체
(add 아니라 replace — 구판 자체가 그 기간 데이터를 안 갖고 있어서 구판은 애초에 답이 될
수 없는 케이스이기 때문). `table_catalog.json`(수동 64개 카탈로그)에도 구판은 이미
빠지고 신판만 등재돼 있어 이 판단과 일치.

### 2) 일반 규칙 스크립트 (자동)

`benchmark/search_experiment/expand_equivalent_gold.py` 추가. 기존 6개 실험
스크립트(`fuse.py`/`reranker_model_ab.py`/`rrf_weight_ab.py`/`gate_ab.py`/`title_ab.py`/
`ctx_prod_ab.py`)는 전부 `gold = {claim_id: set(r["gold"])}` + `t in gold[c]` 패턴으로
채점하므로(gold가 원래 리스트, 정답 2개인 claim도 이미 존재 — `19-09`), 채점 로직 자체는
안 건드리고 `eval_set.json`의 `gold` 리스트에 동등 표를 미리 채워 넣는 방식으로 풀었다.

- gold 표들의 `(stat_id, table_name)`을 DB(`kosis_vdb_tables_qwen`)에서 조회해 같은 조합을
  가진 "형제 표"를 찾는다(stat_id 단독은 같은 설문조사 전체가 묶여 너무 굵어서
  `agent/kosis/version_meta.py`의 실측 근거를 따라 조합으로 판정).
- 형제 표의 `period_start`~`period_end`가 claim의 필요 기간을 포함하면 `gold`에 **추가**
  (기존 값 유지 — replace 아님, "이 기간엔 둘 다 정답"이라는 뜻이므로).
- 기간 파싱기는 `eval_set.json`의 실제 period 값 16종으로 검증 — datetime 문자열, 점
  표기, 분기, 콤마 나열, 괄호 안 연도까지 포함해 15종 인식(1종 "2024년 말"은 정확한 달을
  특정 못 해 의도적으로 건너뜀 — fail-open보다 fail-closed가 안전).
- 기본은 `--dry-run`(미리보기만), `--apply`로 실제 반영.

AWS 서버(7-1)에서 `--apply` 실행 결과: **`35-04a`, `35-04b`에 `DT_1DA7E26S_NEW` 추가**
(형제 표: `DT_1DA7E26S`, 기간 2025-04 포함).

## 검증 — 재측정 결과

`reranker_model_ab.py --model none/--model BAAI/bge-reranker-v2-m3` 재실행(캐시된 후보
풀 재사용, 검색은 다시 안 함) 후 `--report`.

| | 고치기 전 (C 트랙 문서 수치) | 고친 후 |
|---|--:|--:|
| 리랭커 있음(현재) top-1 | 25.7% | **31.4%** |
| 리랭커 있음 MRR | 0.329 | **0.367** |
| 리랭커 없음 top-1 | 11.4% | **15.7%** |
| 리랭커 없음 MRR | 0.167 | **0.224** |
| 후보 풀에 정답 있는 claim(리랭커 상한) | 43/70 (61.4%) | **45/70 (64.3%)** |

전부 개선 방향으로 움직였다 — 채점 오류가 실제로 점수를 깎아먹고 있었다는 게 수치로
확인됨.

## 참고 / 남은 것

- **`reranker_model_results/`에 이전 실험(리랭커 질의 형태 비교, ①②)의 결과 파일이
  그대로 남아있어서 `--report`가 관련 없는 옛 행(`리랭커질의=sentence` 등, 옛 정답지
  기준)까지 같이 보여준다** — 헷갈리니 정리(삭제 또는 다른 폴더로 이동) 권장.
- 한국어 특화 등 나머지 3개 리랭커 모델 비교는 재실행 안 함 — 정답지를 고쳐도 "다른
  모델이 현재 모델보다 낫다"는 결론이 뒤집힐 가능성은 낮다고 판단해 우선순위를 낮춤.
  필요하면 이어서 진행 가능.
- 이 브랜치(`fix/golden-eval-equivalent-tables`)는 `feature/kosis-reembedding`에서 갈라져
  나왔고 아직 그쪽으로 병합 안 됨 — PR 필요.
- 신영님의 리랭커 코드 수정(`_apply_version_freshness_signal`, `agent/kosis/version_meta.py`)
  자체는 여전히 `origin/experiment/pipeline-redesign`에만 있고 `feature/kosis-reembedding`엔
  없음 — 이건 이 작업 범위 밖이지만 팀 공유가 필요해 보임.

## 관련 파일

- `benchmark/search_experiment/eval_set.json` — 골든셋 본체(수정 대상)
- `benchmark/search_experiment/expand_equivalent_gold.py` — 자동 확장 스크립트
- 커밋: `b9f8353`(수동 동기화 7건), `5efc0d0`(스크립트 추가), `d673454`(서버 실행 결과 2건 적용)
