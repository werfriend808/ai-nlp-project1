# Verified Mapping 819건 실행 — summary.md

## 1. 실행 개요
- 대상: `data/article_priority_institution_mentioned.json` 819건 전체 (공공기관 언급 기사 우선순위 셋)
- 실행 시각: 2026-08-30 16:35:04 ~ 2026-08-31 03:49:47 UTC, 총 40,467.5초(약 11시간 14분)
- Discovery 573건 / Evaluation 246건, 70/30 deterministic split (seed=20260830819)
- 재시작 이력: **0건** (GPU-aware 자동재시작 래퍼 `run819_autorestart.sh` 대기만 했고 개입 없음 — 프로세스가 한 번도 죽지 않음)
- 체크포인트: 5건 단위, 최종 `checkpoint.json`(263MB)까지 정상 누적 — 중단 시 재개 가능한 상태 유지 확인됨

## 2. 회귀 테스트 (사전 실행분, 819건 실행 전 완료)
`benchmark/verified_mapping_experiment/regression_test.py` — **PASS** (보호 시나리오 8/8, 구제 경로 대조 시나리오 4/4). row15-c0류 버그("baseline이 이미 HIGH인데 mapping 후보가 재검증에서도 HIGH를 받아 정답을 덮어씀") 재발 방지 규칙을 `pilot20_run.py::process_evaluation_claim()`에 반영하고 검증 완료 후 본 실행 진행.

## 3. Legacy 카탈로그 보존 확인
`agent/mapping/table_catalog.json`, `agent/kosis/table_params.json`(64개 수동 카탈로그) — mtime이 이번 세션 시작 전(2026-08-24)과 동일, **819건 실행 중 전혀 수정되지 않았고 verified_mappings로 자동 승격되지도 않았다.**

## 4. Claim 추출/분류
- 전체 claim 추출: **2,464건**
- extended_claim_type 분포: numeric 1,146 / comparative 895 / superlative 93 / trend 105 / threshold 87 / qualitative 138
- confidence 분포: **HIGH 355(14.4%) / MEDIUM 1,178(47.8%) / LOW 793(32.2%) / UNKNOWN 138(5.6%)**
- abstention rate(UNKNOWN 비율): 5.6%

## 5. Discovery → Verified Mapping 축적
- Discovery 573건에서 HIGH mapping **255건** 생성 (discovery claim 대비 약 14.3%)
- mapping 재사용 시도(evaluation 단계에서 mapping_hit 존재): 569/681건(83.6%) — discovery에서 만든 mapping이 evaluation claim의 후보 풀에 폭넓게 다시 등장함

## 6. Evaluation 단계 A(Baseline)/B(Mapping-only)/C(Mapping-assisted)
- evaluation claim 총 681건 중 **gold(가부 판정 가능한 정답) 확보: 135건(19.8%)**, no_gold 546건(80.2%) — 나머지는 baseline도 mapping도 HIGH를 못 얻어 애초에 recall 채점 대상이 아님(정직하게 보고: 이건 mapping의 한계라기보다 애초에 KOSIS로 확정 검증 가능한 claim 자체가 소수라는 이 프로젝트 전반의 알려진 특성)
- recall_A(baseline, R@1/10/50/100): **0.741 / 0.882 / 0.993 / 1.000**
- recall_B(mapping-only), recall_C(mapping-assisted): **전 구간 1.000** — ⚠️ **주의: 이 수치는 방법론상 계산이 순환적(tautological)이다.** gold_table_id 자체가 "baseline이 HIGH였던 표 또는 mapping이 rescue한 표" 중 하나로 정의되고, assisted_rank 계산이 그 표를 항상 1위로 끌어올리는 방식이라 gold가 존재하는 claim에서는 구조상 항상 rank1이 나온다(pilot20 pilot 때부터 있던 known 한계, 이번에 819 규모에서도 동일 확인). **"100% recall 달성"으로 읽으면 안 되고, "gold가 있는 곳에선 설계상 항상 맞다"는 자기참조적 지표로만 해석해야 한다.**
- evaluation_outcome 분포:
  - `mapping_skipped_baseline_protected` **52건** — baseline이 이미 HIGH라 mapping 재검증 자체를 스킵(보호 규칙 실동작 확인, 이번 실행에서 확실히 작동했다는 직접 증거)
  - `mapping_rescued_claim_improvement` **35건** — baseline이 놓친 걸 mapping이 실제로 구제(진짜 순기능)
  - `mapping_rejected_conflict` **282건(49.6%, mapping_hit 있는 것 중 최대 비중)** — mapping 후보가 있었지만 재검증에서 탈락(불일치/판단불가) → 안전하게 폐기됨
  - `mapping_confirms_baseline` 200건 — mapping이 baseline과 같은 표를 가리켜 추가 정보 없음
  - `no_mapping_applicable` 112건 — 애초에 후보 풀에 mapping 표가 없었음
- **개선 claim 수 35 vs 확인된 regression 0건**(보호 규칙이 100% 작동, mapping이 baseline의 정답을 덮어쓴 사례는 0건)

## 7. 비용
- HCX 호출: 7,088회
- KOSIS API 호출: 2,634회
- 총 소요: 40,467.5초 (819건, article당 평균 49.4초)

## 8. 안정성/인프라
- GPU-aware 자동재시작 래퍼(`run819_autorestart.sh`, `backup/recover_org101_autorestart.sh` 패턴 재사용) 정상 대기, 개입 0회
- 체크포인트 5건 단위 정상 기록(87회+ 갱신 로그 확인), 프로세스 생존 기간 내내 유실 없음

## 9. Q1~Q8 답변
1. **claim 추출 수**: 2,464건
2. **검증 가능(verifiable) claim 수**: 2,464 − 138(UNKNOWN 중 claim_type=전망/해외 등 애초 제외분 포함) ≈ 검증 시도된 claim 다수, 정확히는 confidence 분포의 HIGH+MEDIUM+LOW=2,326건이 실제 KOSIS 조회/판정까지 도달
3. **HIGH 건수**: 355건(14.4%)
4. **HIGH mapping 정밀도**: discovery 255건 중 evaluation에서 재사용 시도된 것(mapping_hit 존재) 569건, 그중 재검증 통과(confirms+rescued) 235건 vs 거부(rejected_conflict) 282건 → **재사용 시도 대비 통과율 약 41.3%**(나머지는 안전하게 폐기, 오탐이 실제 오답으로 이어진 사례는 0건)
5. **mapping으로 구제된 claim 수**: 35건
6. **mapping으로 인한 regression**: **0건**(보호 규칙 52건 정상 작동 확인)
7. **일반화 가능성**: discovery에서 만든 mapping이 evaluation claim 후보 풀에 83.6% 재등장 — topic 재사용성은 높으나, 재검증 통과율이 41.3%로 "무조건 맞는 지름길"은 아니고 "괜찮은 후보 힌트 + 항상 재검증" 설계 그대로가 맞다는 게 실측으로 재확인됨
8. **2,500건 전체 확장 가치**: 있음, 단 조건부. 안전성(0 regression)은 이미 충분히 입증됐고, 순이익(35건/681건=5.1%)은 작지만 확실히 양(+)이다. 다만 recall_B/C=100% 수치의 순환성 문제 때문에 확장 전 "진짜 외부 기준 gold 대비 개선폭"을 재는 별도 평가 설계가 필요하다(현재 gold 정의 자체를 baseline/mapping 성공 여부로 순환 정의하는 방식은 확장판에서도 그대로 쓰면 안 됨).

## 10. 최종 판정: **PROMISING**

메커니즘은 안전하고(0 regression, 보호 규칙 실동작 확인) 순기능도 실측으로 확인됐지만(35건 구제), 효과 규모가 크지 않고(전체 evaluation claim의 5.1%) 재검증 통과율도 41.3%로 낮아 비용 대비 효율은 제한적이다. recall_B/C 지표의 순환성 문제도 있어 "100% 개선"으로 과대 해석해서는 안 된다. ADOPT(production 즉시 반영)로 가기엔 이득이 아직 작고, REJECT/NEEDS_FIX로 보기엔 안전성과 실효성이 실측으로 확인됐다 — **PROMISING으로 판정하고, production 반영 전 외부 기준 gold 기반 재평가를 권장한다.**
