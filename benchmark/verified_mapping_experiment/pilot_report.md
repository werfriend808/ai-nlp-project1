# 50건 Pilot 결과 보고 (DRAFT — 실행 완료 후 수치로 갱신 예정)

이 문서는 실행이 끝나는 대로 `pilot_run_stats.json` + `pilot_claims.jsonl` +
`pilot_candidates.jsonl` + `pilot_verified_mappings.jsonl`의 실제 수치로 채워진다.
아래는 채울 항목의 골격만 미리 잡아둔 것이다.

## 0. 스코프 재확인
이 보고서는 **50건 pilot 결과 + 2,706건 외삽 수치**까지만 다룬다. 2,706건 전체(또는
discovery/validation/evaluation 분할) 실행 여부는 이 보고서가 판단하지 않는다 —
사용자가 아래 수치를 보고 직접 결정한다.

## 1. 추출된 claim 수 / 유형별 분포
(TODO: pilot_run_stats.json.extended_claim_type_distribution,
production_claim_type 분포도 함께)

## 2. HIGH/MEDIUM/LOW·UNKNOWN confidence 분포
(TODO: pilot_run_stats.json.confidence_distribution + mapping_status_distribution)

## 3. KOSIS API 호출 횟수 및 소요시간
(TODO: pilot_run_stats.json.kosis_calls)

## 4. HCX 호출 횟수 및 단계별 소요시간
(TODO: pilot_run_stats.json.hcx_calls, stage_timers_sec)

## 5. 전체 파이프라인 총 소요시간
(TODO: total_elapsed_sec, 단계별 분해)

## 6. 2,706건 전체 외삽
(TODO: 50건 기준 페이스를 2,507건[검색 구분 레이블=True 대상]으로 단순 비례 외삽 —
시간/HCX 호출수/KOSIS 호출수/HCX 비용 추정치. 추정 전제를 명시)

## 7. Self-validation 관련 우려사항
(TODO: HIGH confidence 과다생성 여부, 근거)

## 8. 70건 골든셋 leakage 체크 결과
(TODO: leakage_hits 개수/내용)

## 9. 판단은 사용자 몫
50건 결과만으로 2,706건 전체 실행 여부를 이 보고서가 결정하지 않는다 — 위 수치를
근거로 사용자가 판단한다.
