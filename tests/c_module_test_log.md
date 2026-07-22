# C 담당(5·6단계) 모듈 단위 테스트 기록

- 대상: `agent/kosis/api_client.py`, `agent/kosis/calculator.py`
- 결과: **15/16 PASS** (아래 "발견된 이슈" 참고)

## 실행 방법

프로젝트 루트에서 실행 (실제 KOSIS API를 호출하므로 `.env`에 `KOSIS_API_KEY` 필요):

```
python -m tests.test_kosis_module
```

Windows 콘솔(cmd/PowerShell)에서 한글 결과가 깨져 보이면 UTF-8을 강제해서 실행:

```
# cmd
set PYTHONUTF8=1 && set PYTHONIOENCODING=utf-8 && python -m tests.test_kosis_module

# PowerShell
$env:PYTHONUTF8=1; $env:PYTHONIOENCODING="utf-8"; python -m tests.test_kosis_module
```

맨 아래 `N/16 PASS` 줄로 요약을 확인하고, `❌ FAIL`로 표시된 케이스가 있으면 그 옆의 에러 메시지를 보고 원인을 찾으면 됩니다. (`python agent/pipeline/...`처럼 파일 경로로 직접 실행하면 `agent.*` import가 깨지니 꼭 `-m tests.test_kosis_module` 형태로 실행해야 함.)

## 케이스 목록

| # | 이름 | 내용 | 결과 |
|---|---|---|---|
| 1 | case_01_unemployment_2024_youth | 청년실업률 2024 단일 조회 | ✅ |
| 2 | case_02_unemployment_2023_youth | 청년실업률 2023 단일 조회 | ✅ |
| 3 | case_03_unemployment_male | 청년실업률(남자) | ✅ |
| 4 | case_04_unemployment_female | 청년실업률(여자) | ✅ |
| 5 | case_05_unemployment_all_ages | 전체 연령 실업률 2020 | ✅ |
| 6 | case_06_farm_total | 농가 수(전체) 2024 | ✅ |
| 7 | case_07_farm_20s | 농가 수(20~24세 경영주) | ✅ |
| 8 | case_08_farm_elderly_80plus | 농가 수(80세이상 경영주) | ✅ |
| 9 | case_09_ratio_elderly_farm | 65세 이상 고령농가 비율 계산 (compute_ratio) | ✅ |
| 10 | case_10_sum_all_age_bands_equals_total | 연령대별(T01~T14) 합계 vs 전체(T00) 무결성 체크 | ❌ (아래 이슈 참고) |
| 11 | case_11_change_rate_unemployment | 청년실업률 2023→2024 증감률 | ✅ |
| 12 | case_12_change_unemployment | 청년실업률 2019→2024 증감(절대값) | ✅ |
| 13 | case_13_ratio_zero_denominator_should_fail | 분모 0일 때 CalculationError 발생 확인 (엣지케이스) | ✅ |
| 14 | case_14_sum_mismatched_units_should_fail | 단위 다른 값 합산 시 CalculationError 확인 (엣지케이스) | ✅ |
| 15 | case_15_unknown_table_id_should_fail | 존재하지 않는 table_id일 때 KeyError 확인 (엣지케이스) | ✅ |
| 16 | case_16_region_all_returns_many_rows_should_fail | code_map에 없는 region 값 → KosisApiError 확인 (엣지케이스) | ✅ |

## 발견된 이슈

### 1. (수정 완료) `region.code_map`의 `"전국": "ALL"`이 잘못됨
`objL1=ALL`을 보내면 "전국 총계"가 아니라 **시도 57개 전체**가 반환됨(855행). `api_client.py`는 결과가 1행이 아니면 에러를 던지는 구조라 이 상태로는 절대 성공할 수 없었음. `"전국": "000"`으로 정정. → `table_params.json` 수정 완료.

### 2. (수정 완료) `age.code_map`이 비어있었음 (`DT_1EA1019`)
연령대 코드(T00~T14)가 하나도 매핑되어 있지 않아 `age` 슬롯을 넘기면 항상 KOSIS 에러(`[21] 잘못된 요청 변수`)가 났음. 실제 조회해서 15개 코드 전부 채움.

### 3. (확인 완료, 미해결) `farm_type` 축은 이 표에 없음
`DT_1EA1019`는 지역(region)·연령(age) 두 축만 있고 "영농형태"(과수농가 등) 축 자체가 없음. 브리프 예시(과수 농가 166,558가구)를 재현하려면 완전히 다른 tblId를 새로 찾아야 함 — 아직 미해결, 3단계 표 매핑에서 다룰 문제.

### 4. (신규 발견, 코드 버그 아님) 연령대별 합계가 전체값과 1 차이 남
`case_10`: T01~T14(연령대별 농가 수)를 다 더하면 973,706인데, T00(계)으로 직접 조회하면 973,707. **calculator.py의 `compute_sum` 로직은 정상**이고, KOSIS 원본 데이터 자체에 반올림/집계 방식 차이로 1가구 오차가 있는 것으로 보임. 8단계(설명 생성)에서 이런 미세 오차는 "불일치"로 판정하면 안 되므로, 7단계(판정) 설계 시 완전 일치가 아니라 **허용 오차 범위**를 둬야 한다는 시사점.

## 커버 안 된 것 (다음에 필요하면 추가)
- `DT_1DA7102S`/`DT_1EA1019` 외 나머지 5개 표(`table_catalog.json`에는 있지만 `table_params.json`에는 없음)는 이번 테스트 범위 밖.
- 네트워크 타임아웃/HTTP 5xx 같은 인프라성 실패 케이스는 다루지 않음.
