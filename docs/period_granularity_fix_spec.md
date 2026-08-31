> **2026-08-30 업데이트: STEP1~4 전부 이 세션에서 직접 구현·실제 KOSIS API로 검증
> 완료했습니다.** 아래 스펙은 원래 팀원 전달용으로 작성했으나, 시점 버그가 다른
> 진행 중이던 검증 작업(verified_mapping pilot)을 막고 있어서 먼저 처리했습니다.
> 실제 변경 파일: `agent/pipeline/batch_runner.py`(`_infer_desired_granularity`
> D/H/F 인식 추가, `_build_dynamic_kosis_slots` `_select_prd_se` 재사용하도록 수정),
> `agent/kosis/enrich_objl.py`(`fetch_table_detail`이 `prd_se_list` 전체를 반환하도록
> 확장, `_probe_additional_periods` 신규), `agent/kosis/detail_cache.py`(`prd_se_list`
> 전체를 캐시·반환하도록 수정). 실제 표(`DT_1DA7024S`)로 KOSIS API 실측 검증했고
> 기존 테스트(`tests/test_slot_filler_module.py`, 13/13) 회귀 없음 확인. 아래
> 원본 스펙은 "왜 이렇게 고쳤는지" 배경 설명으로 그대로 남겨둡니다.

---

# VDB 전용 표(28.7만 건) 시점(주기) 처리 보강 작업

## 0. 배경 / 목적

현재 4단계(slot filling)에서 claim이 원하는 시점 주기(월/분기/일/반기/다년 등)를
KOSIS API 조회에 반영하는 로직이 **64개 수동 카탈로그 표(`table_params.json`)에만
있고, VDB로 찾은 나머지 약 28.7만 건 표에는 전혀 없다.**

골든셋 70건으로 실측한 결과: 월 단위 시점이 필요한 claim 21건 중 **17건(전체의
24.3%)이 VDB 전용 표라서 이 매칭이 아예 안 되고, 표의 기본 주기(대부분 연간 "Y")로
무조건 조회되고 있다.** "지난달 취업자 2909만1000명" 같은 claim을 연간 데이터로
비교하면 애초에 일치할 수 없다 — 검증 실패 또는 오판정의 실제 원인 중 하나로
추정된다.

**중요: 이 작업은 embedding_text/vector/검색(Dense/BM25/RRF/CE) 파이프라인과
전혀 무관하다. 재임베딩 불필요.** 이건 표가 이미 검색으로 정확히 찾아진 *이후*,
KOSIS API에서 실제 값을 조회하는 4~5단계에만 영향을 준다.

---

## 1. 절대 원칙

- production 검색 로직(D2/Dense/BM25/RRF/CE) 변경 금지 — 이번 작업과 무관.
- 28.7만 건 전체를 미리 일괄 처리(batch)하지 않는다 — 기존에 이미 있는
  **lazy(필요할 때만, 매칭된 표 한정) + 캐시** 패턴을 그대로 따른다.
- 특정 table_id 하드코딩 금지, 특정 claim 전용 처리 금지.
- KOSIS API 대량 호출 금지 — 이미 있는 `detail_cache`(90일 TTL 캐시) 인프라를
  그대로 재사용해서 신규 호출은 "처음 만나는 표 1회"로 제한한다.
- 이번 작업 전에 실제 코드를 직접 읽고, 아래 grounding과 실제 코드가 일치하는지
  먼저 검증할 것(추측 금지, 이 문서가 실수했을 수도 있다).

---

## 2. 문제의 정확한 위치 — 3개 파일, 순서대로 확인된 실제 원인

### 2-1. `agent/kosis/enrich_objl.py::fetch_table_detail()` — 근본 원인

```python
_PRD_SE_ATTEMPTS = (
    ("M", "200501", "202612"),
    ("A", "2005", "2026"),
    ("Y", "2005", "2026"),
    ("Q", "20051", "20264"),
)
```

`fetch_table_detail()`(약 209번째 줄)은 이 4개 코드(M/A/Y/Q)만 순서대로 시도하고
**"처음 성공한 것 하나"에서 멈춘다.** 그 표가 실제로 여러 주기를 지원해도(예:
연간+월간 둘 다) 코드 하나만 골라서 반환한다. **D(일)/H(반기)/F(다년)/IR(비정기)은
아예 시도조차 안 한다.**

### 2-2. `agent/kosis/detail_cache.py::get_table_detail()` / `_save()`

캐시 미스일 때 `_save(..., prd_se_list=[detail["prd_se"]], ...)`로 저장한다 —
2-1이 하나만 반환하니 캐시에도 **원소 1개짜리 리스트**로만 들어간다. 캐시 히트
시에도 `"prd_se": prd_se_list[0] if prd_se_list else None`로 **첫 번째 값만
꺼내 쓴다.**

### 2-3. `agent/pipeline/batch_runner.py::_build_dynamic_kosis_slots()` (약 1041번째 줄)

VDB 전용 표에 대해 지역/연령/성별 축은 실시간으로 잘 매핑해주지만, **`prd_se`를
고르는 로직 자체가 아예 없다.** `generic_slots.get("prd_se")`가 있으면 그대로
넘기기만 하는데, 이 값은 stage4에서 애초에 채워진 적이 없다(아래 2-4).

### 2-4. `agent/pipeline/batch_runner.py::_infer_desired_granularity()` (542번째 줄)

claim 문장에서 "분기"→Q, "월"/"달"→M만 인식한다. 그 외(일 단위 구체 날짜,
반기, 다년, 비정기)는 인식 자체가 없어서 `None`을 반환하고, 상위 로직이
기본값 "Y"로 폴백한다. **이건 64개 카탈로그 표에도 똑같이 영향을 준다**
(VDB 전용 표만의 문제가 아님 — 이 함수는 두 경로가 공유한다).

참고로 KOSIS 공식 prdSe 코드는 실측상 **D/M/Q/H/Y/F/IR** 7종 + 별도 연간 별칭
"A"가 있다(`agent/kosis/reembed_worker.py`의 `PRD_SE_ATTEMPTS` 주석, 2026-08-25
실측 근거 참고). **F는 "반기"가 아니라 "2/3/4/5/10년 등 다년 주기 전부를
묶은 코드"이고, 반기는 별도로 H다** — 기존 `migrate_prdse_to_list.py` 주석의
"F(반기)"는 오기이니 참고 시 주의.

---

## 3. 수정해야 할 것 (4곳, 순서대로)

### STEP 1 — `_infer_desired_granularity()` 확장 (batch_runner.py:542)

기존 분기/월 인식은 그대로 두고, 아래를 추가로 인식해서 해당 코드를 반환하게
한다.

- **D(일)**: 구체적인 날짜 표현("1월 2일", "지난달 16일", "24일 발표" 등 — 정확한
  정규식/kiwi 토큰 패턴은 실제 골든셋 문장으로 설계·검증할 것, 임의로 만들지 말 것)
- **H(반기)**: "상반기"/"하반기"/"반기" 등
- **F(다년)**: "격년"/"2년마다"/"3년 주기" 등 명시적 다년 표현이 있을 때만
  (없으면 억지로 F를 추정하지 않는다 — 애매하면 기존처럼 None)
- **IR/A**: claim 문장만으로 이 둘을 구분할 신호가 없을 가능성이 높다 — 억지로
  만들지 말고, 표가 지원하는 목록에 Y가 없고 A만 있을 때의 폴백(STEP 4에서
  `_select_prd_se`가 이미 이 폴백을 하고 있으니 여기선 손댈 필요 없을 수 있음,
  실제 코드 재확인할 것)

**반드시 골든셋(`benchmark/search_experiment/eval_set.json`)의 실제 문장으로
패턴을 설계하고, 오탐(false positive)이 없는지 확인할 것.**

### STEP 2 — `fetch_table_detail()` 수정 (enrich_objl.py:60, 209)

`_PRD_SE_ATTEMPTS`에 D/H/F/IR을 추가하고, **"첫 성공에서 멈추기"가 아니라
"시도 가능한 코드를 전부 테스트해서 성공한 것 전부 리스트로 모으기"**로 바꾼다
(`agent/kosis/verify_multi_period_support.py`가 64개 카탈로그 표에 대해 이미
했던 방식과 동일한 원리 — 그 스크립트를 참고해서 로직을 재사용/이식할 것,
새로 설계하지 말 것). 반환값을 `prd_se`(단일값) 대신 `prd_se_list`(리스트)로
바꿔야 한다 — 이 함수를 쓰는 다른 호출부가 있는지 먼저 확인하고 하위 호환을
검토할 것.

**주의**: 시도 코드 수가 4개→8개로 늘면 표 하나당 API 호출이 늘어난다(이미
lazy+캐시라 표당 1회뿐이지만, 그 1회 안에서의 호출 수는 는다) — 실제 latency
증가폭을 소규모로 실측하고 보고할 것.

### STEP 3 — `detail_cache.py` 수정

`_save()`/`get_table_detail()`이 `prd_se_list` 전체를 그대로 저장·반환하도록
수정(`prd_se_list[0]`로 자르는 부분 제거). 반환 dict의 키를 `prd_se`(단일) 대신
`prd_se_list`(리스트)로 바꾸거나, 하위 호환을 위해 둘 다 넣을지 결정할 것
(기존 호출부가 `detail["prd_se"]`를 쓰는 곳이 있으면 깨지지 않게 확인).

**기존 캐시 테이블(`kosis_table_detail_cache`, 90일 TTL)에 이미 저장된
row들은 전부 원소 1개짜리 리스트다 — 이 작업 이후 자연스럽게 TTL 만료되며
새 로직으로 갱신되니, 캐시를 억지로 전부 지우거나 마이그레이션할 필요는
없다(단, 급하면 해당 row들만 골라 무효화하는 것도 검토 가능 — 필수는 아님).**

### STEP 4 — `_build_dynamic_kosis_slots()` 수정 (batch_runner.py:1041)

`detail["prd_se_list"]`와 `_infer_desired_granularity(claim_sentence)`를
**기존에 64개 카탈로그 경로가 이미 쓰고 있는 `_select_prd_se(supported, desired)`
함수에 그대로 넣어서** `kosis_slots["prd_se"]`를 채운다 — 새 선택 로직을 따로
만들지 말고 반드시 이 기존 함수를 재사용할 것(로직 두 벌 관리 방지).

---

## 4. 검증 방법 (필수)

1. 골든셋 70건 중 VDB 전용 표 + 월/분기/일 단위 claim(위 §0에서 확인된 17건 등)을
   대상으로, 수정 전/후 실제로 조회되는 `prd_se`와 반환값이 바뀌는지 확인.
2. 64개 카탈로그 표 쪽 claim들도 회귀(regression) 없는지 확인(`_infer_desired_
   granularity` 변경이 공유 함수라 두 경로 모두에 영향 준다).
3. `detail_cache` 캐시 히트/미스 케이스 둘 다 테스트(새로 조회하는 표 + 이미
   캐시된 표).
4. production DB에는 SELECT만, 캐시 테이블(`kosis_table_detail_cache`) 갱신은
   이 모듈이 원래 하던 정상 동작이니 손대도 무방하나, 실제 조회 대상 데이터
   테이블(`kosis_vdb_tables_qwen` 등)에는 쓰기 금지.
5. 실제 반영 전에 소규모(예: 골든셋 17건 또는 그 이하)로 결과를 먼저 확인하고
   보고할 것 — 한 번에 production 전체에 반영하지 말 것.

---

## 5. 최종 확인 질문 (작업자가 완료 후 답할 것)

1. STEP1의 D/H/F 탐지 정규식이 실제로 오탐 없이 골든셋 17건을 올바르게
   분류하는가?
2. STEP2에서 API 호출 수/latency가 얼마나 늘었는가(표당 1회 기준)?
3. 64개 카탈로그 표 쪽에서 회귀가 발생했는가?
4. 골든셋 17건 중 실제로 몇 건이 이제 올바른 주기로 조회되는가?
