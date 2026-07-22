"""
agent/pipeline/batch_runner.py — 1→2→4→5→6단계 연결 데모 실행기

현재 상태:
    1단계(classifier)·2단계(claim_extractor)가 HCX 실호출로 구현됐고, 4단계
    (clarify_rules, 규칙기반)·5단계(api_client, 실제 KOSIS API 호출)·6단계
    (calculator)도 이미 되어 있습니다. 3,7,8단계(표 매핑/judge/explainer)는
    아직 빈 파일이거나 주석 한 줄뿐이라 이 러너에 포함하지 않았습니다.

    3단계(통계표 자동 매핑)가 없어서, 각 시나리오의 article_text를 1·2단계에
    실제로 흘려보내되(HCX_API_KEY 있으면 진짜로 분류/추출 결과가 출력됨), 어떤
    KOSIS 표·슬롯을 쓸지는 사람이 손으로 미리 정해서 4→5→6으로 이어붙였습니다.
    HCX_API_KEY가 없거나 호출이 실패해도 1·2단계는 건너뛰고 4→5→6은 계속 실행됩니다
    (미리 준비된 claim_sentence로 대체).

    실행 (프로젝트 루트에서):
        python -m agent.pipeline.batch_runner

 HCX_API_KEY 관련 주의:
    hcx_client.py는 환경변수 이름을 HCX_API_KEY로 읽는데, .env 템플릿에는
    NCP_CLOVASTUDIO_API_KEY라는 다른 이름이 주석으로만 남아있습니다. 1,2단계를
    실제로 돌리려면 .env에 HCX_API_KEY=... 를 추가해야 합니다 (이름 안 맞으면
    "HCX_API_KEY가 없습니다" 에러가 남).

 연결하면서 발견한 팀 간 이름/구조 불일치 (3단계 실제 구현 시 정리 필요):
    - D(clarify_rules.REQUIRED_SLOTS)는 slot 이름을 영어(period/region/calc_type)로,
      전체 표에 동일하게 고정해서 씀.
    - B(table_catalog.json의 required_slots)는 한글(시점/지역)로, 표마다 다르게 씀.
    - C(table_params.json의 dimensions)는 표마다 다른 키(gender/age/farm_type 등)를 씀.
    - interfaces.py 4단계 정의상 required_slots는 TableCandidate에서 나와야 하는데
      (표마다 달라야 함) 지금 clarify_rules.REQUIRED_SLOTS는 표 구분 없이 고정값이라
      이 표(DT_1DA7102S)에는 필요 없는 "region"까지 물어보게 됨.
    → 아직 3단계가 없어서 이 셋을 자동으로 연결하는 매핑이 없음. 이 데모에서는
      generic_slots(=D가 보는 것)와 kosis_slots(=C가 실제로 쓰는 것)를 시나리오별로
      사람이 직접 맞춰서 예시로 넣었습니다.
"""

from __future__ import annotations

from agent.preprocessing.classifier import classify
from agent.preprocessing.claim_extractor import extract_claims
from agent.orchestrator.clarify_rules import get_next_clarify_step
from agent.kosis.api_client import KosisApiClient, KosisApiError
from agent.kosis.calculator import KosisCalculator, CalculationError


SCENARIOS = [
    {
        "label": "시나리오 1 — 슬롯 부족 (되묻기만 하고 종료)",
        "article_text": "최근 청년층을 중심으로 고용 시장이 다시 나빠지고 있다는 우려가 나온다. "
        "전문가들은 청년 실업률이 다시 오르는 추세라고 진단했다.",
        "claim_sentence": "청년 실업률이 올랐다",
        "generic_slots": {"period": "2024"},  # region, calc_type 없음
        "table_id": None,
        "kosis_slots": None,
    },
    {
        "label": "시나리오 2 — 단일 조회 (계산 없이 KOSIS 값 그대로)",
        "article_text": "6일 통계청이 발표한 고용동향에 따르면 지난달 청년 실업률이 6%에 육박한 "
        "것으로 나타났다. 청년층 취업자 수는 46개월 만에 감소로 전환했다.",
        "claim_sentence": "지난달 청년 실업률이 6%에 육박했다",
        "generic_slots": {"period": "2024", "region": "전국", "calc_type": "단순조회"},
        "table_id": "DT_1DA7102S",
        "kosis_slots": {"period": "2024", "gender": "전체", "age": "청년(15~29세)"},
    },
    {
        "label": "시나리오 3 — 증감률 계산 (KOSIS 2번 호출 + calculator)",
        "article_text": "일각에서는 2023년과 비교했을 때 올해 청년 실업률이 크게 올랐다는 주장이 "
        "제기됐다. 청년 고용 상황이 눈에 띄게 악화됐다는 것이다.",
        "claim_sentence": "청년 실업률이 작년보다 크게 올랐다",
        "generic_slots": {"period": "2024", "region": "전국", "calc_type": "증감률"},
        "table_id": "DT_1DA7102S",
        "kosis_slots_base": {"period": "2023", "gender": "전체", "age": "청년(15~29세)"},
        "kosis_slots_target": {"period": "2024", "gender": "전체", "age": "청년(15~29세)"},
    },
]


def run_stage_1_2(scenario: dict) -> None:
    """1단계 classifier → 2단계 claim_extractor. HCX_API_KEY 없거나 호출 실패하면
    건너뛰고(에러만 출력) 아래 4→5→6은 미리 준비된 claim_sentence로 계속 진행합니다."""
    try:
        cls_result = classify(scenario["article_text"])
        print(f"[1단계 classifier] {cls_result}")
    except Exception as e:
        print(f"[1단계 classifier] 건너뜀 ({type(e).__name__}: {e})")
        return

    try:
        claims = extract_claims(scenario["article_text"])
        print(f"[2단계 claim_extractor] {len(claims)}개 주장 추출")
        for c in claims:
            print(f"   - {c}")
    except Exception as e:
        print(f"[2단계 claim_extractor] 건너뜀 ({type(e).__name__}: {e})")


def run_scenario(scenario: dict, client: KosisApiClient, calculator: KosisCalculator) -> None:
    print(f"\n{'=' * 60}")
    print(scenario["label"])
    print(f"기사 원문: \"{scenario['article_text']}\"")
    print(f"{'-' * 60}")

    run_stage_1_2(scenario)

    print(f"{'-' * 60}")
    print(f"주장 문장(3단계 표 매핑은 아직 없어 손으로 지정): \"{scenario['claim_sentence']}\"")

    # 4단계: 필수 슬롯 채워졌는지 확인 (규칙 기반, LLM 호출 없음)
    step = get_next_clarify_step(scenario["generic_slots"])
    print(f"[4단계 clarify] missing_slots={step.missing_slots}")
    if step.next_slot_to_ask:
        print(f"[4단계 clarify] → 되물어야 함: \"{step.clarify_question}\"")
        print("[4단계 clarify] 슬롯이 부족해 5,6단계로 진행하지 않습니다.")
        return
    print("[4단계 clarify] 필수 슬롯 모두 채워짐 → 다음 단계 진행")

    if scenario["table_id"] is None:
        return

    # 5단계: KOSIS API 호출 (표별 실제 slots는 kosis_slots* 에 사람이 미리 매핑해둠)
    try:
        if "kosis_slots" in scenario:
            resp = client(scenario["table_id"], scenario["kosis_slots"])
            print(f"[5단계 api_client] KosisApiResponse = {resp}")
            print(f"[결과] 실제 KOSIS 값: {resp.raw_value}{resp.unit} ({resp.period})")
        else:
            base = client(scenario["table_id"], scenario["kosis_slots_base"])
            target = client(scenario["table_id"], scenario["kosis_slots_target"])
            print(f"[5단계 api_client] base   = {base}")
            print(f"[5단계 api_client] target = {target}")

            # 6단계: 표 연산 (반드시 파이썬 계산, LLM 미사용)
            result = calculator.compute_change_rate(base, target)
            print(f"[6단계 calculator] ComputedResult = {result}")
            print(f"[결과] 증감률: {result.raw_value}{result.unit} ({result.period})")
    except (KosisApiError, CalculationError) as e:
        print(f"[오류] {type(e).__name__}: {e}")
    except Exception as e:
        print(f"[오류] KOSIS 호출 실패 (네트워크/API 키 확인 필요): {type(e).__name__}: {e}")


def main() -> None:
    try:
        client = KosisApiClient()
    except RuntimeError as e:
        print(f"[중단] {e}")
        return

    calculator = KosisCalculator()
    for scenario in SCENARIOS:
        run_scenario(scenario, client, calculator)


if __name__ == "__main__":
    main()
