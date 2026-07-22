# 1→2→3→4→5→6단계 전체 CSV 순회 실행
"""
agent/pipeline/batch_runner.py — 1→2→3→4→5→6단계 전체 자동 연결 실행기

⚠️ 이전 버전과의 차이:
    예전에는 3단계(통계표 자동 매핑)가 없어서 table_id/claim_sentence를 시나리오마다
    사람이 손으로 지정했습니다. 지금은 B의 search_and_rerank()가 완성되어 있어서
    2단계가 뽑은 Claim을 그대로 3단계에 흘려보내 table_id를 자동으로 정합니다.

    4단계(slot_filler/clarify)도 마찬가지로, generic_slots를 손으로 채워두는 대신
    D의 fill_slots()/clarify()를 실제로 호출합니다. clarify()가 되묻기 질문을
    반환하면 시나리오에 준비된 clarify_reply(사용자가 한 번 더 답했다고 가정한
    발화)로 한 번 더 채워보고, 그래도 부족하면 그 주장은 스킵합니다.

    5단계로 넘어가기 전에 D의 generic slots(period/region/calc_type, 표 구분 없이
    고정)를 C의 table_params.json에 정의된 표별 dimensions(gender/age 등)로
    변환하는 다리(build_kosis_slots)가 필요합니다 — 이 변환이 3단계 표 매핑이
    실제로 없던 시절엔 존재하지 않았던 부분입니다.

    실행 (프로젝트 루트에서):
        python -m agent.pipeline.batch_runner

⚠️ 사전 준비물:
    .env에 HCX_API_KEY, KOSIS_API_KEY 둘 다 필요합니다. 1·2·4단계는 HCX API를,
    5단계는 KOSIS API를 실제로 호출합니다 (더미 아님).

⚠️ 연결하면서 실제로 드러난 팀 간 불일치 (일부러 감추지 않고 그대로 노출시킴):
    1. B(table_catalog.json)와 C(table_params.json)가 서로 다른 표를 가리키는 경우가
       있습니다. 예: "청년 실업률" 계열 주장에 대해 B는 DT_1DA7001S(성별 경제활동인구
       총괄)를 최상위로 매칭하는데, C의 table_params.json에는 그 표가 없고 대신
       DT_1DA7102S(성/연령별 실업률)만 등록돼 있습니다. → build_kosis_slots가
       table_params.json에 없는 table_id를 만나면 None을 반환하고, 그 주장은
       "5단계 파라미터 없음"으로 표시하며 건너뜁니다.
    2. D(clarify_rules.REQUIRED_SLOTS)는 모든 표에 대해 region을 무조건 필수로
       요구하는데, 실제 KOSIS 표 중에는(DT_1DA7102S처럼) 지역 축 자체가 없는 표도
       있습니다. 이 경우도 사람이 미리 안 걸러주면 "지역이 없는 표인데 지역을
       되묻는" 상황이 그대로 재현됩니다 (아래 실행 결과 참고).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

from agent.preprocessing.classifier import classify
from agent.preprocessing.claim_extractor import extract_claims
from agent.mapping.keyword_search import keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import search_and_rerank
from agent.orchestrator.slot_filler import fill_slots
from agent.orchestrator.clarify import clarify
from agent.kosis.api_client import KosisApiClient, KosisApiError
from agent.kosis.calculator import KosisCalculator, CalculationError

TABLE_PARAMS_PATH = Path(__file__).parent.parent / "kosis" / "table_params.json"


ARTICLES = [
    {
        "label": "시나리오 1 — 청년 실업률 (KOSIS 지역 축 없는 표라 되묻기에서 막힘)",
        "published_date": date(2025, 1, 6),
        "article_text": (
            "6일 통계청이 발표한 고용동향에 따르면 지난달 청년 실업률이 6%에 육박한 "
            "것으로 나타났다. 청년층 취업자 수는 46개월 만에 감소로 전환했다."
        ),
        "clarify_reply": "전국 기준으로 작년 대비 증감률 알려줘",
    },
    {
        "label": "시나리오 2 — 소비자물가 (전 과정 자동 연결)",
        "published_date": date(2025, 2, 5),
        "article_text": (
            "5일 통계청이 발표한 소비자물가동향에 따르면 지난달 소비자물가가 전년 "
            "동월 대비 2.2% 오른 것으로 나타났다."
        ),
        "clarify_reply": "전국 기준으로 작년 대비 증감률 알려줘",
    },
]


def build_kosis_slots(table_id: str, generic_slots: dict, table_params: dict) -> Optional[dict]:
    """D의 generic slots(period/region/calc_type, 표 구분 없이 고정)를
    C의 table_params.json에 정의된 표별 dimensions로 변환한다.

    table_params.json에 이 table_id 자체가 없으면 None을 반환한다 (B가 고른 표를
    C가 아직 조사 안 한 경우 — 위 모듈 docstring 이슈 1 참고).
    """
    if table_id not in table_params:
        return None

    base = table_params[table_id]
    kosis_slots: dict = {"period": generic_slots.get("period")}

    for dim_name, dim in base.get("dimensions", {}).items():
        # 이 표에 정의된 축(dim_name)만 채운다. generic_slots에 값이 있으면 쓰고,
        # 없으면 표의 default_value로 채운다 (예: region이 없는 표는 gender/age만 봄).
        value = generic_slots.get(dim_name)
        kosis_slots[dim_name] = value if value is not None else dim.get("default_value")

    return kosis_slots


def run_stage_4(claim_sentence: str, clarify_reply: Optional[str], article_date: date) -> Optional[dict]:
    """4단계: fill_slots + clarify. 한 번에 안 채워지면 clarify_reply로 한 번 더 시도.
    그래도 부족하면 None (되묻기 미해결 → 5단계로 못 감)을 반환한다."""
    slots = fill_slots(claim_sentence, {}, article_date)
    question = clarify(slots)
    print(f"[4단계 slot_filler] 1차 슬롯: {slots}")

    if question and clarify_reply:
        print(f"[4단계 clarify] 되묻기: \"{question}\" → (준비된 답변) \"{clarify_reply}\"")
        slots = fill_slots(clarify_reply, slots, article_date)
        question = clarify(slots)
        print(f"[4단계 slot_filler] 2차 슬롯: {slots}")

    if question:
        print(f"[4단계 clarify] 여전히 부족 → 되묻기: \"{question}\" (여기서 중단)")
        return None

    print("[4단계 clarify] 필수 슬롯 모두 채워짐 → 5단계 진행")
    return slots


def run_stage_5_6(
    table_id: str,
    generic_slots: dict,
    table_params: dict,
    client: KosisApiClient,
    calculator: KosisCalculator,
) -> None:
    kosis_slots = build_kosis_slots(table_id, generic_slots, table_params)
    if kosis_slots is None:
        print(
            f"[5단계 api_client] '{table_id}'가 table_params.json에 없음 "
            "→ C가 아직 이 표를 조사하지 않음 (알려진 갭, 스킵)"
        )
        return

    calc_type = generic_slots.get("calc_type")
    try:
        if calc_type in ("증감", "증감률") and kosis_slots.get("period"):
            base_slots = dict(kosis_slots, period=str(int(kosis_slots["period"]) - 1))
            base_resp = client(table_id, base_slots)
            target_resp = client(table_id, kosis_slots)
            print(f"[5단계 api_client] base   = {base_resp}")
            print(f"[5단계 api_client] target = {target_resp}")

            calc_fn = calculator.compute_change_rate if calc_type == "증감률" else calculator.compute_change
            result = calc_fn(base_resp, target_resp)
            print(f"[6단계 calculator] {result}")
        else:
            resp = client(table_id, kosis_slots)
            print(f"[5단계 api_client] {resp}")
            print("[6단계 calculator] 단순 조회 (calc_type 없음/미지원) → 계산 없이 값 그대로 사용")
    except (KosisApiError, CalculationError) as e:
        print(f"[오류] {type(e).__name__}: {e}")
    except Exception as e:
        print(f"[오류] {type(e).__name__}: {e}")


def run_article(
    article: dict,
    client: KosisApiClient,
    calculator: KosisCalculator,
    table_params: dict,
    embedding_cache: dict,
) -> None:
    print(f"\n{'=' * 60}")
    print(article["label"])
    print(f"기사 원문: \"{article['article_text']}\"")
    print(f"{'-' * 60}")

    try:
        cls_result = classify(article["article_text"])
        print(f"[1단계 classifier] {cls_result}")
    except Exception as e:
        print(f"[1단계 classifier] 실패 ({type(e).__name__}: {e}) → 이 기사 스킵")
        return

    if not cls_result.label:
        print("[1단계 classifier] 무관한 기사로 판정 → 스킵")
        return

    try:
        claims = extract_claims(article["article_text"])
        print(f"[2단계 claim_extractor] {len(claims)}개 주장 추출")
    except Exception as e:
        print(f"[2단계 claim_extractor] 실패 ({type(e).__name__}: {e}) → 이 기사 스킵")
        return

    for claim in claims:
        print(f"{'-' * 60}")
        print(f"주장: \"{claim.sentence}\" (claim_type={claim.claim_type})")

        candidates = search_and_rerank(
            claim,
            keyword_fn=keyword_search,
            embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
        )
        if not candidates:
            print("[3단계 매핑] 매칭되는 표 없음 → 스킵")
            continue

        top = candidates[0]
        print(f"[3단계 매핑] 최상위 후보: {top.table_name} ({top.table_id}) score={top.score:.3f}")

        slots = run_stage_4(claim.sentence, article.get("clarify_reply"), article["published_date"])
        if slots is None:
            continue

        run_stage_5_6(top.table_id, slots, table_params, client, calculator)


def main() -> None:
    try:
        client = KosisApiClient()
    except RuntimeError as e:
        print(f"[중단] {e}")
        return

    calculator = KosisCalculator()

    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)

    embedding_cache = build_table_embedding_cache()

    for article in ARTICLES:
        run_article(article, client, calculator, table_params, embedding_cache)


if __name__ == "__main__":
    main()
