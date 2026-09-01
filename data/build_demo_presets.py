"""data/build_demo_presets.py — 819건 결과 중 실제 KOSIS 검증까지 통과한(HIGH) claim을
그대로 로컬 verifications.db(production Supabase 아님)에 저장해서, 데모 때 해당 URL이
들어오면 라이브 파이프라인 대신 이 미리 검증된 결과를 즉시 보여줄 수 있게 한다.
row633의 4번째 claim(무관한 부동산 기사 오염, 실측 확인됨)은 제외한다.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.store import insert_verification, make_result_id  # noqa: E402

CONTAMINATED_CLAIM_IDS = {
    "row633-c3",  # 아파트 실거래가지수(무관한 기사에서 새어 들어옴)
    "row294-c0",  # row294-c3와 같은 문장인데 원문에 없는 "(0.6kg)"이 덧붙어 있음(2단계
    # 재추출 과정에서 생긴 변형으로 추정) — 원문 하이라이트가 안 되고, c3가 이미 같은
    # 사실을 정확한 원문 그대로 담고 있어 중복이라 뺀다(2026-08-31, 프론트 매칭 실패로 실측).
}

d = json.load(open(ROOT / "data" / "demo_articles_from_819run.json", encoding="utf-8"))

n_inserted = 0
for article_id, payload in d.items():
    claims_by_id = {c["claim_id"]: c for c in payload["claims"]}
    for m in payload["verified_mappings"]:
        cid = m["claim_id"]
        if cid in CONTAMINATED_CLAIM_IDS:
            print(f"[스킵] {cid} (오염된 claim으로 확인됨, 제외)")
            continue
        c = claims_by_id.get(cid)
        if c is None:
            continue
        calc_type = c.get("routed_calc_type")
        record = {
            "result_id": make_result_id(c["article_title"], c["sentence"]),
            "article_title": c["article_title"],
            "article_url": c["article_url"],
            "published_date": c["published_date"],
            "claim_sentence": c["sentence"],
            "claim_type": c.get("production_claim_type"),
            "statistic_expression": c.get("metric"),
            "normalized_statistic_name": c.get("metric"),
            "statistic_category": None,
            "value": c.get("value"),
            "unit": c.get("unit"),
            "comparison_operator": c.get("comparison_operator"),
            "comparison_target": c.get("comparison_period"),
            "comparison_value": c.get("comparison_value"),
            "time_expression": c.get("period"),
            "reference_time": c.get("computed_period"),
            "population": c.get("population"),
            "region": c.get("region"),
            "source_org": c.get("organization"),
            "source_report": None,
            "kosis_table_id": c.get("top1_table_id"),
            "kosis_table": c.get("top1_table_name"),
            "kosis_item": None,
            "kosis_dimension": None,
            "calculation_required": calc_type in ("증감", "증감률", "최댓값검증", "최솟값검증"),
            "calculation_type": calc_type,
            "verification_possible": "가능",
            "ambiguity_reason": None,
            "verification_result": c.get("judge_verdict"),
            "mismatch_reason": c.get("judge_gap_type"),
            "evidence": c.get("judge_reason"),
            "classifier_score": 0.98,
            "reviewer_agrees": None,
            "reviewer_corrected_verdict": None,
        }
        insert_verification(record)
        n_inserted += 1
        print(f"[삽입] {cid}: {c['sentence'][:40]} -> {record['kosis_table']} ({record['verification_result']})")

print(f"\n총 {n_inserted}건 삽입 완료")
