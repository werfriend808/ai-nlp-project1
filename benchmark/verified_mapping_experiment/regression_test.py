"""benchmark/verified_mapping_experiment/regression_test.py
================================================================
pilot20_report.md §10에서 발견된 row15-c0류 버그의 회귀 테스트.

버그: baseline이 이미 독립 검증으로 confidence=HIGH인데, mapping 후보를 재검증했더니
그 후보도 HIGH가 나와서(그럴듯하지만 실제로는 틀린 다른 표) assisted/gold를 덮어써버림
→ R@1이 1(정답)에서 0(오답)으로 악화.

수정 규칙(pilot20_run.py의 process_evaluation_claim, 2026-08-30): baseline이 이미
HIGH면 mapping 후보를 재검증조차 하지 않는다.

이 테스트는 실제 KOSIS API/HCX 호출 없이(search_and_rerank, verify_table_for_claim,
enrich_candidates_with_db를 monkeypatch) row15-c0와 동일한 형태의 시나리오를 순수
로직 레벨에서 재현하고, 수정된 로직이 baseline을 보호하는지 확인한다.
production 코드는 전혀 건드리지 않는다(import만).
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(ROOT))

import pilot20_run as p20  # noqa: E402
import pilot_run as pr  # noqa: E402
from agent.interfaces import Claim, TableCandidate  # noqa: E402


def _fake_article():
    return {
        "article_id": "regression-row15-c0",
        "split": "evaluation",
        "selection_bucket": "institution_mentioned",
        "article_title": "회귀테스트용 가짜 기사",
        "article_url": None,
        "published_date": "2026-01-01",
    }


def _fake_claim():
    return Claim(
        sentence="지난달 취업자 수는 2909만 1000명으로 집계됐다.",
        claim_type="규모",
        period="2025-12",
        unit="명",
        statistic_expression="취업자 수",
        value=29091000.0,
        source_org="통계청",
        prev_sentence="통계청이 발표한 고용동향에 따르면",
    )


def run_regression() -> dict:
    correct = TableCandidate(
        table_id="DT_CORRECT_001", table_name="경제활동인구조사 취업자수(정답 표)",
        score=0.95, source_meta=None, org_id="101",
    )
    plausible_wrong = TableCandidate(
        table_id="DT_WRONG_002", table_name="지역별고용조사 취업자수(오답이지만 그럴듯한 표)",
        score=0.90, source_meta=None, org_id="101",
    )
    candidates = [correct, plausible_wrong]

    # baseline(top1=correct)은 독립 검증 HIGH, mapping 후보(plausible_wrong)도 재검증하면
    # HIGH가 나오는(버그 재현) 상황을 그대로 스텁으로 강제한다.
    verify_calls = {"n": 0}

    def fake_verify_table_for_claim(article, claim, calc_type, table_id, table_name, org_id,
                                      table_params, catalog_by_id, client, calculator):
        verify_calls["n"] += 1
        if table_id == correct.table_id:
            return {"table_id": table_id, "table_name": table_name, "confidence": "HIGH",
                     "confidence_reason": "실측치 일치", "judge_verdict": "match",
                     "computed_value": 29091000.0, "computed_unit": "명", "computed_period": "2025-12"}
        if table_id == plausible_wrong.table_id:
            # 버그 재현 조건: 이 후보도 독립 검증 결과 HIGH (다른 값이지만 judge가 통과시켰다고 가정)
            return {"table_id": table_id, "table_name": table_name, "confidence": "HIGH",
                     "confidence_reason": "실측치 일치(다른 지역 축, 우연히 근접값)", "judge_verdict": "match",
                     "computed_value": 29088000.0, "computed_unit": "명", "computed_period": "2025-12"}
        raise AssertionError(f"unexpected table_id probed: {table_id}")

    p20.verify_table_for_claim = fake_verify_table_for_claim

    orig_search_and_rerank = pr.search_and_rerank
    orig_enrich = pr.enrich_candidates_with_db
    pr.search_and_rerank = lambda *a, **kw: candidates
    pr.enrich_candidates_with_db = lambda *a, **kw: None

    try:
        discovery_mappings = [{
            "table_id": plausible_wrong.table_id, "organization": "통계청",
            "mapping_source": "regression_stub",
        }]
        rec, cand_rows, verification_pair = p20.process_evaluation_claim(
            article=_fake_article(), claim=_fake_claim(), claim_id="regression-claim-1",
            table_params={}, catalog_by_id={}, doc_texts={"texts": {}, "_emb_cache": {}},
            vdb_fn=None, bm25_fn=None, client=None, calculator=None, db_conn=None,
            discovery_mappings=discovery_mappings,
        )
    finally:
        pr.search_and_rerank = orig_search_and_rerank
        pr.enrich_candidates_with_db = orig_enrich

    checks = []

    def check(name, cond, detail=""):
        checks.append({"name": name, "pass": bool(cond), "detail": detail})

    check("verify_table_for_claim이 정답 후보에 대해 1회만 호출됨(mapping 후보는 재검증 안 함)",
          verify_calls["n"] == 1, f"실제 호출 횟수={verify_calls['n']}")
    check("baseline confidence == HIGH",
          rec["confidence"] == "HIGH", f"실제={rec['confidence']}")
    check("assisted_top1_table_id가 baseline(정답)을 유지함(덮어쓰기 안 됨)",
          rec["assisted_top1_table_id"] == correct.table_id,
          f"실제={rec['assisted_top1_table_id']} (버그면 {plausible_wrong.table_id})")
    check("gold_table_id == baseline(정답 표)",
          rec["gold_table_id"] == correct.table_id, f"실제={rec['gold_table_id']}")
    check("assisted_recall[1] == 1 (버그면 0으로 떨어짐 — row15-c0와 동일 증상)",
          rec["assisted_recall"][1] == 1, f"실제={rec['assisted_recall']}")
    check("mapping_applied == False (baseline 보호로 적용 안 됨)",
          rec["mapping_applied"] is False, f"실제={rec['mapping_applied']}")
    check("ab_outcome == mapping_skipped_baseline_protected (신규 라벨)",
          rec["ab_outcome"] == "mapping_skipped_baseline_protected", f"실제={rec['ab_outcome']}")
    check("mapping_reject_reason에 baseline 보호 규칙이 명시됨",
          rec.get("mapping_reject_reason") and "baseline" in rec["mapping_reject_reason"],
          f"실제={rec.get('mapping_reject_reason')}")

    all_pass = all(c["pass"] for c in checks)
    result = {"verdict": "REGRESSION_TEST_PASS" if all_pass else "REGRESSION_TEST_FAIL", "checks": checks}
    return result


def run_rescue_sanity_check() -> dict:
    """보호 규칙이 mapping의 정상 순기능(baseline이 HIGH가 아닐 때 mapping이 구제하는 경로)까지
    막아버리지 않았는지 확인하는 대조 시나리오. baseline은 검증 실패(MEDIUM), mapping 후보는
    독립 검증 HIGH → 이 경우는 여전히 mapping이 적용돼야 한다(row4 pilot20 사례와 동일 성격)."""
    weak_top1 = TableCandidate(
        table_id="DT_WEAK_003", table_name="애매한 표(검증 실패)",
        score=0.80, source_meta=None, org_id="101",
    )
    rescuer = TableCandidate(
        table_id="DT_RESCUE_004", table_name="mapping이 찾아준 정답 표",
        score=0.75, source_meta=None, org_id="101",
    )
    candidates = [weak_top1, rescuer]
    verify_calls = {"n": 0}

    def fake_verify(article, claim, calc_type, table_id, table_name, org_id,
                     table_params, catalog_by_id, client, calculator):
        verify_calls["n"] += 1
        if table_id == weak_top1.table_id:
            return {"table_id": table_id, "table_name": table_name, "confidence": "MEDIUM",
                     "confidence_reason": "값 불일치", "judge_verdict": "mismatch",
                     "computed_value": 1.0, "computed_unit": "명", "computed_period": "2025-12"}
        if table_id == rescuer.table_id:
            return {"table_id": table_id, "table_name": table_name, "confidence": "HIGH",
                     "confidence_reason": "실측치 일치", "judge_verdict": "match",
                     "computed_value": 29091000.0, "computed_unit": "명", "computed_period": "2025-12"}
        raise AssertionError(f"unexpected table_id probed: {table_id}")

    p20.verify_table_for_claim = fake_verify
    orig_search_and_rerank = pr.search_and_rerank
    orig_enrich = pr.enrich_candidates_with_db
    pr.search_and_rerank = lambda *a, **kw: candidates
    pr.enrich_candidates_with_db = lambda *a, **kw: None
    try:
        discovery_mappings = [{"table_id": rescuer.table_id, "organization": "통계청",
                                "mapping_source": "regression_stub"}]
        rec, _, _ = p20.process_evaluation_claim(
            article=_fake_article(), claim=_fake_claim(), claim_id="regression-claim-2",
            table_params={}, catalog_by_id={}, doc_texts={"texts": {}, "_emb_cache": {}},
            vdb_fn=None, bm25_fn=None, client=None, calculator=None, db_conn=None,
            discovery_mappings=discovery_mappings,
        )
    finally:
        pr.search_and_rerank = orig_search_and_rerank
        pr.enrich_candidates_with_db = orig_enrich

    checks = []

    def check(name, cond, detail=""):
        checks.append({"name": name, "pass": bool(cond), "detail": detail})

    check("mapping 후보 재검증이 실제로 호출됨(baseline이 HIGH가 아니므로 보호 규칙 미적용)",
          verify_calls["n"] == 2, f"실제 호출 횟수={verify_calls['n']}")
    check("mapping_applied == True (정상 구제 경로는 여전히 동작)",
          rec["mapping_applied"] is True, f"실제={rec['mapping_applied']}")
    check("assisted_top1_table_id == rescuer 표 (구제됨)",
          rec["assisted_top1_table_id"] == rescuer.table_id, f"실제={rec['assisted_top1_table_id']}")
    check("ab_outcome == mapping_rescued_claim_improvement",
          rec["ab_outcome"] == "mapping_rescued_claim_improvement", f"실제={rec['ab_outcome']}")

    all_pass = all(c["pass"] for c in checks)
    return {"verdict": "SANITY_CHECK_PASS" if all_pass else "SANITY_CHECK_FAIL", "checks": checks}


if __name__ == "__main__":
    import json
    result = run_regression()
    sanity = run_rescue_sanity_check()
    print(json.dumps({"regression": result, "rescue_sanity_check": sanity}, ensure_ascii=False, indent=2))
    for label, r in (("REGRESSION", result), ("RESCUE_SANITY", sanity)):
        for c in r["checks"]:
            mark = "OK " if c["pass"] else "FAIL"
            print(f"[{label}][{mark}] {c['name']} :: {c['detail']}")
    print()
    overall = "PASS" if result["verdict"].endswith("PASS") and sanity["verdict"].endswith("PASS") else "FAIL"
    print(f"=== REGRESSION_TEST = {overall} ===")
    sys.exit(0 if overall == "PASS" else 1)
