"""
tests/test_stage_7_8_diagnostics.py — 7·8단계(judge/explain) 종단 진단 테스트

agent/pipeline/batch_runner.py의 ARTICLES 중 tests/pipeline_integration_log.md(Day3, 7/24)가
실제로 7·8단계까지 도달했다고 기록한 시나리오(1/2/8)만 골라 1→8단계를 전부 실제로 돌리고,
judge()/explain()에 들어가는 Claim/ComputedResult의 세부 필드(batch_runner.py의 콘솔 출력에는
claim.sentence/claim_type만 찍히고 value/unit/period 등은 안 보임)까지 그대로 노출해서
7·8단계에서 보완할 점을 찾기 위한 진단 테스트입니다.

구조적 불변조건(예외 없이 끝까지 도는지, verdict/gap_type이 유효한 값인지, 판단불가에
limitation이 반드시 있는지)은 PASS/FAIL로 검사합니다. 반면 claim_type과 computed.calc_type이
실제로 같은 "종류"의 값인지(예: 수준값 vs 변화율, 기간 표현을 수치로 오인)는 코드가 아직
옳고 그름을 판단할 수 없는 영역이라(tests/pipeline_integration_log.md 이슈 4 참고) fail로
막지 않고 WARNING으로만 표시합니다 — judge.py를 고칠지 말지는 사람이 보고 정하는 몫입니다.

실제 HCX API + KOSIS API를 호출합니다 (.env에 HCX_API_KEY, KOSIS_API_KEY 필요).

실행 (프로젝트 루트에서):
    python -m tests.test_stage_7_8_diagnostics
"""

from __future__ import annotations

import json
import re

from agent.mapping.embedding_search import build_table_embedding_cache, embedding_search
from agent.mapping.keyword_search import keyword_search
from agent.mapping.reranker import search_and_rerank
from agent.kosis.api_client import KosisApiClient
from agent.kosis.calculator import KosisCalculator
from agent.preprocessing.classifier import classify
from agent.preprocessing.claim_extractor import extract_claims
from agent.pipeline.batch_runner import (
    ARTICLES,
    TABLE_PARAMS_PATH,
    _load_table_catalog_by_id,
    run_stage_4,
    run_stage_5_6,
    run_stage_7_8,
)

VALID_VERDICTS = {"일치", "불일치", "판단불가"}
VALID_GAP_TYPES = {None, "수치", "기간", "모집단", "과장표현"}

# 나머지 시나리오는 3~4단계(표매칭 신뢰도 낮음/되묻기 미해결)에서 막혀서 judge/explain을
# 아예 안 타므로 이 진단 테스트 대상이 아니다 (batch_runner.py 실행 결과로 확인됨).
TARGET_LABEL_PREFIXES = ["시나리오 1 ", "시나리오 2 ", "시나리오 8 "]

_WARNINGS: list[str] = []


class _Env:
    """실제 API 클라이언트/카탈로그/캐시를 한 번만 만들어서 케이스끼리 공유."""

    def __init__(self) -> None:
        self.client = KosisApiClient()
        self.calculator = KosisCalculator()
        with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
            self.table_params = json.load(f)
        self.catalog_by_id = _load_table_catalog_by_id()
        self.embedding_cache = build_table_embedding_cache()


_ENV: _Env


def _run_claim_through_stage_8(article: dict, claim) -> list[dict]:
    """claim 하나를 3→4→5→6→7→8단계로 실제로 돌리고 진단 정보를 반환한다 (막히면 빈 리스트)."""
    candidates = search_and_rerank(
        claim,
        keyword_fn=keyword_search,
        embedding_fn=lambda c: embedding_search(c, cache=_ENV.embedding_cache),
        document_texts={tid: t["embedding_text"] for tid, t in _ENV.catalog_by_id.items()},
    )
    if not candidates:
        return []
    top = candidates[0]
    if top.source_meta and "unverified" in top.source_meta:
        return []  # batch_runner.py와 동일하게 검증 안 된 매칭은 판정 자체를 스킵

    slots = run_stage_4(claim.sentence, article.get("clarify_reply"), article["published_date"])
    if slots is None:
        return []

    computed = run_stage_5_6(top.table_id, slots, _ENV.table_params, _ENV.client, _ENV.calculator)
    if computed is None:
        return []

    outcome = run_stage_7_8(claim, top, computed)
    if outcome is None:
        return []

    verdict, explanation = outcome
    return [
        {
            "claim": claim,
            "table_id": top.table_id,
            "computed": computed,
            "verdict": verdict,
            "explanation": explanation,
        }
    ]


def _run_article_to_stage_8(article: dict) -> list[dict]:
    cls_result = classify(article["article_text"])
    if not cls_result.label:
        return []
    claims = extract_claims(article["article_text"])

    diagnostics: list[dict] = []
    for claim in claims:
        diagnostics.extend(_run_claim_through_stage_8(article, claim))
    return diagnostics


def _print_diag(diag: dict) -> None:
    claim = diag["claim"]
    computed = diag["computed"]
    verdict = diag["verdict"]
    explanation = diag["explanation"]
    print(f"\n  주장: \"{claim.sentence}\"")
    print(
        f"    claim: claim_type={claim.claim_type} value={claim.value} unit={claim.unit} "
        f"period={claim.period} comparison_operator={claim.comparison_operator}"
    )
    print(f"    table_id={diag['table_id']}")
    print(f"    computed: {computed}")
    print(f"    verdict: {verdict}")
    if explanation is not None:
        print(f"    explanation.limitation: {explanation.limitation}")
    else:
        print("    explanation: None (8단계 실패)")


def _assert_structural_invariants(diag: dict) -> None:
    verdict = diag["verdict"]
    explanation = diag["explanation"]
    assert verdict.verdict in VALID_VERDICTS, f"알 수 없는 verdict 값: {verdict.verdict!r}"
    assert verdict.gap_type in VALID_GAP_TYPES, f"알 수 없는 gap_type 값: {verdict.gap_type!r}"
    if verdict.verdict == "판단불가":
        assert explanation is not None and explanation.limitation, (
            f"판단불가인데 8단계 explanation에 limitation이 없음 (얼버무림) — "
            f"claim=\"{diag['claim'].sentence}\""
        )
    if explanation is not None:
        assert explanation.explanation_text, "8단계 explanation_text가 비어있음"


def _flag_semantic_issues(diag: dict) -> None:
    """옳고 그름을 여기서 확정하지 않고, 사람이 봐야 할 의심 지점만 모아둔다."""
    claim = diag["claim"]
    computed = diag["computed"]
    verdict = diag["verdict"]

    # claim_type(기사가 주장하는 종류)과 computed.calc_type(실제 계산된 종류)이 다른데도
    # 그대로 숫자 비교됐는지 — 이슈 4(pipeline_integration_log.md)와 같은 유형.
    if claim.claim_type == "규모" and computed.calc_type in ("증감", "증감률"):
        _WARNINGS.append(
            f"[claim_type/calc_type 불일치] \"{claim.sentence}\" — claim_type=규모(수준값 주장)인데 "
            f"computed.calc_type={computed.calc_type}(변화율)와 비교됨 → 종류가 다른 값을 직접 "
            f"비교했을 가능성 (verdict.reason: {verdict.reason})"
        )

    # claim.value가 "OO개월" 같은 기간 표현에서 그대로 온 것으로 보이는지 (46개월 → 46 오인 등).
    if claim.value is not None:
        m = re.search(rf"{int(claim.value)}\s*개월", claim.sentence)
        if m:
            _WARNINGS.append(
                f"[기간 표현 오인 의심] \"{claim.sentence}\" — claim.value={claim.value}가 문장 중 "
                f"\"{m.group(0)}\"(기간 표현)에서 온 것일 수 있음 (verdict.reason: {verdict.reason})"
            )

    # claim은 월 단위(예: "전년 동월 대비")인데 computed.period는 연 단위로 보이고, 판정
    # 근거에 시점/기간 언급이 전혀 없는 경우 — 이슈 1-2가 조용히 묻혔을 가능성.
    if claim.period and "월" in claim.period and "개월" not in claim.period:
        computed_period = computed.period or ""
        if "월" not in computed_period and not any(k in verdict.reason for k in ("시점", "기간")):
            _WARNINGS.append(
                f"[기간 단위 불일치 미반영 의심] \"{claim.sentence}\" — claim.period={claim.period!r}"
                f"(월 단위로 보임) vs computed.period={computed_period!r}(연 단위로 보임)인데 "
                f"verdict.reason에 시점/기간 관련 언급이 없음: {verdict.reason}"
            )


def _run_case(label_prefix: str) -> None:
    article = next(a for a in ARTICLES if a["label"].startswith(label_prefix))
    diagnostics = _run_article_to_stage_8(article)
    assert diagnostics, (
        f"'{label_prefix}'가 7·8단계까지 도달한 주장을 하나도 못 만들었음 "
        "(3~6단계 배선이 끊겼을 수 있음 — 이전엔 도달했던 시나리오임)"
    )
    for diag in diagnostics:
        _print_diag(diag)
        _assert_structural_invariants(diag)
        _flag_semantic_issues(diag)


def case_scenario_1_youth_unemployment() -> None:
    _run_case("시나리오 1 ")


def case_scenario_2_cpi() -> None:
    _run_case("시나리오 2 ")


def case_scenario_8_births() -> None:
    _run_case("시나리오 8 ")


CASES = [
    case_scenario_1_youth_unemployment,
    case_scenario_2_cpi,
    case_scenario_8_births,
]


def main() -> None:
    global _ENV
    _ENV = _Env()

    results = []
    for case in CASES:
        print(f"\n{'=' * 70}")
        print(f"[실행] {case.__name__}")
        print(f"{'=' * 70}")
        try:
            case()
            results.append((case.__name__, "PASS", ""))
        except Exception as e:
            results.append((case.__name__, "FAIL", f"{type(e).__name__}: {e}"))

    print(f"\n{'=' * 70}")
    print(f"총 {len(results)}건 실행")
    print(f"{'=' * 70}")
    for name, status, detail in results:
        mark = "✅" if status == "PASS" else "❌"
        line = f"{mark} {status}  {name}"
        if detail:
            line += f"  — {detail}"
        print(line)

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n{n_pass}/{len(results)} PASS")

    print(f"\n{'=' * 70}")
    print(f"7·8단계 보완 필요 의심 지점 (WARNING, {len(_WARNINGS)}건) — fail은 아니지만 사람 검토 필요")
    print(f"{'=' * 70}")
    if not _WARNINGS:
        print("(없음)")
    for w in _WARNINGS:
        print(f"  - {w}")


if __name__ == "__main__":
    main()
