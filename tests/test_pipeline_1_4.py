"""
tests/test_pipeline_1_4.py — 1~4단계 파이프라인 연결(agent/pipeline/pipeline_1_4.py) 단위 테스트

agent.mapping.embedding_search/reranker(e5-large + BGE reranker)를 한 프로세스에서 같이
로딩하면 이 환경에서 세그폴트가 재현되는 문제(2026-08-04 확인, transformers 5.14.1 버전
이슈로 추정 — 원인 규명은 보류 중)가 있어서, 3단계(search_and_rerank)는 실제 모델을 호출
하지 않고 미리 정해둔 TableCandidate로 모킹한다. 4단계 슬롯필링(fill_slots/dimension_hints/
LLM 2차)은 실제 HCX API를 그대로 호출한다 — 여기는 임베딩/리랭커 모델과 무관해서 영향 없음.

실행 (프로젝트 루트에서, HCX_API_KEY 필요):
    python -m tests.test_pipeline_1_4
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

from agent.interfaces import Claim, TableCandidate
import agent.pipeline.pipeline_1_4 as pipeline_1_4

TABLE_PARAMS_PATH = Path(__file__).parent.parent / "agent" / "kosis" / "table_params.json"
with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
    TABLE_PARAMS = json.load(f)

ARTICLE_DATE = date(2025, 1, 6)


def _resolve(sentence: str, table_id: str, table_name: str = "테스트표") -> pipeline_1_4.Stage4Result:
    """search_and_rerank를 모킹해서 지정한 table_id가 항상 최상위 후보로 나온 것처럼 만든다."""
    candidate = TableCandidate(
        table_id=table_id, table_name=table_name, score=0.9, required_slots=[], source_meta="keyword"
    )
    claim = Claim(sentence=sentence, claim_type="규모")
    with patch.object(pipeline_1_4, "search_and_rerank", return_value=[candidate]):
        return pipeline_1_4.resolve_claim_1_to_4(
            claim, ARTICLE_DATE, embedding_cache={}, document_texts={}, table_params=TABLE_PARAMS
        )


def case_01_no_candidates_means_no_table_match():
    claim = Claim(sentence="아무 표에도 안 걸리는 문장", claim_type="규모")
    with patch.object(pipeline_1_4, "search_and_rerank", return_value=[]):
        result = pipeline_1_4.resolve_claim_1_to_4(
            claim, ARTICLE_DATE, embedding_cache={}, document_texts={}, table_params=TABLE_PARAMS
        )
    assert result.status == "3단계_매칭없음", result.status


def case_02_unverified_embedding_only_match_is_flagged():
    candidate = TableCandidate(
        table_id="DT_1DA7102S", table_name="성/연령별 실업률", score=0.4,
        required_slots=[], source_meta="embedding-only, unverified",
    )
    claim = Claim(sentence="아무 문장", claim_type="규모")
    with patch.object(pipeline_1_4, "search_and_rerank", return_value=[candidate]):
        result = pipeline_1_4.resolve_claim_1_to_4(
            claim, ARTICLE_DATE, embedding_cache={}, document_texts={}, table_params=TABLE_PARAMS
        )
    assert result.status == "3단계_매칭_불충분", result.status


def case_03_period_missing_blocks_stage5():
    """DT_1DA7102S는 region 축이 없어서(gender/age뿐) region 미해결은 안 걸려야 하지만,
    period가 아예 없으면 4단계 미해결이어야 한다."""
    result = _resolve("청년 실업률이 심각한 수준이다", "DT_1DA7102S")
    assert result.status == "4단계_미해결", result.status
    assert "period" in result.missing_slots, result.missing_slots
    assert "region" not in result.missing_slots, "DT_1DA7102S엔 region 축 자체가 없음"


def case_04_full_slots_ready_for_stage5():
    result = _resolve("2024년 청년 실업률이 6%에 육박했다", "DT_1DA7102S")
    assert result.status == "5단계_진행가능", (result.status, result.missing_slots)
    assert result.slots.get("period") == "2024"
    assert result.dimension_hints.get("age") == "청년(15~29세)"
    assert result.kosis_slots is not None


def case_05_comparison_keyword_without_calc_type_is_unresolved():
    """'대비'(비교 의도 키워드)가 있는데 calc_type을 못 뽑으면 미해결이어야 한다."""
    result = _resolve("2024년 전년 대비 청년 실업률 수치다", "DT_1DA7102S")
    if result.status == "4단계_미해결":
        assert "calc_type" in result.missing_slots, result.missing_slots
    else:
        # LLM이 '증감률' 등으로 calc_type을 실제로 뽑아낸 경우도 정상이라 실패로 안 봄
        assert result.slots.get("calc_type"), "비교 의도인데 calc_type도 안 뽑히고 미해결도 아님"


def case_06_region_required_when_no_safe_national_default():
    """DT_1R11006_FRM101의 country 축은 default_value가 어떤 나라 라벨과도 안 맞아
    (안전한 "전국류" 기본값이 없음) 국가 언급이 없으면 여전히 미해결이어야 한다."""
    result = _resolve("2024년 수출액이 크게 늘었다", "DT_1R11006_FRM101")
    assert result.status == "4단계_미해결", (result.status, result.missing_slots)
    assert "region" in result.missing_slots, result.missing_slots


def case_06b_region_not_required_when_default_is_national():
    """DT_1B04005N의 region 기본값은 "전국"이라, 지역 언급이 없어도 조용히 전국으로
    채워져서 5단계로 진행 가능해야 한다 (2026-08-05 실제 기사 테스트로 발견한 버그 수정
    — "2020년 인구주택총조사 응답률 96.3%"처럼 전국 단위 통계가 지역 미언급을 이유로
    불필요하게 미해결 처리되던 것)."""
    result = _resolve("2024년 인구는 630만명이었다", "DT_1B04005N")
    assert "region" not in result.missing_slots, result.missing_slots


def case_07_decade_age_detected_as_special_resolution():
    result = _resolve("2024년 전국 20대 인구는 630만명이었다", "DT_1B04005N")
    assert result.special_resolution == "나이대_다중코드", result.special_resolution


def case_08_since_event_extremum_detected():
    result = _resolve("코로나 이후 청년 실업률이 최고치를 기록했다", "DT_1DA7102S")
    assert result.special_resolution == "극값_이벤트기준", result.special_resolution


def case_09_all_time_extremum_detected():
    result = _resolve("작년 수출액이 역대 최대를 기록했다", "DT_1DA7102S")
    assert result.special_resolution == "극값_역대", result.special_resolution


def case_10_extremum_detected_does_not_require_calc_type():
    """극값(N년만에 등)이 감지됐으면 "증가" 같은 비교 키워드가 있어도 calc_type 미해결로
    막으면 안 된다 (2026-08-05 실제 기사 "10년 만에 처음이다"에서 발견한 통합 누락 수정)."""
    result = _resolve("2024년 청년 실업률 증가는 8년 만에 처음이다", "DT_1DA7102S")
    assert result.special_resolution == "극값_N년만에", result.special_resolution
    assert "calc_type" not in result.missing_slots, result.missing_slots
    assert result.status == "5단계_진행가능", (result.status, result.missing_slots)


def case_11_relative_month_without_year_resolves_via_article_date():
    """"지난 10월"처럼 연도 없이 상대적으로 월만 가리키는 표현은 article_date(2025-01-06)
    기준으로 "2024년 10월"로 계산돼야 한다 (2026-08-05 실제 배치에서 이런 표현이 계속
    period 미해결로 빠지는 걸 확인하고 추가한 _resolve_relative_month_period 회귀 테스트)."""
    result = _resolve("지난 10월 청년 실업률이 6%에 육박했다", "DT_1DA7102S")
    assert result.slots.get("period") == "202410", result.slots
    assert result.status == "5단계_진행가능", (result.status, result.missing_slots)


CASES = [
    case_01_no_candidates_means_no_table_match,
    case_02_unverified_embedding_only_match_is_flagged,
    case_03_period_missing_blocks_stage5,
    case_04_full_slots_ready_for_stage5,
    case_05_comparison_keyword_without_calc_type_is_unresolved,
    case_06_region_required_when_no_safe_national_default,
    case_06b_region_not_required_when_default_is_national,
    case_07_decade_age_detected_as_special_resolution,
    case_08_since_event_extremum_detected,
    case_09_all_time_extremum_detected,
    case_10_extremum_detected_does_not_require_calc_type,
    case_11_relative_month_without_year_resolves_via_article_date,
]


def main() -> None:
    print(f"총 {len(CASES)}건 실행 (3단계는 모킹, 4단계는 실제 HCX API 호출)")
    print("=" * 70)
    results = []
    for case in CASES:
        try:
            case()
            results.append((case.__name__, "PASS", ""))
        except Exception as e:  # noqa: BLE001 - 테스트 러너라 실패 원인만 보고 계속 진행
            results.append((case.__name__, "FAIL", str(e)))

    for name, status, detail in results:
        mark = "✅" if status == "PASS" else "❌"
        print(f"{mark} {status}  {name}" + (f" — {detail}" if detail else ""))

    passed = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n{passed}/{len(results)} PASS")


if __name__ == "__main__":
    main()