"""
agent/pipeline/verify_1_3.py — 1→2→3단계(분류→주장추출→통계표매핑)만 연결해서 검증

pipeline_1_4.py는 3단계 뒤에 4단계(슬롯필링/되묻기)까지 이어붙이는데, 4단계는 아직
확인 대상이 아니라서 3단계(표 매핑) 결과까지만 보고 멈추는 가벼운 버전.
pipeline_1_4.py의 기사 로딩 로직(load_articles_from_csv)은 그대로 재사용한다.

⚠️ 2.5단계: source_filter.py 연결 (2026-08-05)
    1~3단계를 실제로 이어붙여 돌려보니, 2단계가 뽑은 claim 중 상당수가 애초에 KOSIS로
    검증 불가능한 출처(외국 정부기관, 화이트리스트에 없는 민간·공공기관 등)라 3단계가
    "매칭_불충분"으로 떨어뜨리는 경우가 많았다. B의 3단계 자체 골든셋 평가(~90%)는
    "KOSIS에 대응 표가 있다고 이미 확인된 claim"만 대상이라 이 케이스들이 애초에
    빠져있었는데, 실제 파이프라인은 이런 것까지 다 3단계로 넘기고 있었던 것.

    source_filter.py(2단계 이후 출처기관 검증 필터, 이미 작성돼 있었지만 어디에도
    연결 안 돼 있던 모듈)를 2→3단계 사이에 끼워 넣어서, KOSIS로 검증 불가능한 출처의
    claim은 3단계로 보내지 않고 미리 걸러낸다. 40건 샘플 실측: 52개 claim 중 15개
    (28.8%, 중국 국가통계국/한국농수산식품유통공사/출처불명)가 필터링됨.

시각화용 구조화 출력 (2026-08-05):
    기사별/claim별 단계 통과 현황을 JSON으로 덤프해서 시각화 아티팩트에 바로 쓸 수
    있게 한다 (--json <경로> 옵션, 기본값 없으면 저장 안 함).

실행 (프로젝트 루트에서):
    python -m agent.pipeline.verify_1_3                          # ARTICLES 시나리오
    python -m agent.pipeline.verify_1_3 --csv                    # data_set.csv 실제 기사 샘플
    python -m agent.pipeline.verify_1_3 --csv --json out.json    # 결과를 JSON으로도 저장
"""

from __future__ import annotations

import json
import sys

from agent.preprocessing.classifier import classify, ClassifierError
from agent.preprocessing.claim_extractor import extract_claims, ClaimExtractorError
from agent.preprocessing.source_filter import resolve_claim_sources, classify_source
from agent.mapping.keyword_search import keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import search_and_rerank, load_document_texts
from agent.pipeline.pipeline_1_4 import load_articles_from_csv


def _compute_outcome(record: dict) -> str:
    """기사 하나가 파이프라인 어디서 최종적으로 멈췄는지 한 단어로 요약한다
    (시각화에서 1단계 위주가 아니라 2·3단계 탈락도 한눈에 보이게 하기 위함)."""
    if not record["stage1_passed"]:
        return "1단계_오류" if record.get("stage1_error") else "1단계_탈락"
    if record["stage2_status"] == "실패":
        return "2단계_실패"
    if record["stage2_status"] == "0건":
        return "2단계_0건추출"
    claims = record["claims"]
    if claims and all(c["stage"] == "2.5_필터링됨" for c in claims):
        return "2.5단계_전부필터링"
    stage3_claims = [c for c in claims if c["stage"] != "2.5_필터링됨"]
    if not stage3_claims:
        return "2.5단계_전부필터링"
    n_matched = sum(1 for c in stage3_claims if c["stage"] == "3_매칭됨")
    if n_matched == 0:
        return "3단계_전부실패"
    if n_matched == len(stage3_claims):
        return "3단계_전부매칭"
    return "3단계_일부매칭"


def run_article_1_3(article: dict, *, embedding_cache: dict, document_texts: dict) -> dict:
    """기사 1건을 1→2→2.5→3단계까지 돌려서, 시각화에 바로 쓸 수 있는 구조화 레코드를 반환한다."""
    title = article.get("article_title") or article.get("label", "(제목 없음)")
    print(f"\n{'=' * 60}")
    print(title)

    record: dict = {
        "title": title,
        "article_url": article.get("article_url"),
        "stage1_attempted": True,
        "stage1_passed": False,
        "stage1_score": None,
        "stage1_reason": None,
        "stage1_error": None,
        "stage2_status": "미시도",  # 미시도 | 실패 | 0건 | 성공
        "stage2_error": None,
        "claims": [],
    }

    try:
        cls_result = classify(article["article_text"])
    except ClassifierError as e:
        print(f"[1단계] 분류 실패 ({e}) → 기사 스킵")
        record["stage1_reason"] = f"실패: {e}"
        record["stage1_error"] = str(e)
        record["outcome"] = _compute_outcome(record)
        return record

    record["stage1_score"] = cls_result.score
    record["stage1_reason"] = cls_result.reason

    if not cls_result.label:
        print(f"[1단계] 무관한 기사로 판정(score={cls_result.score:.2f}) → 스킵")
        record["outcome"] = _compute_outcome(record)
        return record
    record["stage1_passed"] = True
    print(f"[1단계] 관련 기사 판정(score={cls_result.score:.2f})")

    try:
        claims = extract_claims(article["article_text"])
    except ClaimExtractorError as e:
        print(f"[2단계] 주장 추출 실패 ({e}) → 기사 스킵")
        record["stage2_status"] = "실패"
        record["stage2_error"] = str(e)
        record["outcome"] = _compute_outcome(record)
        return record
    print(f"[2단계] 주장 {len(claims)}건 추출")
    record["stage2_status"] = "0건" if len(claims) == 0 else "성공"
    record["stage2_claim_count"] = len(claims)

    # --- 2.5단계: 출처기관 검증 필터 (source_filter.py) ---
    claims = resolve_claim_sources(claims, classifier_reason=cls_result.reason)
    verifiable_claims = []
    for claim in claims:
        verdict = classify_source(claim.source_org)
        if verdict == "kosis_verified":
            verifiable_claims.append(claim)
        else:
            print(
                f"  - \"{claim.sentence[:50]}\" → [2.5단계] 출처필터링됨 "
                f"(source_org={claim.source_org!r}, 판정={verdict})"
            )
            record["claims"].append(
                {
                    "sentence": claim.sentence,
                    "source_org": claim.source_org,
                    "stage": "2.5_필터링됨",
                    "detail": verdict,
                }
            )

    for claim in verifiable_claims:
        claim_record = {"sentence": claim.sentence, "source_org": claim.source_org}
        try:
            candidates = search_and_rerank(
                claim,
                keyword_fn=keyword_search,
                embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
                top_k=3,
                document_texts=document_texts,
            )
        except Exception as e:  # noqa: BLE001 - 점검 스크립트, 계속 진행
            print(f"  - \"{claim.sentence[:50]}\" → [3단계] 오류: {type(e).__name__}: {e}")
            claim_record.update(stage="3_오류", detail=str(e))
            record["claims"].append(claim_record)
            continue

        if not candidates:
            print(f"  - \"{claim.sentence[:50]}\" → [3단계] 매칭없음")
            claim_record.update(stage="3_매칭없음")
            record["claims"].append(claim_record)
            continue

        top = candidates[0]
        if top.source_meta and "unverified" in top.source_meta:
            print(f"  - \"{claim.sentence[:50]}\" → [3단계] 매칭_불충분 (임베딩 전용, 신뢰도 낮음)")
            claim_record.update(
                stage="3_매칭불충분", table_id=top.table_id, table_name=top.table_name, score=top.score
            )
            record["claims"].append(claim_record)
            continue

        print(
            f"  - \"{claim.sentence[:50]}\" → [3단계] 매칭됨: "
            f"{top.table_name}({top.table_id}) score={top.score:.3f}"
        )
        claim_record.update(
            stage="3_매칭됨", table_id=top.table_id, table_name=top.table_name, score=top.score
        )
        record["claims"].append(claim_record)

    record["outcome"] = _compute_outcome(record)
    return record


def print_summary(all_records: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print("1~3단계 파이프라인 완성도 요약")
    print(f"{'=' * 60}")

    n_articles = len(all_records)
    n_passed_stage1 = sum(1 for r in all_records if r["stage1_passed"])
    all_claims = [c for r in all_records for c in r["claims"]]
    total = len(all_claims)

    print(f"\n기사 {n_articles}건 중 1단계 통과 {n_passed_stage1}건")

    outcome_counts: dict[str, int] = {}
    for r in all_records:
        outcome_counts[r["outcome"]] = outcome_counts.get(r["outcome"], 0) + 1
    print("\n[기사별 최종 도달 단계]")
    for outcome, n in sorted(outcome_counts.items()):
        print(f"  {outcome}: {n}건 ({n / n_articles * 100:.1f}%)")

    print(f"\n표 매핑 시도된 주장: {total}건")
    if total == 0:
        return

    counts: dict[str, int] = {}
    for c in all_claims:
        counts[c["stage"]] = counts.get(c["stage"], 0) + 1
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}건 ({n / total * 100:.1f}%)")


def main(use_csv_sample: bool = False, csv_n: int = 15, json_out: str | None = None) -> None:
    embedding_cache = build_table_embedding_cache()
    document_texts = load_document_texts()

    if use_csv_sample:
        articles = load_articles_from_csv(n=csv_n)
    else:
        from agent.pipeline.batch_runner import ARTICLES  # 시나리오 재사용(읽기 전용)

        articles = ARTICLES

    all_records: list[dict] = []
    for article in articles:
        all_records.append(
            run_article_1_3(article, embedding_cache=embedding_cache, document_texts=document_texts)
        )

    print_summary(all_records)

    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"\n[JSON 저장] {json_out}")


if __name__ == "__main__":
    _json_out = None
    if "--json" in sys.argv:
        _idx = sys.argv.index("--json")
        _json_out = sys.argv[_idx + 1] if _idx + 1 < len(sys.argv) else None
    main(use_csv_sample="--csv" in sys.argv, json_out=_json_out)
