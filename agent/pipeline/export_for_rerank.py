"""
agent/pipeline/export_for_rerank.py — 1~3단계(키워드+64개 카탈로그 임베딩 병합까지)를
로컬에서 돌리고, VDB 조회+리랭킹은 코랩에 넘긴다.

배경: 리랭커(BAAI/bge-reranker-v2-m3, 568M)는 로컬(RAM 7.4GB)에서 로드 자체가
세그폴트라 코랩이 꼭 필요하다. 임베딩(e5-small, 64개 카탈로그용)은 claim마다 반복
호출하면 세그폴트가 나지만(2026-08-13 실측), 여러 문장을 한 번의 encode() 호출에
모아서 배치로 처리하면 로컬에서도 안전하다(2026-08-15 실측). 그래서 파이프라인을
이렇게 나눈다:
  1) (이 스크립트, 로컬) 분류→claim 추출→출처필터→키워드매칭→(배치)임베딩(64개 카탈로그)
     까지 수행하고, merged_candidates(키워드+임베딩 후보, VDB는 아직 안 섞임)를
     claim별로 만들어 저장
  2) (코랩, notebooks/reranker_colab.ipynb) VDB(KOSIS 표 28만7천여 개) 조회를 마저
     수행해 merged_candidates에 합친 뒤 리랭킹까지 수행
  3) (resume_after_rerank.py, 로컬) 최종 결과를 받아 4~8단계 마저 실행

2026-08-19: VDB 조회를 로컬에서 코랩으로 옮겼다 — VDB 임베딩 모델이 Qwen3-Embedding-4B
(4B 파라미터)로 바뀌면서 로컬(RAM 7.4GB)에서 아예 못 돌게 됐다(리랭커 568M도 이미
세그폴트였는데 그보다 7배 큼). 그래서 이 스크립트는 이제 VDB를 건드리지 않고, 64개
카탈로그(keyword+embedding)까지만 로컬에서 병합해서 내보낸다.

batch_runner.py의 run_article()과 최대한 같은 코드를 재사용한다 — 이 스크립트가 하는
1~2단계 로직(분류/추출/필터/전망·해외국가 즉시판정/키워드매칭)은 run_article()과 동일해야
나중에 결과가 갈라지지 않는다.

사용법 (프로젝트 루트에서):
    python -m agent.pipeline.export_for_rerank --csv --csv-n 30
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from agent.preprocessing.classifier import classify
from agent.preprocessing.claim_extractor import extract_claims, recover_missed_claims
from agent.preprocessing.source_filter import resolve_claim_sources, filter_verifiable_claims
from agent.mapping.keyword_search import keyword_search
from agent.mapping.embedding_search import batch_embedding_search, build_table_embedding_cache
from agent.mapping.reranker import _merge_candidates
from agent.orchestrator.calc_type_router import _mentions_foreign_country
from agent.interfaces import Claim, TableCandidate, Verdict
from db.store import insert_verification

from agent.pipeline.batch_runner import (
    ARTICLES,
    DEFAULT_CLARIFY_REPLY,
    _build_verification_record,
    _load_table_catalog_by_id,
    load_articles_from_csv,
)

PENDING_PATH = Path(__file__).parent.parent.parent / "data" / "rerank_pending.json"


def _article_to_json(article: dict) -> dict:
    out = dict(article)
    if isinstance(out.get("published_date"), date):
        out["published_date"] = out["published_date"].isoformat()
    return out


def export_article(
    article: dict,
    catalog_by_id: dict,
    pending_items: list[dict],
) -> None:
    """기사 하나를 1~2단계+키워드매칭까지 처리한다 (임베딩·리랭킹은 코랩에서).

    전망/해외국가 즉시판정 claim은 run_article()과 동일하게 여기서 바로 DB에 저장하고
    끝낸다(리랭킹 대상이 아니므로 pending에 안 넣음). 나머지 claim은 keyword_search로 나온
    후보 목록을 pending_items에 추가한다 — 임베딩 매칭은 코랩 쪽에서 카탈로그 전체를
    받아 직접 수행한다(_merge_candidates도 코랩에서 재현).
    """
    print(f"\n{'=' * 60}")
    print(article["label"])
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
        claims = recover_missed_claims(article["article_text"], claims)
        print(f"[2단계 claim_extractor] {len(claims)}개 주장 추출")
    except Exception as e:
        print(f"[2단계 claim_extractor] 실패 ({type(e).__name__}: {e}) → 이 기사 스킵")
        return

    claims = resolve_claim_sources(claims, cls_result.reason)
    before_filter = len(claims)
    claims = filter_verifiable_claims(claims)
    if before_filter != len(claims):
        print(f"[2단계 출처 필터] {before_filter}개 중 {before_filter - len(claims)}개 제외 (KOSIS 미검증 출처)")

    article_json = _article_to_json(article)

    for i, claim in enumerate(claims):
        print(f"{'-' * 60}")
        print(f"주장: \"{claim.sentence}\" (claim_type={claim.claim_type})")

        if claim.claim_type == "전망":
            print("[분류] claim_type='전망'(미래 예측) → 즉시 판단불가 처리 (run_article과 동일)")
            verdict = Verdict(verdict="판단불가", gap_type=None, reason="미래 예측 주장은 공식 통계로 검증 불가")
            try:
                insert_verification(
                    _build_verification_record(
                        article=article, claim=claim, top=None, generic_slots=None,
                        table_params=None, computed=None, verdict=verdict, explanation=None,
                        cls_result=cls_result, catalog_by_id=catalog_by_id,
                        verification_possible="불가", ambiguity_reason="미래 예측 주장(claim_type=전망)은 검증 대상 아님",
                    )
                )
            except Exception as e:
                print(f"[DB 저장] 실패 ({type(e).__name__}: {e}) → 저장만 스킵")
            continue

        if _mentions_foreign_country(claim.population, claim.region, claim.comparison_target):
            print("[분류] 해외 국가 포함 → 즉시 판단불가 처리 (run_article과 동일)")
            verdict = Verdict(verdict="판단불가", gap_type=None, reason="해외 국가/지역 통계는 KOSIS(국내 통계)로 검증 불가")
            try:
                insert_verification(
                    _build_verification_record(
                        article=article, claim=claim, top=None, generic_slots=None,
                        table_params=None, computed=None, verdict=verdict, explanation=None,
                        cls_result=cls_result, catalog_by_id=catalog_by_id,
                        verification_possible="불가", ambiguity_reason="해외 국가/지역 데이터는 KOSIS 검증 대상 아님",
                    )
                )
            except Exception as e:
                print(f"[DB 저장] 실패 ({type(e).__name__}: {e}) → 저장만 스킵")
            continue

        try:
            kw_results = keyword_search(claim)
        except Exception as e:
            print(f"[3단계 키워드매칭] 실패 ({type(e).__name__}: {e}) → 이 주장 스킵")
            continue

        # keyword_search가 0건이어도 그대로 내보낸다 — 뒤에서 임베딩/VDB로 후보를 찾을 수도
        # 있으므로, 여기서 "매칭 없음"으로 확정하지 않는다. embedding/vdb 병합은 전체 기사를
        # 다 돌고 나서 한 번의 배치 호출로 처리한다(claim마다 반복 호출하면 세그폴트 위험).
        print(f"[3단계 키워드매칭] {len(kw_results)}개 후보 (임베딩·VDB 병합은 배치로 나중에)")
        pending_items.append(
            {
                "item_id": f"{article['label']}::{i}",
                "article": article_json,
                "claim_obj": claim,
                "cls_result": dataclasses.asdict(cls_result),
                "keyword_candidates": kw_results,
            }
        )


def _merge_all_candidates(pending_items: list[dict]) -> None:
    """전체 claim의 임베딩(64개 카탈로그)을 한 번의 배치 호출로 처리하고, keyword/embedding
    후보를 claim마다 병합해 pending_items에 채워 넣는다(claim마다 반복 호출하면 세그폴트
    위험 — 2026-08-13/15 실측). VDB(28만7천여 개)는 여기서 안 합친다 — 2026-08-19,
    VDB 임베딩 모델(Qwen3-Embedding-4B)이 로컬에서 못 돌아서 코랩(reranker_colab.ipynb)
    으로 옮겼다."""
    if not pending_items:
        return

    sentences = [item["claim_obj"].sentence for item in pending_items]

    print(f"\n[배치 임베딩] claim {len(sentences)}건을 한 번에 인코딩 중(64개 카탈로그 비교용)...")
    cache = build_table_embedding_cache()
    emb_results_list = batch_embedding_search(sentences, cache=cache)
    print("[배치 임베딩] 완료")

    for item, emb_results in zip(pending_items, emb_results_list):
        merged = _merge_candidates(item["keyword_candidates"], emb_results, None)
        item["merged_candidates"] = [
            {
                "table_id": c.table_id,
                "table_name": c.table_name,
                "score": c.score,
                "required_slots": c.required_slots,
                "source_meta": c.source_meta,
            }
            for c in merged
        ]


def main(use_csv_sample: bool = False, csv_n: int = 15, csv_seed: int = 42) -> None:
    catalog_by_id = _load_table_catalog_by_id()

    articles = load_articles_from_csv(n=csv_n, seed=csv_seed) if use_csv_sample else ARTICLES

    pending_items: list[dict] = []
    for article in articles:
        export_article(article, catalog_by_id, pending_items)

    _merge_all_candidates(pending_items)

    # 카탈로그 표별 embedding_text — 코랩 리랭킹 단계에서 문서 텍스트로 필요하다(짧은
    # table_name만 넘기면 리랭커 성능이 떨어짐, test_mapping.py에서 이미 확인된 사실).
    catalog_export = {
        tid: {"table_name": t.get("title", tid), "embedding_text": t.get("embedding_text", t.get("title", tid))}
        for tid, t in catalog_by_id.items()
    }

    # claim_obj(Claim 객체)는 JSON으로 못 그대로 직렬화하니 dict로 바꾸고, keyword_candidates도
    # 이제 merged_candidates 안에 다 들어있으니 중복으로 안 남긴다.
    serializable_items = []
    for item in pending_items:
        serializable_items.append(
            {
                "item_id": item["item_id"],
                "article": item["article"],
                "claim": dataclasses.asdict(item["claim_obj"]),
                "cls_result": item["cls_result"],
                "merged_candidates": item.get("merged_candidates", []),
            }
        )

    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(
        json.dumps({"catalog": catalog_export, "items": serializable_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[export] 리랭킹 대기 claim {len(serializable_items)}건 + 카탈로그 {len(catalog_export)}개 표를 "
          f"{PENDING_PATH}에 저장했습니다. (merged_candidates에 keyword+embedding 후보만 포함됨, VDB는 코랩에서 추가)")
    print("다음: 이 파일을 코랩(notebooks/reranker_colab.ipynb)에 업로드하면 VDB 조회+리랭킹까지 실행됩니다.")


def _parse_int_flag(argv: list[str], flag: str, default: int) -> int:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith(f"{flag}="):
            return int(arg.split("=", 1)[1])
    return default


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main(
        use_csv_sample="--csv" in sys.argv,
        csv_n=_parse_int_flag(sys.argv, "--csv-n", 15),
        csv_seed=_parse_int_flag(sys.argv, "--csv-seed", 42),
    )
