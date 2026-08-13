"""
agent/pipeline/export_for_rerank.py — 1~2단계+키워드매칭까지를 로컬에서 돌리고,
임베딩·리랭킹에 필요한 claim별 후보/카탈로그 정보를 파일로 내보낸다.

배경: 임베딩(intfloat/multilingual-e5-large)과 리랭커(BAAI/bge-reranker-v2-m3, 568M)
둘 다 로컬(RAM 7.4GB)에서 문제가 있다 — 리랭커는 로드 자체가 세그폴트, 임베딩은 큰 배치
한 번은 되는데(build_table_embedding_cache) claim마다 반복 호출하면 세그폴트(2026-08-13
실측, 원인 미상 — 이 컴퓨터의 고질적 리소스 불안정으로 추정). 코랩(T4, RAM 12GB+)에서는
둘 다 문제없이 동작 확인됨. 그래서 파이프라인을 세 조각으로 나눈다:
  1) (이 스크립트, 로컬) 분류→claim 추출→출처필터→키워드매칭까지만 하고,
     claim 목록 + 카탈로그(임베딩용 텍스트)를 JSON으로 저장 — 임베딩/리랭킹은 안 함
  2) (코랩, notebooks/reranker_colab.ipynb) 그 JSON을 읽어 임베딩 매칭 + 리랭킹까지
     전부 실행하고 결과 저장
  3) (resume_after_rerank.py, 로컬) 최종 결과를 받아 4~8단계 마저 실행

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
from agent.orchestrator.calc_type_router import _mentions_foreign_country
from agent.interfaces import Verdict
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

        # keyword_search가 0건이어도 그대로 내보낸다 — 코랩에서 임베딩으로 후보를 찾을 수도
        # 있으므로, 여기서 "매칭 없음"으로 확정하지 않는다(그건 코랩이 임베딩까지 합친 뒤
        # 최종 판단할 일).
        print(f"[3단계 키워드매칭] {len(kw_results)}개 후보 (임베딩·리랭킹은 코랩에서, 코랩으로 넘김)")
        pending_items.append(
            {
                "item_id": f"{article['label']}::{i}",
                "article": article_json,
                "claim": dataclasses.asdict(claim),
                "cls_result": dataclasses.asdict(cls_result),
                "keyword_candidates": [
                    {
                        "table_id": c.table_id,
                        "table_name": c.table_name,
                        "score": c.score,
                        "required_slots": c.required_slots,
                        "source_meta": c.source_meta,
                    }
                    for c in kw_results
                ],
            }
        )


def main(use_csv_sample: bool = False, csv_n: int = 15, csv_seed: int = 42) -> None:
    catalog_by_id = _load_table_catalog_by_id()

    articles = load_articles_from_csv(n=csv_n, seed=csv_seed) if use_csv_sample else ARTICLES

    pending_items: list[dict] = []
    for article in articles:
        export_article(article, catalog_by_id, pending_items)

    # 임베딩 매칭을 코랩에서 하려면 카탈로그(표별 embedding_text)가 통째로 필요하다 —
    # keyword_candidates에 걸리지 않은 claim도 코랩에서는 임베딩으로 후보를 찾아야 하므로,
    # "이 claim에서 keyword로 이미 찾은 표"뿐 아니라 카탈로그 전체를 같이 내보낸다.
    catalog_export = {
        tid: {"table_name": t.get("title", tid), "embedding_text": t.get("embedding_text", t.get("title", tid))}
        for tid, t in catalog_by_id.items()
    }

    PENDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    PENDING_PATH.write_text(
        json.dumps({"catalog": catalog_export, "items": pending_items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[export] 리랭킹 대기 claim {len(pending_items)}건 + 카탈로그 {len(catalog_export)}개 표를 "
          f"{PENDING_PATH}에 저장했습니다.")
    print("다음: 이 파일을 코랩(notebooks/reranker_colab.ipynb)에 업로드해서 임베딩+리랭킹을 실행하세요.")


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
