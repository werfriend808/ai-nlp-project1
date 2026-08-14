"""
agent/pipeline/resume_after_rerank.py — 코랩에서 리랭킹한 결과를 받아 4~8단계를 마저 실행한다.

3단계 분리 흐름의 마지막 조각:
    export_for_rerank.py(로컬) → notebooks/reranker_colab.ipynb(코랩) → 이 스크립트(로컬)

data/rerank_pending.json(원본 claim·후보 정보)과 data/rerank_results.json(코랩에서 받은
리랭킹 점수)을 item_id 기준으로 합쳐서, 각 claim의 최종 top 후보를 정하고
batch_runner._finish_claim_with_top_candidate()로 4~8단계를 마저 실행한다 — run_article()의
리랭킹 이후 로직과 완전히 동일한 함수를 공유하므로 결과가 갈라지지 않는다.

사용법 (프로젝트 루트에서, rerank_results.json을 data/ 밑에 받아둔 뒤):
    python -m agent.pipeline.resume_after_rerank
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

from agent.interfaces import Claim, ClassificationResult, TableCandidate
from agent.kosis.api_client import KosisApiClient
from agent.kosis.calculator import KosisCalculator

from agent.pipeline.batch_runner import (
    TABLE_PARAMS_PATH,
    _finish_claim_with_top_candidate,
    _load_table_catalog_by_id,
    print_review_summary,
)

PENDING_PATH = Path(__file__).parent.parent.parent / "data" / "rerank_pending.json"
RESULTS_PATH = Path(__file__).parent.parent.parent / "data" / "rerank_results.json"


def _article_from_json(data: dict) -> dict:
    out = dict(data)
    if isinstance(out.get("published_date"), str):
        y, m, d = (int(v) for v in out["published_date"].split("-"))
        out["published_date"] = date(y, m, d)
    return out


def main() -> None:
    if not PENDING_PATH.exists():
        raise SystemExit(f"{PENDING_PATH}가 없습니다. 먼저 export_for_rerank.py를 실행하세요.")
    if not RESULTS_PATH.exists():
        raise SystemExit(
            f"{RESULTS_PATH}가 없습니다. 코랩(reranker_colab.ipynb)에서 리랭킹 결과를 받아 "
            f"이 경로에 저장하세요."
        )

    pending = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    reranked_by_id = {item["item_id"]: item["candidates"] for item in results["items"]}

    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)
    catalog_by_id = _load_table_catalog_by_id()
    client = KosisApiClient()
    calculator = KosisCalculator()

    all_results: list[dict] = []
    skipped = 0
    for item in pending["items"]:
        item_id = item["item_id"]
        reranked_candidates = reranked_by_id.get(item_id)
        if not reranked_candidates:
            print(f"[건너뜀] {item_id} — 리랭킹 결과 파일에 없음(코랩에서 빠졌거나 아직 안 돌림)")
            skipped += 1
            continue

        article = _article_from_json(item["article"])
        claim = Claim(**item["claim"])
        cls_result = ClassificationResult(**item["cls_result"])
        top = TableCandidate(
            table_id=reranked_candidates[0]["table_id"],
            table_name=reranked_candidates[0]["table_name"],
            score=reranked_candidates[0]["score"],
            required_slots=reranked_candidates[0].get("required_slots", []),
            source_meta=reranked_candidates[0].get("source_meta"),
        )

        print(f"\n{'-' * 60}")
        print(f"[재개] {item_id} — 주장: \"{claim.sentence}\"")
        result = _finish_claim_with_top_candidate(
            article, claim, cls_result, top, table_params, client, calculator, catalog_by_id
        )
        if result is not None:
            all_results.append(result)

    if skipped:
        print(f"\n[안내] {skipped}건은 리랭킹 결과가 없어 건너뜀")
    print_review_summary(all_results)
    print(f"\n[DB] {len(all_results)}건을 data/verifications.db에 저장했습니다.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
