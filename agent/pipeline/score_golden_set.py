"""
agent/pipeline/score_golden_set.py — 골든셋 기준 파이프라인 자동 채점

notebooks/merge_golden_set.py로 만든 data/data_set_golden.csv(골든셋 기사만 추린 것)를
batch_runner.py의 기존 함수(run_article 등)로 그대로 1~8단계+DB 저장까지 돌린 다음, DB
결과를 notebooks/verdict_golden_merged.csv(정답 — 판정 골든셋 원본 또는 매핑 골든셋에서
유도된 것)와 비교해서 정확도를 계산한다.

⚠️ 왜 claim_id로 직접 조인을 못 하는가: 골든셋의 claim_id는 사람이 수동으로 뽑은 claim에
붙인 값이고, 우리 파이프라인(2단계 claim_extractor)은 같은 기사에서 매번 독립적으로 자기만의
claim을 뽑는다 — 정확히 같은 문장이 안 나올 수 있다. 그래서 두 단계로 연결한다:
  1) article_url로 같은 기사에서 나온 우리 파이프라인 claim들을 모으고
  2) 그 안에서 golden claim_sentence와 문장 유사도(difflib.SequenceMatcher)가 가장 높은
     걸 "우리 시스템의 답"으로 삼는다. 유사도가 SIMILARITY_THRESHOLD 미만이면 "우리
     파이프라인이 이 주장을 아예 못 찾음"으로 보고 시스템 판정을 "확인불가"로 취급한다
     (사람은 검증하려 했는데 우리가 놓쳐서 판정을 못 한 것도 실질적으로 확인불가와 같은
     상황이라서).

사용법 (프로젝트 루트에서, .env에 HCX_API_KEY/KOSIS_API_KEY 필요):
    python -m agent.pipeline.score_golden_set             # 골든셋 기사 전체 실행 + 채점
    python -m agent.pipeline.score_golden_set --n 5        # 앞 5건만(빠른 검증용) 실행 + 채점
    python -m agent.pipeline.score_golden_set --score-only # 파이프라인 재실행 없이 채점만
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import pandas as pd

from agent.kosis.api_client import KosisApiClient
from agent.kosis.calculator import KosisCalculator
from agent.mapping.embedding_search import build_table_embedding_cache
from agent.pipeline.batch_runner import (
    DEFAULT_CLARIFY_REPLY,
    TABLE_PARAMS_PATH,
    _clean_scraped_article_text,
    _load_table_catalog_by_id,
    run_article,
)
from db.store import DB_PATH, fetch_all

ROOT_DIR = Path(__file__).parent.parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
DATA_GOLDEN_PATH = ROOT_DIR / "data" / "data_set_golden.csv"
CLAIMS_GOLDEN_PATH = NOTEBOOKS_DIR / "claims_golden_merged.csv"
VERDICT_GOLDEN_PATH = NOTEBOOKS_DIR / "verdict_golden_merged.csv"
REPORT_PATH = ROOT_DIR / "tests" / "golden_set_scoring_report.md"

SIMILARITY_THRESHOLD = 0.5
LABELS = ["일치", "불일치", "확인불가"]


def _normalize_url(url: object) -> str:
    return str(url).strip().rstrip("/")


def _to_article_dict(row: dict) -> dict:
    """data_set_golden.csv 한 행을 batch_runner.run_article()이 기대하는 article dict로
    변환 (batch_runner.load_articles_from_csv()와 동일한 규칙)."""
    try:
        y, m, d = (int(v) for v in row["작성일"].split("-"))
        published = date(y, m, d)
    except (KeyError, ValueError):
        published = date(2025, 1, 1)
    title = row.get("기사제목", "")
    return {
        "label": f"[golden] {title[:40]}",
        "article_title": title,
        "article_url": row.get("URL"),
        "published_date": published,
        "article_text": _clean_scraped_article_text(title, row["기사 본문 전체"]),
        "clarify_reply": DEFAULT_CLARIFY_REPLY,
    }


def run_pipeline_on_golden_articles(n: Optional[int] = None) -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()  # 이전 실행 결과와 안 섞이게 깨끗한 상태에서 시작

    client = KosisApiClient()
    calculator = KosisCalculator()
    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)
    catalog_by_id = _load_table_catalog_by_id()
    embedding_cache = build_table_embedding_cache()

    with open(DATA_GOLDEN_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if n is not None:
        rows = rows[:n]

    print(f"[score_golden_set] 골든셋 기사 {len(rows)}건 파이프라인 실행 시작")
    for i, row in enumerate(rows, 1):
        article = _to_article_dict(row)
        print(f"\n[{i}/{len(rows)}] {article['label']}")
        run_article(article, client, calculator, table_params, embedding_cache, catalog_by_id)


def _load_golden_verdicts() -> pd.DataFrame:
    claims = pd.read_csv(CLAIMS_GOLDEN_PATH)[["claim_id", "article_url", "claim_sentence"]]
    verdict = pd.read_csv(VERDICT_GOLDEN_PATH)[["claim_id", "verification_result"]]
    merged = claims.merge(verdict, on="claim_id", how="inner")
    merged["url_norm"] = merged["article_url"].map(_normalize_url)
    return merged


def _group_pipeline_records_by_url(records: list[dict]) -> dict[str, list[dict]]:
    by_url: dict[str, list[dict]] = {}
    for r in records:
        if not r.get("article_url"):
            continue
        by_url.setdefault(_normalize_url(r["article_url"]), []).append(r)
    return by_url


def _best_match(golden_sentence: str, candidates: list[dict]) -> tuple[Optional[dict], float]:
    best, best_score = None, 0.0
    for rec in candidates:
        score = SequenceMatcher(None, golden_sentence, rec.get("claim_sentence") or "").ratio()
        if score > best_score:
            best, best_score = rec, score
    return best, best_score


def _system_label(rec: Optional[dict]) -> str:
    if rec is None:
        return "확인불가"
    result = rec.get("verification_result")
    return result if result in ("일치", "불일치") else "확인불가"  # None/판단불가 -> 확인불가로 통일


def score() -> None:
    golden = _load_golden_verdicts()
    by_url = _group_pipeline_records_by_url(fetch_all())

    rows_out = []
    for _, g in golden.iterrows():
        candidates = by_url.get(g["url_norm"], [])
        match, sim = _best_match(g["claim_sentence"], candidates)
        matched = match is not None and sim >= SIMILARITY_THRESHOLD
        rows_out.append(
            {
                "claim_id": g["claim_id"],
                "claim_sentence": g["claim_sentence"][:40],
                "gold": g["verification_result"],
                "system": _system_label(match if matched else None),
                "matched": matched,
                "similarity": round(sim, 3),
            }
        )

    df = pd.DataFrame(rows_out)
    n_total = len(df)
    correct_mask = df["gold"] == df["system"]
    full_acc = correct_mask.sum() / n_total if n_total else 0.0

    judged = df[df["gold"] != "확인불가"]
    n_judged = len(judged)
    judged_correct = (judged["gold"] == judged["system"]).sum()
    judged_acc = judged_correct / n_judged if n_judged else 0.0

    n_unmatched = (~df["matched"]).sum()
    confusion = Counter(zip(df["gold"], df["system"]))

    _write_report(df, n_total, correct_mask.sum(), full_acc, n_judged, judged_correct, judged_acc, n_unmatched, confusion)
    print(f"\n[score] 전체 정확도(3분류): {correct_mask.sum()}/{n_total} ({full_acc:.1%})")
    print(f"[score] 확인불가 제외 정확도: {judged_correct}/{n_judged} ({judged_acc:.1%})")
    print(f"[score] 파이프라인이 못 찾은 claim: {n_unmatched}/{n_total}")
    print(f"[score] 리포트 -> {REPORT_PATH}")


def _write_report(
    df: pd.DataFrame,
    n_total: int,
    n_correct: int,
    full_acc: float,
    n_judged: int,
    judged_correct: int,
    judged_acc: float,
    n_unmatched: int,
    confusion: Counter,
) -> None:
    lines = ["# 골든셋 자동 채점 리포트\n\n"]
    lines.append(f"- 대상 claim: {n_total}건 (골든셋 판정 기준)\n")
    lines.append(f"- 전체 정확도(일치/불일치/확인불가 3분류): {n_correct}/{n_total} ({full_acc:.1%})\n")
    lines.append(f"- 확인불가 제외 정확도(일치/불일치만): {judged_correct}/{n_judged} ({judged_acc:.1%})\n")
    lines.append(
        f"- 우리 파이프라인이 골든 claim을 아예 못 찾은 건(문장 유사도 {SIMILARITY_THRESHOLD} 미만): "
        f"{n_unmatched}/{n_total}\n\n"
    )

    lines.append("## 혼동행렬 (행=사람 판정, 열=시스템 판정)\n\n")
    lines.append("| 사람\\시스템 | " + " | ".join(LABELS) + " |\n")
    lines.append("|---|" + "---|" * len(LABELS) + "\n")
    for gold_label in LABELS:
        row_counts = [confusion.get((gold_label, sys_label), 0) for sys_label in LABELS]
        lines.append(f"| {gold_label} | " + " | ".join(str(c) for c in row_counts) + " |\n")
    lines.append("\n")

    lines.append("## claim별 상세\n\n")
    lines.append(df.to_markdown(index=False))
    lines.append("\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    n_arg = None
    if "--n" in sys.argv:
        n_arg = int(sys.argv[sys.argv.index("--n") + 1])
    if "--score-only" not in sys.argv:
        run_pipeline_on_golden_articles(n=n_arg)
    score()
