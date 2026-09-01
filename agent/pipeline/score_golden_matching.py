"""
agent/pipeline/score_golden_matching.py — 1~3단계(분류·추출·매핑)만 떼어서 채점

score_golden_set.py는 1~8단계 전체(최종 일치/불일치 판정)를 채점하는데, 8단계가 곱해지는
구조라 각 단계가 개별적으로 괜찮아도 전체 정확도가 크게 낮아진다. 반대로 팀원의
agent/mapping/golden_set.py + eval_metrics.py는 "사람이 이미 뽑아둔 정답 문장"을 그대로
3단계에 넣고 Recall@K만 재서, 2단계(claim_extractor)가 실제로 그 주장을 잘 찾아내는지는
전혀 반영하지 않는다.

이 스크립트는 그 중간 — **원본 기사부터 시작하되(2단계도 우리 파이프라인이 직접 함),
딱 3단계 매핑까지만** 채점한다: "우리 시스템이 스스로 찾은 claim의 kosis_table_id가
골든셋 정답과 같은가". 4~8단계(슬롯 채우기/API/계산/판정)는 관여하지 않는다.

data/verifications.db에 이미 kosis_table_id가 기록돼 있으므로(score_golden_set.py로 이미
파이프라인을 돌려놓은 상태) API를 다시 호출하지 않고 그 결과를 재사용한다.

사용법 (프로젝트 루트에서, data/verifications.db가 이미 있어야 함):
    python -m agent.pipeline.score_golden_matching
"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import pandas as pd

from agent.mapping.golden_set import TABLE_ID_OVERRIDES
from db.store import fetch_all

ROOT_DIR = Path(__file__).parent.parent.parent
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
CLAIMS_GOLDEN_PATH = NOTEBOOKS_DIR / "claims_golden_merged.csv"
MAPPING_GOLDEN_PATH = NOTEBOOKS_DIR / "mapping_golden_merged.csv"
REPORT_PATH = ROOT_DIR / "tests" / "golden_matching_scoring_report.md"

SIMILARITY_THRESHOLD = 0.5
# 3단계 매핑 자체가 못 맞히는 게 정상인 케이스 — 골든셋의 golden_set.py와 동일 기준.
_NO_ANSWER_STATUSES = {"매칭 실패", "미완료"}


def _normalize_url(url: object) -> str:
    return str(url).strip().rstrip("/")


def _load_golden() -> pd.DataFrame:
    claims = pd.read_csv(CLAIMS_GOLDEN_PATH)[["claim_id", "article_url", "claim_sentence"]]
    mapping = pd.read_csv(MAPPING_GOLDEN_PATH)[["claim_id", "kosis_table_id", "match_status"]]
    merged = claims.merge(mapping, on="claim_id", how="inner")
    merged = merged[~merged["match_status"].isin(_NO_ANSWER_STATUSES)].copy()
    merged["gold_table_id"] = merged["kosis_table_id"].map(lambda t: TABLE_ID_OVERRIDES.get(t, t))
    merged["url_norm"] = merged["article_url"].map(_normalize_url)
    return merged


def _group_by_url(records: list[dict]) -> dict[str, list[dict]]:
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


def score() -> None:
    golden = _load_golden()
    by_url = _group_by_url(fetch_all())

    rows_out = []
    for _, g in golden.iterrows():
        candidates = by_url.get(g["url_norm"], [])
        match, sim = _best_match(g["claim_sentence"], candidates)
        matched = match is not None and sim >= SIMILARITY_THRESHOLD
        system_table_id = match.get("kosis_table_id") if matched else None
        rows_out.append(
            {
                "claim_id": g["claim_id"],
                "claim_sentence": g["claim_sentence"][:40],
                "gold_table_id": g["gold_table_id"],
                "system_table_id": system_table_id,
                "extracted": matched,  # 2단계가 이 주장을 찾았는지
                "table_correct": matched and system_table_id == g["gold_table_id"],
                "similarity": round(sim, 3),
            }
        )

    df = pd.DataFrame(rows_out)
    n_total = len(df)
    n_extracted = df["extracted"].sum()
    n_correct = df["table_correct"].sum()

    extracted_df = df[df["extracted"]]
    correct_given_extracted = extracted_df["table_correct"].sum()
    n_extracted_denom = len(extracted_df)

    _write_report(df, n_total, n_extracted, n_correct, correct_given_extracted, n_extracted_denom)

    print(f"[score_matching] 골든셋 평가 대상(매칭 실패 제외): {n_total}건")
    print(f"[score_matching] 2단계가 주장을 찾음: {n_extracted}/{n_total} ({n_extracted / n_total:.1%})")
    print(
        f"[score_matching] 표 매칭 정확도(전체 대비): {n_correct}/{n_total} ({n_correct / n_total:.1%})"
    )
    print(
        f"[score_matching] 표 매칭 정확도(찾은 것 중, 3단계만): "
        f"{correct_given_extracted}/{n_extracted_denom} "
        f"({correct_given_extracted / n_extracted_denom:.1%})" if n_extracted_denom else "N/A"
    )
    print(f"[score_matching] 리포트 -> {REPORT_PATH}")


def _write_report(df, n_total, n_extracted, n_correct, correct_given_extracted, n_extracted_denom) -> None:
    lines = ["# 1~3단계(분류·추출·매핑) 채점 리포트\n\n"]
    lines.append(
        "팀원 평가(golden_set.py)와 다르게, 정답 문장을 미리 안 주고 실제 기사부터 우리 "
        "2단계 claim_extractor가 직접 문장을 찾게 한 뒤 3단계 매핑 결과를 채점합니다. "
        "4~8단계(슬롯 채우기/API/계산/판정)는 관여하지 않습니다.\n\n"
    )
    lines.append(f"- 평가 대상(골든셋 match_status=매칭 실패/미완료 제외): {n_total}건\n")
    lines.append(f"- **2단계가 이 주장을 찾아냄**: {n_extracted}/{n_total} ({n_extracted/n_total:.1%})\n")
    lines.append(f"- **표 매칭 정확도 (전체 대비)**: {n_correct}/{n_total} ({n_correct/n_total:.1%})\n")
    if n_extracted_denom:
        lines.append(
            f"- **표 매칭 정확도 (2단계가 찾은 것만, 순수 3단계 성능)**: "
            f"{correct_given_extracted}/{n_extracted_denom} ({correct_given_extracted/n_extracted_denom:.1%})\n\n"
        )

    lines.append("## claim별 상세\n\n")
    lines.append(df.to_markdown(index=False))
    lines.append("\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    score()
