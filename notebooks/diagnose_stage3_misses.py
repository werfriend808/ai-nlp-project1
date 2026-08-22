"""
notebooks/diagnose_stage3_misses.py — 3단계 표 매칭 미스 70건(정확히는 정답표 있는
claim 전체)을 "세부분류 불일치(정답표 VDB 텍스트에 claim 핵심 용어가 아예 없음)" vs
"그 외(용어는 있는데 못 찾음)"로 자동 분류한다.

19개 표를 수동으로 하나씩 파는 대신, "정답표의 VDB text에 statistic_expression의
핵심 단어가 문자열로 들어있는가"를 객관적으로 검사한다 — 이게 없으면(자살률 케이스처럼)
검색어를 아무리 잘 만들어도 구조적으로 못 찾는 게 당연하고, 있는데도 못 찾으면 랭킹/
임베딩 품질 등 다른 원인일 가능성이 높다.

사용법: python -m notebooks.diagnose_stage3_misses
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

GOLDEN_PATH = Path(__file__).parent / "골든셋_통합.xlsx"
REPORT_PATH = Path(__file__).parent / "stage3_miss_diagnosis.md"


def _normalize(text: str) -> str:
    """공백/가운뎃점 제거 비교용 정규화."""
    return re.sub(r"[\s·・]+", "", str(text))


def _core_terms(statistic_expression: str) -> list[str]:
    """statistic_expression에서 핵심 명사 후보를 뽑는다. 조사/불필요한 수식어를
    엄밀히 걸러내긴 어려우니, 2자 이상 연속 한글/영문 토큰을 다 후보로 삼고
    정규화된 표 텍스트 안에 부분 문자열로 있는지를 본다(형태소 분석 없이도
    "자살률" 같은 핵심어는 이 방식으로 충분히 잡힘)."""
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", str(statistic_expression))
    return tokens


def main() -> None:
    df7 = pd.read_excel(GOLDEN_PATH, sheet_name="7단계_판정목록")
    df2 = pd.read_excel(GOLDEN_PATH, sheet_name="2단계_claim목록")
    df7 = df7.merge(
        df2[["claim_id", "statistic_expression", "population", "region", "source_org"]],
        on="claim_id", how="left",
    )
    ids_stripped = df7["matched_table_id(3단계)"].astype(str).str.strip()
    evalable = df7[ids_stripped != "없음"].reset_index(drop=True)

    all_tbl_ids: set[str] = set()
    for v in evalable["matched_table_id(3단계)"]:
        all_tbl_ids.update(x.strip() for x in str(v).split(","))

    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor()
    cur.execute("select tbl_id, text from kosis_vdb_tables where tbl_id = ANY(%s)", (list(all_tbl_ids),))
    table_text = {r[0]: r[1] for r in cur.fetchall()}
    missing_in_vdb = all_tbl_ids - set(table_text.keys())

    vocab_gap = []      # 정답표 텍스트에 핵심 용어가 전혀 없음 -> 구조적으로 못 찾을 수밖에 없음
    vocab_present = []  # 핵심 용어가 있는데도(우리가 알기론) 검색어 구성에 따라 다름 -> 재확인 필요
    no_table_data = []  # 정답 tbl_id 자체가 VDB에 없음(별도 카탈로그이거나 오타 등)

    for _, row in evalable.iterrows():
        gold_ids = [x.strip() for x in str(row["matched_table_id(3단계)"]).split(",")]
        stat_expr = row.get("statistic_expression")
        terms = _core_terms(stat_expr) if isinstance(stat_expr, str) else []

        texts = []
        any_missing = False
        for gid in gold_ids:
            if gid in table_text:
                texts.append(_normalize(table_text[gid]))
            else:
                any_missing = True

        if not texts:
            no_table_data.append((row["claim_id"], gold_ids, stat_expr))
            continue

        combined_text = " ".join(texts)
        matched_terms = [t for t in terms if _normalize(t) in combined_text]

        record = {
            "claim_id": row["claim_id"],
            "sentence": row["sentence(원문 그대로)"],
            "statistic_expression": stat_expr,
            "gold_ids": gold_ids,
            "gold_text": [table_text.get(g, "(VDB에 없음)") for g in gold_ids],
            "terms": terms,
            "matched_terms": matched_terms,
        }
        if not matched_terms and terms:
            vocab_gap.append(record)
        else:
            vocab_present.append(record)

    n = len(evalable)
    report = ["# 3단계 매핑 실패 원인 분류 (정답표 있는 claim 전체 기준)\n\n"]
    report.append(f"- 평가 대상: {n}건\n")
    report.append(f"- **세부분류 불일치**(정답표 VDB 텍스트에 statistic_expression 핵심 용어가 전혀 없음): "
                   f"{len(vocab_gap)}건 ({len(vocab_gap)/n:.1%})\n")
    report.append(f"- **용어는 존재**(핵심 용어가 정답표 텍스트에 있음 — 그래도 못 찾았다면 다른 원인): "
                   f"{len(vocab_present)}건 ({len(vocab_present)/n:.1%})\n")
    if no_table_data:
        report.append(f"- 정답 tbl_id가 VDB에 아예 없음(카탈로그 누락/오타 등): {len(no_table_data)}건\n")
    report.append(f"- VDB에 없는 gold tbl_id: {sorted(missing_in_vdb)}\n\n")

    report.append("## 세부분류 불일치 목록\n\n")
    for r in vocab_gap:
        report.append(f"- **[{r['claim_id']}]** stat_expr={r['statistic_expression']!r}\n")
        report.append(f"  - 문장: {r['sentence'][:80]}\n")
        report.append(f"  - 정답표: {r['gold_ids']} -> {r['gold_text']}\n\n")

    report.append("## 용어는 있는데 (그래도) 확인 필요한 목록\n\n")
    for r in vocab_present:
        report.append(f"- **[{r['claim_id']}]** stat_expr={r['statistic_expression']!r}, 매칭용어={r['matched_terms']}\n")
        report.append(f"  - 문장: {r['sentence'][:80]}\n")
        report.append(f"  - 정답표: {r['gold_ids']} -> {r['gold_text']}\n\n")

    REPORT_PATH.write_text("".join(report), encoding="utf-8")

    print(f"평가 대상: {n}건")
    print(f"세부분류 불일치: {len(vocab_gap)}건 ({len(vocab_gap)/n:.1%})")
    print(f"용어는 존재: {len(vocab_present)}건 ({len(vocab_present)/n:.1%})")
    if no_table_data:
        print(f"정답표 VDB에 없음: {len(no_table_data)}건")
    print(f"리포트 -> {REPORT_PATH}")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
