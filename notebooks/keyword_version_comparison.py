"""
notebooks/keyword_version_comparison.py — keyword_search 품질 개선을 위한
keyword 구성 버전(V1~V4) 비교 실험.

GPU 불필요 — keyword_search는 규칙 기반(모델 없음)이라 로컬 CPU에서 바로 실행 가능.
(embedding/reranker 평가는 별도로 Colab A100에서 진행 — notebooks/embedding_text_comparison.ipynb 참고)

실행: python notebooks/keyword_version_comparison.py  (프로젝트 루트에서)

버전 정의(agent/mapping/keyword_versions.json):
  V1 baseline    - table_catalog.json 현재 keywords
  V2 synonym     - V1 + 동의어/사용자 검색표현
  V3 article     - V2 + 실제 기사체 표현
  V4 extended    - V3 + 확장 표현(약어/영문명/전문용어)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from agent.interfaces import Claim
from agent.mapping.eval_metrics import evaluate, summarize
from agent.mapping.golden_set import load_golden_set
from agent.mapping.keyword_search import keyword_search

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "agent" / "mapping" / "table_catalog.json"
VERSIONS_PATH = ROOT / "agent" / "mapping" / "keyword_versions.json"
CLAIMS_XLSX = ROOT / "notebooks" / "추출 골든셋 단위 분리.xlsx"
MAPPING_XLSX = ROOT / "notebooks" / "매핑 골든셋 ord 추가.xlsx"

VERSION_KEYS = {
    "V1(baseline)": "v1_baseline",
    "V2(+동의어)": "v2_synonym",
    "V3(+기사체)": "v3_article",
    "V4(+확장표현)": "v4_extended",
}


def build_catalog_for_version(base_catalog: list[dict], versions: dict, version_key: str) -> list[dict]:
    """base_catalog의 keywords만 해당 버전 값으로 치환한 복사본을 만든다."""
    cat = copy.deepcopy(base_catalog)
    for table in cat:
        tid = table["tblId"]
        table["keywords"] = versions["tables"][tid][version_key]
    return cat


def make_rank_fn(catalog: list[dict], top_k: int = 5):
    def rank_fn(sentence: str) -> list[str]:
        claim = Claim(sentence=sentence, claim_type="규모")
        candidates = keyword_search(claim, top_k=top_k, catalog=catalog)
        return [c.table_id for c in candidates]

    return rank_fn


def main() -> None:
    base_catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["tables"]
    versions = json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))

    examples = load_golden_set(claims_path=CLAIMS_XLSX, mapping_path=MAPPING_XLSX, catalog_path=CATALOG_PATH)
    print(f"평가 가능 claim: {len(examples)}건\n")

    results = []
    per_version_examples: dict[str, dict] = {}

    for version_label, version_key in VERSION_KEYS.items():
        cat = build_catalog_for_version(base_catalog, versions, version_key)
        rank_fn = make_rank_fn(cat)
        result = evaluate(version_label, rank_fn, examples)
        results.append(result)
        per_version_examples[version_label] = {p["claim_id"]: p for p in result.per_example}
        print(
            f"[{version_label}] top-1={result.accuracy:.1%}  Recall@3={result.recall_at_k[3]:.1%}  "
            f"Recall@5={result.recall_at_k[5]:.1%}  MRR={result.mrr:.3f}  "
            f"latency={result.mean_latency_ms:.2f}ms"
        )

    print()
    summary_df = summarize(results)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(ROOT / "notebooks" / "keyword_version_comparison_summary.csv", index=False, encoding="utf-8-sig")

    # V1 -> V4 개선/악화 사례 비교 (claim 단위)
    v1 = per_version_examples["V1(baseline)"]
    v4 = per_version_examples["V4(+확장표현)"]
    rows = []
    for ex in examples:
        p1, p4 = v1[ex.claim_id], v4[ex.claim_id]
        rows.append(
            {
                "claim_id": ex.claim_id,
                "confidence": ex.confidence,
                "gold_table_id": ex.gold_table_id,
                "sentence": ex.sentence[:60],
                "V1_rank": p1["rank"],
                "V4_rank": p4["rank"],
                "V1_top1": p1["top1"],
                "V4_top1": p4["top1"],
                "V4_vs_V1": (
                    "개선" if (p4["rank"] or 99) < (p1["rank"] or 99)
                    else "악화" if (p4["rank"] or 99) > (p1["rank"] or 99)
                    else "동일"
                ),
            }
        )
    import pandas as pd

    diff_df = pd.DataFrame(rows)
    print("\nV4가 V1보다 개선/악화/동일 건수:")
    print(diff_df["V4_vs_V1"].value_counts())
    diff_df.to_csv(
        ROOT / "notebooks" / "keyword_version_comparison_v1_vs_v4_diff.csv", index=False, encoding="utf-8-sig"
    )
    print("\n저장 완료: keyword_version_comparison_summary.csv, keyword_version_comparison_v1_vs_v4_diff.csv")


if __name__ == "__main__":
    main()
