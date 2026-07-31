"""
agent/mapping/eval_metrics.py — 임베딩/리랭커/하이브리드 비교 실험 공용 지표 계산.

notebooks/embedding_model_comparison.ipynb, reranker_model_comparison.ipynb,
hybrid_ablation.ipynb에서 공통으로 쓴다. golden_set.py의 GoldExample 리스트와
"claim 문장 -> 정렬된 table_id 리스트" 함수(rank_fn)만 있으면 Recall@K/MRR/latency를
한 번에 계산해서 EvalResult로 돌려준다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from agent.mapping.golden_set import GoldExample

DEFAULT_K_LIST = (1, 3, 5)


@dataclass
class EvalResult:
    model: str
    n: int
    recall_at_k: dict[int, float]
    mrr: float
    mean_latency_ms: float
    param_count: Optional[str] = None
    per_example: list[dict] = field(default_factory=list)  # 오류 분석용 상세 기록

    @property
    def accuracy(self) -> float:
        """top-1 정확도 = Recall@1."""
        return self.recall_at_k.get(1, 0.0)


def rank_metrics(
    ranked_ids: list[str], gold_id: str, k_list: tuple[int, ...] = DEFAULT_K_LIST
) -> tuple[dict[int, bool], float]:
    """정렬된 table_id 리스트 하나에 대한 Recall@k(dict)와 reciprocal rank를 계산."""
    recall = {k: gold_id in ranked_ids[:k] for k in k_list}
    rr = 1.0 / (ranked_ids.index(gold_id) + 1) if gold_id in ranked_ids else 0.0
    return recall, rr


def evaluate(
    model_name: str,
    rank_fn: Callable[[str], list[str]],
    examples: list[GoldExample],
    *,
    k_list: tuple[int, ...] = DEFAULT_K_LIST,
    param_count: Optional[str] = None,
) -> EvalResult:
    """골든셋 전체에 rank_fn을 돌려서 Recall@K/MRR/평균 latency를 집계한다.

    rank_fn(sentence) -> 정렬된(best-first) table_id 리스트. top-k는 rank_fn 쪽에서
    max(k_list) 이상 반환한다고 가정한다(적게 반환하면 그만큼 recall이 낮게 잡힘).
    """
    correct_at_k = {k: 0 for k in k_list}
    total_rr = 0.0
    latencies: list[float] = []
    per_example: list[dict] = []

    for ex in examples:
        t0 = time.perf_counter()
        ranked = rank_fn(ex.sentence)
        latencies.append((time.perf_counter() - t0) * 1000)

        recall, rr = rank_metrics(ranked, ex.gold_table_id, k_list)
        for k in k_list:
            correct_at_k[k] += int(recall[k])
        total_rr += rr

        per_example.append(
            {
                "claim_id": ex.claim_id,
                "gold_table_id": ex.gold_table_id,
                "confidence": ex.confidence,
                "rank": ranked.index(ex.gold_table_id) + 1 if ex.gold_table_id in ranked else None,
                "top1": ranked[0] if ranked else None,
            }
        )

    n = len(examples)
    return EvalResult(
        model=model_name,
        n=n,
        recall_at_k={k: correct_at_k[k] / n for k in k_list},
        mrr=total_rr / n,
        mean_latency_ms=sum(latencies) / n,
        param_count=param_count,
        per_example=per_example,
    )


def rrf_merge(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion. 여러 순위 리스트(각각 table_id, best-first)를
    score(id) = sum(1 / (k + rank))로 합쳐 하나의 순위로 반환한다.

    키워드 점수와 임베딩 코사인 유사도는 스케일이 달라(게다가 임베딩이 더미 폴백일 땐
    노이즈에 가까움 — reranker.py의 _merge_candidates 참고) 절대 점수로 직접 비교하면
    안 된다는 게 기존 코드의 판단이었다. RRF는 순위만 쓰기 때문에 그 문제를 원천적으로
    피해간다 — "키워드 후보 우선" 이진 규칙의 대안으로 hybrid_ablation.ipynb에서 비교.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, table_id in enumerate(ranking, start=1):
            scores[table_id] = scores.get(table_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda tid: scores[tid], reverse=True)


def summarize(results: list[EvalResult], k_list: tuple[int, ...] = DEFAULT_K_LIST):
    """결과 리스트를 발표 슬라이드용 pandas DataFrame으로 정리."""
    import pandas as pd

    rows = []
    for r in results:
        row = {"모델": r.model, "표본수": r.n, "top-1 정확도": f"{r.accuracy * 100:.1f}%"}
        for k in k_list:
            if k == 1:
                continue
            row[f"Recall@{k}"] = f"{r.recall_at_k[k] * 100:.1f}%"
        row["MRR"] = f"{r.mrr:.3f}"
        row["평균 latency(ms)"] = f"{r.mean_latency_ms:.1f}"
        if r.param_count:
            row["파라미터"] = r.param_count
        rows.append(row)
    return pd.DataFrame(rows)
