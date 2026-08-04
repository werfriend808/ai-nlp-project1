"""
agent/mapping/test_mapping.py — 3단계(keyword_search/embedding_search/reranker) 통합 평가

golden_set.py가 관리하는 실제 골든셋(엑셀 2종 join + override, load_golden_set() 참고)을
그대로 로드해서 keyword/embedding/rerank 각각의 Recall@1/3/5·MRR을 계산한다.

예전엔 12개 문장을 이 파일에 직접 하드코딩해서 썼는데, golden_set.py가 이미 관리하는
정답과 별도로 손으로 베껴 쓰다 보니 골든셋이 갱신돼도 여기는 안 따라가서 정답이 어긋나는
문제가 있었다("청년 실업률" 문장의 기대값이 카탈로그 override 결정(DT_1DA7102S)과 달리
DT_1DA7001S로 박혀 있었음 — keyword/embedding/rerank 셋 다 정답을 냈는데도 전부 X 처리됨).
golden_set을 직접 로드하면 이 드리프트가 원천적으로 사라진다.

리랭커 평가 시 document_texts로 table_catalog.json의 embedding_text(키워드+설명 포함)를
넘긴다. table_name(짧은 제목)만 넘기면 리랭커가 원래 성능을 못 내는데(모델 비교 실험에서
이미 확인됨), search_and_rerank()가 예전엔 이걸 안 넘겨서 리랭커 점수가 실제보다 낮게
나오고 있었다.

실행:
    python -m agent.mapping.test_mapping   (프로젝트 루트에서)
"""

from __future__ import annotations

from agent.mapping.embedding_search import build_table_embedding_cache, embedding_search
from agent.mapping.eval_metrics import EvalResult, evaluate
from agent.mapping.golden_set import load_golden_set
from agent.mapping.keyword_search import keyword_search
from agent.mapping.reranker import search_and_rerank, load_document_texts

try:
    from agent.interfaces import Claim
except ImportError:
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class Claim:  # type: ignore[no-redef]
        sentence: str
        claim_type: str
        period: Optional[str] = None
        unit: Optional[str] = None
        population: Optional[str] = None

TOP_K = 5  # eval_metrics.DEFAULT_K_LIST의 최댓값(Recall@5)을 커버하려면 rank_fn이 5개는 반환해야 함


def _print_result(r: EvalResult) -> None:
    recall_str = "  ".join(f"Recall@{k}={r.recall_at_k[k] * 100:.1f}%" for k in sorted(r.recall_at_k))
    print(
        f"[{r.model}] top-1={r.accuracy * 100:.1f}%  {recall_str}  "
        f"MRR={r.mrr:.3f}  avg_latency={r.mean_latency_ms:.1f}ms  (n={r.n})"
    )


def _print_failures(r: EvalResult, examples_by_id: dict[str, str], limit: int = 5) -> None:
    failures = [e for e in r.per_example if e["rank"] != 1]
    if not failures:
        return
    print(f"  top-1 오답 {len(failures)}건 중 {min(limit, len(failures))}건:")
    for f in failures[:limit]:
        sentence = examples_by_id.get(f["claim_id"], "")
        print(f"    [{f['claim_id']}] 기대={f['gold_table_id']} 실제={f['top1']} | {sentence[:40]}")


def run_tests(top_k: int = TOP_K) -> None:
    examples = load_golden_set()
    document_texts = load_document_texts()
    embedding_cache = build_table_embedding_cache()
    examples_by_id = {e.claim_id: e.sentence for e in examples}

    def kw_rank(sentence: str) -> list[str]:
        claim = Claim(sentence=sentence, claim_type="")
        return [c.table_id for c in keyword_search(claim, top_k=top_k)]

    def emb_rank(sentence: str) -> list[str]:
        claim = Claim(sentence=sentence, claim_type="")
        return [c.table_id for c in embedding_search(claim, top_k=top_k, cache=embedding_cache)]

    def rerank_rank(sentence: str) -> list[str]:
        claim = Claim(sentence=sentence, claim_type="")
        candidates = search_and_rerank(
            claim,
            keyword_fn=lambda c: keyword_search(c, top_k=top_k),
            embedding_fn=lambda c: embedding_search(c, top_k=top_k, cache=embedding_cache),
            top_k=top_k,
            document_texts=document_texts,
        )
        return [c.table_id for c in candidates]

    print(f"=== golden_set {len(examples)}건으로 keyword/embedding/rerank 평가 ===\n")

    results = [
        evaluate("keyword_search", kw_rank, examples),
        evaluate("embedding_search (multilingual-e5-large)", emb_rank, examples),
        evaluate("rerank (bge-reranker-v2-m3, rich doc text)", rerank_rank, examples),
    ]

    for r in results:
        _print_result(r)
        _print_failures(r, examples_by_id)
        print()

    print("=== 결과 요약 ===")
    for r in results:
        print(f"{r.model} top-1 정답률: {r.accuracy * 100:.1f}% ({int(r.accuracy * r.n)}/{r.n})")
    print(f"reranker(최종) 결과가 실질적인 3단계 정확도 - keyword/embedding 후보를 합쳐 재정렬한 값.")


if __name__ == "__main__":
    run_tests()
