"""
agent/mapping/reranker.py — 3단계: 리랭커로 후보 재정렬

팀 계약(interfaces.py) 기준:
입력: Claim 1건 + TableCandidate 리스트 (keyword_search/embedding_search에서 모은 후보들)
출력: TableCandidate 리스트 (재정렬, top-k)

모델: Qwen3-Reranker-4B (notebooks/reranker_model_comparison.ipynb 비교 실험 결과 채택 —
50개 라벨링 케이스 기준 top-1 64%로 4종(BGE-reranker-v2-m3 38%, ko-reranker 24%,
bge-reranker-v2-m3-ko 10%) 중 1위). 분류 헤드가 아니라 생성형(decoder) 리랭커라
"yes"/"no" 토큰의 확률을 관련도 점수로 쓴다.

3단계 전체 흐름:
  keyword_search 결과 + embedding_search 결과 → table_id 기준 합치기(중복 제거)
  → rerank()로 최종 top-k 재정렬

사전 준비물:
    pip install transformers torch accelerate
    GPU 권장(4B 파라미터 모델). VRAM이 부족하면 환경변수
    KOSIS_RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B 로 교체.
    transformers/torch가 없거나 모델 로딩·추론에 실패하면 rerank_scores()가 None을
    반환해서, rerank()가 기존 score(키워드 매칭 점수 or 임베딩 유사도)를 그대로 정렬
    기준으로 쓰는 항등(identity) 폴백으로 자동으로 넘어간다.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

try:
    from agent.interfaces import Claim, TableCandidate
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class Claim:  # type: ignore[no-redef]
        sentence: str
        claim_type: str
        period: Optional[str] = None
        unit: Optional[str] = None
        population: Optional[str] = None

    @dataclass
    class TableCandidate:  # type: ignore[no-redef]
        table_id: str
        table_name: str
        score: float
        required_slots: list = field(default_factory=list)
        source_meta: Optional[str] = None


class RerankerError(RuntimeError):
    """리랭커 모델 호출 실패."""


RERANKER_MODEL = os.environ.get("KOSIS_RERANKER_MODEL", "Qwen/Qwen3-Reranker-4B")
_RERANKER_MAX_LENGTH = 4096
_RERANKER_INSTRUCTION = (
    "Given a Korean news claim sentence, judge whether the KOSIS statistical table "
    "description is the correct match"
)
_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_reranker_singleton = None  # (tokenizer, model, prefix_tokens, suffix_tokens, true_id, false_id)


def _get_reranker():
    """Qwen3-Reranker를 lazy하게 로딩해서 프로세스 전체에서 재사용한다."""
    global _reranker_singleton
    if _reranker_singleton is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL, padding_side="left")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()

        prefix_tokens = tokenizer.encode(_RERANKER_PREFIX, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(_RERANKER_SUFFIX, add_special_tokens=False)
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")

        _reranker_singleton = (
            tokenizer, model, prefix_tokens, suffix_tokens, token_true_id, token_false_id,
        )
    return _reranker_singleton


# ---------------------------------------------------------------------------
# 실제 리랭커 모델 연동 지점. Qwen3-Reranker를 로컬에서 호출한다.
# transformers/torch가 없거나 모델 로딩·추론에 실패하면 None을 반환해서
# rerank()가 후보의 기존 score(키워드 매칭 점수 or 임베딩 유사도)를
# 그대로 정렬 기준으로 쓰는 항등(identity) 폴백으로 넘어가게 한다.
# ---------------------------------------------------------------------------
def rerank_scores(query: str, documents: list[str]) -> Optional[list[float]]:
    if not documents:
        return []

    try:
        import torch

        tokenizer, model, prefix_tokens, suffix_tokens, token_true_id, token_false_id = _get_reranker()
        device = model.device

        pairs = [
            f"<Instruct>: {_RERANKER_INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]
        scores: list[float] = []
        batch_size = 8
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                inputs = tokenizer(
                    batch, padding=False, truncation="longest_first",
                    return_attention_mask=False,
                    max_length=_RERANKER_MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
                )
                for j, ids in enumerate(inputs["input_ids"]):
                    inputs["input_ids"][j] = prefix_tokens + ids + suffix_tokens
                inputs = tokenizer.pad(
                    inputs, padding=True, return_tensors="pt", max_length=_RERANKER_MAX_LENGTH
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                logits = model(**inputs).logits[:, -1, :]
                stacked = torch.stack(
                    [logits[:, token_false_id], logits[:, token_true_id]], dim=1
                )
                probs = torch.nn.functional.log_softmax(stacked, dim=1)
                scores.extend(probs[:, 1].exp().tolist())
        return scores
    except Exception as exc:  # noqa: BLE001 - 로딩/추론 실패는 전부 항등 폴백 대상
        print(f"[reranker] Qwen3-Reranker 사용 불가({exc!r}) - 항등(identity) 정렬로 폴백합니다.")
        return None


def _merge_candidates(
    keyword_candidates: list[TableCandidate],
    embedding_candidates: list[TableCandidate],
) -> list[TableCandidate]:
    """keyword_search와 embedding_search 후보를 table_id 기준으로 합친다.

    실제 리랭커/임베딩 API가 붙기 전까지 embedding_search의 코사인 유사도는
    의미 신호가 아니라 노이즈에 가까웠다 (embed_texts의 해시 기반 폴백 참고).
    그래서 두 score를 크기로 직접 비교하지 않는다:
      - keyword_search가 찾은 표는 그 score를 그대로 신뢰 가능한 신호로 쓴다.
      - embedding_search가 추가로 찾은 표(keyword가 못 찾은 것)는 recall 보충용으로만
        살려두고 "unverified"로 표시해서, rerank_scores()가 실제로 점수를 매겨 재평가하게 한다.
    입력으로 받은 candidate 객체는 변형하지 않고 dataclasses.replace로 복사본만 만든다.
    """
    merged: dict[str, TableCandidate] = {}

    for cand in keyword_candidates:
        merged[cand.table_id] = cand

    for cand in embedding_candidates:
        existing = merged.get(cand.table_id)
        if existing is None:
            merged[cand.table_id] = replace(
                cand, source_meta=f"{cand.source_meta} (embedding-only, unverified)"
            )
        else:
            merged[cand.table_id] = replace(
                existing, source_meta=f"{existing.source_meta} | {cand.source_meta}"
            )
    return list(merged.values())


def rerank(
    claim: Claim,
    candidates: list[TableCandidate],
    *,
    top_k: int = 5,
    document_texts: Optional[dict[str, str]] = None,
) -> list[TableCandidate]:
    """후보 TableCandidate 리스트를 리랭커로 재정렬한다.

    document_texts: table_id -> 임베딩/설명 텍스트. 넘기지 않으면 table_name으로 대체.
    """
    if not candidates:
        return []

    documents = [
        (document_texts or {}).get(c.table_id, c.table_name) for c in candidates
    ]
    scores = rerank_scores(claim.sentence, documents)

    if scores is None:
        # 리랭커 모델을 못 쓰는 상황(의존성 미설치 등) — 항등 폴백.
        # embedding-only(unverified) 후보는 코사인 유사도가 노이즈에 가까울 수 있어서
        # score 크기만으로 정렬하면 keyword_search가 검증한 후보를 밀어낸다.
        # 검증된 후보를 항상 먼저 두고, 그 안에서만 score 내림차순으로 정렬한다.
        def _sort_key(c: TableCandidate) -> tuple[bool, float]:
            unverified = "(embedding-only, unverified)" in (c.source_meta or "")
            return (unverified, -c.score)

        return sorted(candidates, key=_sort_key)[:top_k]

    reranked: list[TableCandidate] = []
    for cand, score in zip(candidates, scores):
        reranked.append(
            TableCandidate(
                table_id=cand.table_id,
                table_name=cand.table_name,
                score=score,
                required_slots=cand.required_slots,
                source_meta=f"{cand.source_meta} | reranked",
            )
        )

    reranked.sort(key=lambda c: c.score, reverse=True)
    return reranked[:top_k]


def search_and_rerank(
    claim: Claim,
    *,
    keyword_fn,
    embedding_fn,
    top_k: int = 5,
) -> list[TableCandidate]:
    """3단계 전체 흐름: keyword_search + embedding_search 결과를 합쳐 rerank까지 수행.

    keyword_fn, embedding_fn: 각각 keyword_search(claim), embedding_search(claim) 함수를 주입.
    """
    kw_results = keyword_fn(claim)
    emb_results = embedding_fn(claim)
    merged = _merge_candidates(kw_results, emb_results)
    return rerank(claim, merged, top_k=top_k)


if __name__ == "__main__":
    # python -m agent.mapping.reranker
    from agent.mapping.keyword_search import keyword_search
    from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache

    cache = build_table_embedding_cache()

    test_claims = [
        Claim(sentence="지난달 청년 실업률이 6%에 육박했다", claim_type="규모"),
        Claim(sentence="지난달 소비자물가가 전년 동월 대비 2.2% 올랐다", claim_type="증감률"),
        Claim(sentence="전국 주택 매매가격이 지수화 기준으로 하락세를 보였다", claim_type="비교"),
        Claim(sentence="출생아 수가 14.6% 증가했다", claim_type="증감률"),
        Claim(sentence="지난해 수출이 6838억달러로 역대 최대를 기록했다", claim_type="규모"),
    ]
    for c in test_claims:
        result = search_and_rerank(
            c,
            keyword_fn=keyword_search,
            embedding_fn=lambda claim: embedding_search(claim, cache=cache),
        )
        print(f"\n[{c.sentence}]")
        for r in result[:3]:
            print(f"  - {r.table_name} ({r.table_id}) score={r.score:.3f} | {r.source_meta}")
