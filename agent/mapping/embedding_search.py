"""
agent/mapping/embedding_search.py — 3단계: 임베딩 기반 top-k 검색

팀 계약(interfaces.py) 기준:
입력: Claim 1건
출력: TableCandidate의 리스트 (top-k, 코사인 유사도 기준)

모델: Qwen3-Embedding-4B (notebooks/embedding_model_comparison.ipynb 비교 실험 결과 채택 —
50개 라벨링 케이스 기준 top-1 70%로 4종(BGE-m3 44%, KoE5 20%, KoSimCSE 8%) 중 1위)

⚠️ 배치 임베딩 원칙(Day2 09:00-10:00 작업):
table_catalog.json의 embedding_text는 최초 1회만 임베딩해서 캐시 파일(TABLE_EMBEDDING_CACHE)에
저장한다. 검색할 때마다 표 20여 개를 매번 재임베딩하지 않는다 — 여기서 실제로 모델을
다시 부르는 건 "새 Claim 문장" 하나뿐이다. 캐시는 모델명이 바뀌면 자동으로 재생성된다.

사전 준비물:
    pip install sentence-transformers torch
    GPU 권장(4B 파라미터 모델). VRAM이 부족하면 환경변수
    KOSIS_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B 로 교체.
    sentence-transformers/torch가 없거나 모델 로딩에 실패하면 해시 기반 더미 벡터로
    자동 폴백한다 (의미 유사도가 반영되지 않으니 개발/테스트 전용).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
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

CATALOG_PATH = Path(__file__).parent / "table_catalog.json"
EMBEDDING_CACHE_PATH = Path(__file__).parent / "table_embeddings_cache.json"
EMBEDDING_MODEL = os.environ.get("KOSIS_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")

# Qwen3-Embedding 계열은 쿼리 쪽에 이 instruction을 붙여야 검색 성능이 나온다(권장 사용법).
# 문서(표 설명) 쪽은 instruction 없이 그대로 인코딩한다.
_QWEN_QUERY_INSTRUCTION = (
    "Given a Korean news claim sentence, retrieve the KOSIS statistical table "
    "description that best matches it"
)


class EmbeddingError(RuntimeError):
    """임베딩 모델 호출 실패."""


_model_singleton = None  # SentenceTransformer 인스턴스 lazy 캐시 (프로세스당 1회만 로딩)


def _get_embedding_model():
    """Qwen3-Embedding을 lazy하게 로딩해서 프로세스 전체에서 재사용한다."""
    global _model_singleton
    if _model_singleton is None:
        from sentence_transformers import SentenceTransformer  # 없으면 ImportError -> 폴백

        _model_singleton = SentenceTransformer(EMBEDDING_MODEL)
    return _model_singleton


# ---------------------------------------------------------------------------
# 실제 임베딩 모델 연동 지점. Qwen3-Embedding을 로컬에서 호출한다.
# sentence-transformers/torch가 없거나 모델 로딩·추론에 실패하면 해시 기반 더미 벡터로
# 폴백해서 "파이프라인이 끊기지 않고 돌아가는지"는 항상 보장한다.
# ---------------------------------------------------------------------------
def embed_texts(
    texts: list[str], *, model: str = EMBEDDING_MODEL, is_query: bool = False
) -> list[list[float]]:
    try:
        st_model = _get_embedding_model()
        inputs = (
            [f"Instruct: {_QWEN_QUERY_INSTRUCTION}\nQuery: {t}" for t in texts]
            if is_query
            else texts
        )
        return st_model.encode(inputs, convert_to_numpy=True).tolist()
    except Exception as exc:  # noqa: BLE001 - 로딩/추론 실패는 전부 더미 폴백 대상
        print(f"[embedding_search] Qwen3-Embedding 사용 불가({exc!r}) - 더미 벡터로 폴백합니다.")

    # --- 폴백: 해시 기반 더미 임베딩 (개발/테스트 전용, 의미 유사도는 반영 안 됨) ---
    dim = 64
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * dim
        for token in text:
            idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        vectors.append([v / norm for v in vec])
    return vectors


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


def _load_catalog(path: Path = CATALOG_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다.")
    return json.loads(path.read_text(encoding="utf-8"))["tables"]


def build_table_embedding_cache(
    *,
    catalog_path: Path = CATALOG_PATH,
    cache_path: Path = EMBEDDING_CACHE_PATH,
    force: bool = False,
) -> dict:
    """table_catalog.json의 embedding_text를 최초 1회 임베딩해서 캐시 파일로 저장한다.

    이미 캐시가 있고 표 개수·모델명이 그대로면 재임베딩하지 않는다 (배치 임베딩 원칙).
    모델을 바꾸면(EMBEDDING_MODEL 변경) 캐시가 자동으로 무효화되어 재생성된다.
    """
    tables = _load_catalog(catalog_path)

    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("model") == EMBEDDING_MODEL and len(cached.get("entries", [])) == len(tables):
            return cached

    texts = [t["embedding_text"] for t in tables]
    vectors = embed_texts(texts, is_query=False)

    entries = [
        {
            "table_id": t["tblId"],
            "table_name": t["title"],
            "required_slots": t.get("required_slots", []),
            "embedding_text": t["embedding_text"],
            "vector": vec,
        }
        for t, vec in zip(tables, vectors)
    ]
    cache = {"model": EMBEDDING_MODEL, "entries": entries}
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def embedding_search(
    claim: Claim,
    *,
    top_k: int = 5,
    cache: Optional[dict] = None,
) -> list[TableCandidate]:
    """Claim 1건을 임베딩해서 캐시된 표 벡터들과 코사인 유사도로 top-k를 반환한다."""
    cache = cache or build_table_embedding_cache()
    query_vec = embed_texts([claim.sentence], is_query=True)[0]

    scored: list[TableCandidate] = []
    for entry in cache["entries"]:
        sim = _cosine_similarity(query_vec, entry["vector"])
        scored.append(
            TableCandidate(
                table_id=entry["table_id"],
                table_name=entry["table_name"],
                score=sim,
                required_slots=entry.get("required_slots", []),
                source_meta=f"embedding_search model={cache.get('model')}",
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    # python -m agent.mapping.embedding_search
    test_claims = [
        Claim(sentence="지난달 청년 실업률이 6%에 육박했다", claim_type="규모"),
        Claim(sentence="취업자 수가 46개월 만에 감소 전환했다", claim_type="증감률"),
        Claim(sentence="지난달 소비자물가가 전년 동월 대비 2.2% 올랐다", claim_type="증감률"),
        Claim(sentence="전국 주택 매매가격이 지수화 기준으로 하락세를 보였다", claim_type="비교"),
        Claim(sentence="출생아 수가 14.6% 증가했다", claim_type="증감률"),
    ]
    cache = build_table_embedding_cache()
    for c in test_claims:
        results = embedding_search(c, cache=cache)
        print(f"\n[{c.sentence}]")
        for r in results[:3]:
            print(f"  - {r.table_name} ({r.table_id}) score={r.score:.3f}")
