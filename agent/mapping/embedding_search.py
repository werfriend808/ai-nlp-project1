"""
agent/mapping/embedding_search.py — 3단계: 임베딩 기반 top-k 검색

팀 계약(interfaces.py) 기준:
입력: Claim 1건
출력: TableCandidate의 리스트 (top-k, 코사인 유사도 기준)

모델: intfloat/multilingual-e5-large (notebooks/embedding_model_comparison.ipynb 비교
실험 결과 채택. 560M 파라미터. e5 계열 권장 사용법대로 쿼리 쪽엔 "query: ", 문서 쪽엔
"passage: " 프리픽스를 붙여 비대칭 인코딩한다 — 안 붙이면 검색 성능이 크게 떨어진다)

⚠️ 배치 임베딩 원칙(Day2 09:00-10:00 작업):
table_catalog.json의 embedding_text는 최초 1회만 임베딩해서 캐시 파일(TABLE_EMBEDDING_CACHE)에
저장한다. 검색할 때마다 표 20여 개를 매번 재임베딩하지 않는다 — 여기서 실제로 모델을
다시 부르는 건 "새 Claim 문장" 하나뿐이다. 캐시는 모델명이 바뀌면 자동으로 재생성된다.

사전 준비물:
    pip install sentence-transformers torch
    sentence-transformers/torch가 없거나 모델 로딩에 실패하면 해시 기반 더미 벡터로
    자동 폴백한다 (의미 유사도가 반영되지 않으니 개발/테스트 전용).
"""

from __future__ import annotations

import os

# 임베딩 모델(SentenceTransformer)과 리랭커(CrossEncoder)를 같은 프로세스에서 함께 로딩하면
# Windows에서 OpenMP 런타임(libiomp5md.dll)이 중복 로드되어 세그폴트가 난다(2026-08-03 확인,
# test_mapping.py가 임베딩 모델 로딩 직후 예외 없이 죽던 원인). torch가 로딩되기 전에
# 미리 설정해야 하므로 다른 import보다 먼저 둔다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Hugging Face Hub CDN(CloudFront)으로의 연결이 SSL 소켓 connect() 단계에서 무한 대기하는
# 현상을 실측으로 확인함(2026-08-05, 로컬 네트워크의 IPv6 라우팅 문제로 추정 — 같은 호스트에
# IPv4로는 즉시 연결되는데 IPv6로는 연결 자체가 안 됨). socket.getaddrinfo가 IPv4 주소만
# 반환하도록 패치하고 기본 타임아웃을 걸어서, 최악의 경우에도 무한 멈춤 대신 타임아웃
# 예외로 끝나게 한다. torch/sentence-transformers가 로딩되기 전에 걸어야 하므로 여기서 설정.
import socket  # noqa: E402

if not getattr(socket, "_ipv4_only_patched", False):
    _original_getaddrinfo = socket.getaddrinfo
    # agent/kosis/query_vdb.py가 chromadb HttpClient 연결 시 이 패치를 잠깐 되돌려야 해서
    # (아래 패치가 걸려있으면 로컬 Chroma 서버 연결이 깨지는 게 실측 확인됨, 2026-08-15),
    # 원본 함수를 socket 모듈에 속성으로 남겨서 다른 모듈에서도 복구할 수 있게 한다.
    socket._original_getaddrinfo = _original_getaddrinfo

    def _getaddrinfo_ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _getaddrinfo_ipv4_only
    socket._patched_getaddrinfo = _getaddrinfo_ipv4_only
    socket.setdefaulttimeout(30)
    socket._ipv4_only_patched = True

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

try:
    from agent.interfaces import Claim, TableCandidate, dense_query_text
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class Claim:  # type: ignore[no-redef]
        sentence: str
        claim_type: str
        period: Optional[str] = None
        unit: Optional[str] = None
        population: Optional[str] = None
        prev_sentence: Optional[str] = None

    @dataclass
    class TableCandidate:  # type: ignore[no-redef]
        table_id: str
        table_name: str
        score: float
        required_slots: list = field(default_factory=list)
        source_meta: Optional[str] = None

    def dense_query_text(claim) -> str:  # type: ignore[no-redef]
        """agent.interfaces.dense_query_text의 단독 실행용 폴백(로직 동일, Context D2)."""
        prev = getattr(claim, "prev_sentence", None)
        return f"{prev} {claim.sentence}".strip() if prev else claim.sentence

CATALOG_PATH = Path(__file__).parent / "table_catalog.json"
EMBEDDING_CACHE_PATH = Path(__file__).parent / "table_embeddings_cache.json"
EMBEDDING_MODEL = os.environ.get("KOSIS_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

# 일부 환경(CPU/torch 스레딩 조합 등)에서 임베딩 모델 인코딩 호출이 예외 없이 그냥
# 멈춰버리는 경우가 있다 (모델 로딩 자체는 끝나는데 encode() 연산에서 무한 대기 — 세그폴트와
# 달리 크래시도 아니라서 아래 try/except로도 못 잡는다). 이럴 땐 .env에
# KOSIS_DISABLE_EMBEDDING=1을 넣어서 모델 시도 자체를 건너뛰고 더미 벡터 폴백으로 바로
# 간다 (keyword_search가 이미 찾은 표는 정상 동작, 못 찾은 표만 "통계 없음"으로 처리됨).
_DISABLE_EMBEDDING = os.environ.get("KOSIS_DISABLE_EMBEDDING", "").strip().lower() in ("1", "true", "yes")

# e5 계열은 쿼리/문서를 서로 다른 프리픽스로 인코딩해야 검색 성능이 나온다(권장 사용법).
_E5_QUERY_PREFIX = "query: "
_E5_PASSAGE_PREFIX = "passage: "


class EmbeddingError(RuntimeError):
    """임베딩 모델 호출 실패."""


_model_singleton = None  # SentenceTransformer 인스턴스 lazy 캐시 (프로세스당 1회만 로딩)


def _get_embedding_model():
    """임베딩 모델을 lazy하게 로딩해서 프로세스 전체에서 재사용한다."""
    global _model_singleton
    if _model_singleton is None:
        from sentence_transformers import SentenceTransformer  # 없으면 ImportError -> 폴백

        _model_singleton = SentenceTransformer(EMBEDDING_MODEL)
    return _model_singleton


# ---------------------------------------------------------------------------
# 실제 임베딩 모델 연동 지점. multilingual-e5-large를 로컬에서 호출한다. e5 계열은
# 쿼리 쪽엔 "query: ", 문서 쪽엔 "passage: " 프리픽스를 붙여야 검색 성능이 나온다.
# sentence-transformers/torch가 없거나 모델 로딩·추론에 실패하면 해시 기반 더미 벡터로
# 폴백해서 "파이프라인이 끊기지 않고 돌아가는지"는 항상 보장한다.
# ---------------------------------------------------------------------------
def embed_texts(
    texts: list[str], *, model: str = EMBEDDING_MODEL, is_query: bool = False
) -> list[list[float]]:
    if not _DISABLE_EMBEDDING:
        try:
            st_model = _get_embedding_model()
            prefix = _E5_QUERY_PREFIX if is_query else _E5_PASSAGE_PREFIX
            inputs = [prefix + t for t in texts]
            return st_model.encode(inputs, convert_to_numpy=True).tolist()
        except Exception as exc:  # noqa: BLE001 - 로딩/추론 실패는 전부 더미 폴백 대상
            print(f"[embedding_search] {EMBEDDING_MODEL} 사용 불가({exc!r}) - 더미 벡터로 폴백합니다.")

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

    이미 캐시가 있고 표별 embedding_text·모델명이 그대로면 재임베딩하지 않는다(배치 임베딩
    원칙). 모델을 바꾸거나(EMBEDDING_MODEL 변경) 표를 추가/삭제하거나 기존 표의
    embedding_text 내용만 수정해도 캐시가 자동으로 무효화되어 재생성된다 — 예전엔 표
    개수만 비교해서, 표 개수 변화 없이 embedding_text만 고치면(카탈로그 메타데이터 보강
    작업에서 흔함) 캐시가 갱신 안 된 채로 조용히 재사용되는 버그가 있었다(2026-08-05,
    keywords만 보강한 6개 표 때문에 force=True를 수동으로 줘야 했던 사례로 확인).
    """
    tables = _load_catalog(catalog_path)
    current_texts = {t["tblId"]: t["embedding_text"] for t in tables}

    if cache_path.exists() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_texts = {e["table_id"]: e.get("embedding_text") for e in cached.get("entries", [])}
        if cached.get("model") == EMBEDDING_MODEL and cached_texts == current_texts:
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
    query_vec = embed_texts([dense_query_text(claim)], is_query=True)[0]

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


def batch_embedding_search(
    sentences: list[str], *, top_k: int = 5, cache: Optional[dict] = None
) -> list[list[TableCandidate]]:
    """여러 claim 문장을 한 번의 encode() 호출로 임베딩해서, 각각의 top-k를 반환한다.

    2026-08-15 실측: embedding_search()를 claim마다 반복 호출하면(=encode()를 여러 번
    나눠 부르면) 로컬에서 세그폴트가 재현됐지만, 문장 여러 개를 한 번의 encode() 호출에
    몰아서 넘기면(40개 배치 테스트로 확인) 문제없이 안전했다 — build_table_embedding_cache()
    가 표 64개를 한 번에 배치 임베딩할 때도 항상 안전했던 것과 같은 패턴. 그래서 claim이
    여러 개일 때는 이 함수로 한 번에 처리하고, embedding_search()(단건)는 claim이 하나뿐인
    경우에만 쓴다."""
    cache = cache or build_table_embedding_cache()
    if not sentences:
        return []
    query_vecs = embed_texts(sentences, is_query=True)

    results: list[list[TableCandidate]] = []
    for query_vec in query_vecs:
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
        results.append(scored[:top_k])
    return results


def embed_sentences_batch(sentences: list[str]) -> list[list[float]]:
    """claim 문장 여러 개를 한 번의 encode() 호출로 쿼리 임베딩만 반환한다(카탈로그 비교 없이).

    KOSIS VDB(Chroma, agent/kosis/chroma_db) 조회용 쿼리 벡터를 만들 때 쓴다 — 64개
    카탈로그 매칭과 같은 모델(e5-large)로 만들어야 벡터 공간이 맞는다."""
    if not sentences:
        return []
    return embed_texts(sentences, is_query=True)


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
