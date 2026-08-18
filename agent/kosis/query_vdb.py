"""
agent/kosis/query_vdb.py — KOSIS 표 28만7천여 개 Chroma VDB를 미리 계산된 쿼리 벡터로 조회한다.

Chroma는 로컬 서버 모드로 띄워야 한다(agent/kosis/build_vdb_index.py 참고, 대용량에서
임베디드 PersistentClient가 인덱스 손상을 일으키는 문제 실측 확인됨):
    chroma run --path agent/kosis/chroma_db --port 8100

이 모듈은 임베딩 모델을 직접 불러오지 않는다 — 쿼리 벡터는 이미 다른 곳(예:
embedding_search.embed_sentences_batch, 64개 카탈로그 매칭과 같은 e5-large 배치 호출)에서
만들어 온 것을 받기만 한다. 그래야 임베딩 모델 로딩/호출이 한 곳(배치)으로만 몰려서
반복 호출로 인한 세그폴트 위험을 피한다.
"""

from __future__ import annotations

from typing import Optional

try:
    from agent.interfaces import TableCandidate
except ImportError:
    from dataclasses import dataclass, field

    @dataclass
    class TableCandidate:  # type: ignore[no-redef]
        table_id: str
        table_name: str
        score: float
        required_slots: list = field(default_factory=list)
        source_meta: Optional[str] = None

CHROMA_HOST = "localhost"
CHROMA_PORT = 8100
COLLECTION_NAME = "kosis_vdb_tables"

# 28만여 개 중 상위 몇 개까지 후보로 볼지.
# 2026-08-17 변경(3 -> 10): Chroma(HNSW 근사 검색)가 실제로 진짜 정답을 상위 3등 안에
# 못 넣는 사례가 실측 확인됐다(브루트포스 기준 13등인 표가 search_ef=5000에서도 전혀
# 안 잡힘 — 이 사례 자체는 top_k를 늘려도 해결 안 되는 별개의 인덱스 파라미터 문제로 남아
# 있지만, "진짜 정답이 4~10등 사이인" 훨씬 흔한 완화된 케이스는 top_k를 넓히면 구제된다).
# VDB 후보는 어차피 "unverified"로 표시돼서 리랭커가 다시 평가하므로(_merge_candidates
# 참고), 여기서 늘려도 잘못된 표가 부당하게 승격될 위험은 없다 — 그냥 리랭커가 볼 후보
# 풀이 넓어질 뿐이다. 너무 키우면 노이즈(28만개 중 상위권 밖은 코사인 유사도가 사실상
# 무관한 값)와 claim당 리랭커 호출 비용만 늘어나므로, keyword/embedding-64(top 5)보다
# 약간 넓은 10 정도로 절충한다.
VDB_TOP_K = 10
# 절대 유사도 하한선. "28만개 중 그나마 제일 비슷한 것"이어도 이 밑이면 사실상 무관하다고
# 보고 후보에서 제외한다 — top-k 개수 제한만으로는 못 거르는 노이즈를 추가로 막는다.
VDB_MIN_SIMILARITY = 0.75

_client = None  # PersistentClient/HttpClient는 프로세스당 한 번만 만들어 재사용


def _get_client():
    global _client
    if _client is None:
        import socket

        import chromadb

        # 2026-08-15 실측: agent/mapping/embedding_search.py가 HF Hub 접속 안정화를 위해
        # socket.getaddrinfo를 IPv4 전용으로 몰래 패치해두는데, 이게 chromadb의 HttpClient
        # (httpx 기반)의 로컬 서버 연결을 깨뜨리는 걸 확인했다("Could not connect to a
        # Chroma server" — 패치 없이는 정상 연결됨, 패치만 있으면 host를 localhost든
        # 127.0.0.1로 직접 줘도 재현). 정확한 내부 원인은 httpx/anyio 쪽이라 못 밝혔지만,
        # 클라이언트를 만드는 짧은 순간만 원래 getaddrinfo로 되돌렸다가 복구하면 우회된다.
        patched = getattr(socket, "_ipv4_only_patched", False)
        original = getattr(socket, "_original_getaddrinfo", None)
        patched_fn = getattr(socket, "_patched_getaddrinfo", None)
        if patched and original is not None:
            socket.getaddrinfo = original
        try:
            _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        finally:
            if patched and patched_fn is not None:
                socket.getaddrinfo = patched_fn
    return _client


class VdbUnavailableError(RuntimeError):
    """Chroma 서버에 연결할 수 없음 (chroma run으로 띄웠는지 확인 필요)."""


def batch_query_vdb(
    query_vectors: list[list[float]], *, top_k: int = VDB_TOP_K
) -> list[list[TableCandidate]]:
    """쿼리 벡터 여러 개를 한 번에 Chroma에 조회한다(claim 개수만큼의 후보 리스트 반환).

    Chroma 서버가 안 떠 있으면 VdbUnavailableError를 낸다 — 이 경우 호출부가 VDB
    없이(64개 카탈로그만으로) 계속 진행할지 말지 판단해야 한다."""
    if not query_vectors:
        return []

    # 2026-08-15 실측: 같은 프로세스에서 임베딩 모델(torch/sentence-transformers)을 막
    # 로딩한 직후 바로 Chroma HTTP 연결을 시도하면 일시적으로 연결이 실패하는 현상이
    # 있었다(단독으로 다시 시도하면 바로 성공) — 리소스 경합으로 인한 일시적 문제로
    # 추정되어, 짧은 재시도를 넣는다.
    import time

    last_error: Optional[Exception] = None
    collection = None
    for attempt in range(3):
        try:
            client = _get_client()
            collection = client.get_collection(COLLECTION_NAME)
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            time.sleep(2)
    if collection is None:
        raise VdbUnavailableError(
            f"Chroma 서버({CHROMA_HOST}:{CHROMA_PORT})에 연결 못 했습니다(3회 재시도): {last_error}\n"
            f"'chroma run --path agent/kosis/chroma_db --port {CHROMA_PORT}'로 먼저 띄우세요."
        ) from last_error

    raw = collection.query(query_embeddings=query_vectors, n_results=top_k)

    results: list[list[TableCandidate]] = []
    for ids, docs, dists in zip(raw["ids"], raw["documents"], raw["distances"]):
        candidates = []
        for tbl_id, doc, dist in zip(ids, docs, dists):
            # Chroma는 cosine space에서 거리(distance)를 반환한다 — 유사도 = 1 - 거리.
            similarity = 1.0 - dist
            if similarity < VDB_MIN_SIMILARITY:
                continue
            candidates.append(
                TableCandidate(
                    table_id=tbl_id,
                    table_name=doc,
                    score=similarity,
                    required_slots=[],
                    source_meta="kosis_vdb model=intfloat/multilingual-e5-large",
                )
            )
        results.append(candidates)
    return results
