"""
agent/kosis/query_vdb.py — KOSIS 표 28만7천여 개 VDB(Supabase/pgvector)를 미리 계산된
쿼리 벡터로 조회한다.

2026-08-18: Chroma(로컬 서버 전용)에서 Supabase(pgvector)로 옮겼다 — 코랩 등 외부에서도
접근 가능해서, 이전엔 로컬 Chroma 서버에 못 붙어 VDB를 건너뛰어야 했던 배치 경로에서도
이제 VDB를 정상적으로 쓸 수 있다. .env의 SUPABASE_DB_URL로 접속한다.

이 모듈은 임베딩 모델을 직접 불러오지 않는다 — 쿼리 벡터는 이미 다른 곳에서 만들어 온
것을 받기만 한다. 쿼리 벡터의 모델과 VDB에 적재된 벡터의 모델(vdb_embedding_colab.ipynb)이
반드시 같아야 한다 — 다르면 벡터 공간이 안 맞아서 유사도 값 자체가 무의미해진다.

⚠️ 2026-08-19: VDB를 Qwen3-Embedding-4B(truncate_dim=1024)로 바꾸면서, 쿼리 벡터도
더 이상 embedding_search.embed_sentences_batch()(로컬, e5 계열 — 64개 카탈로그 매칭
전용으로 남음)로 만들면 안 된다. Qwen3-Embedding-4B는 로컬(RAM 7.4GB)에서 못 돌아가서,
claim 쿼리도 코랩에서 같은 모델·같은 truncate_dim으로 임베딩해야 한다 — 이 부분은
아직 export_for_rerank.py/reranker_colab.ipynb 쪽에 반영 전이다(진행 중인 작업).
"""

from __future__ import annotations

import os
from typing import Optional

import psycopg2

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

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
        org_id: Optional[str] = None

TABLE_NAME = "kosis_vdb_tables"

# 28만여 개 중 상위 몇 개까지 후보로 볼지.
# 2026-08-17 변경(3 -> 10): 근사 최근접 이웃 검색(HNSW)이 실제로 진짜 정답을 상위 3등
# 안에 못 넣는 사례가 실측 확인됐다. VDB 후보는 어차피 "unverified"로 표시돼서 리랭커가
# 다시 평가하므로(_merge_candidates 참고), 여기서 늘려도 잘못된 표가 부당하게 승격될
# 위험은 없다 — 그냥 리랭커가 볼 후보 풀이 넓어질 뿐이다.
# 2026-08-21(PHASE 8): 환경변수 DENSE_TOP_K로 배포 시점에 조정 가능하게 함. 기본값은
# 그동안 골든셋으로 검증해온 10을 그대로 유지(하위 호환).
VDB_TOP_K = int(os.environ.get("DENSE_TOP_K", "10"))
# 절대 유사도 하한선. "28만개 중 그나마 제일 비슷한 것"이어도 이 밑이면 사실상 무관하다고
# 보고 후보에서 제외한다 — top-k 개수 제한만으로는 못 거르는 노이즈를 추가로 막는다.
# 2026-08-19: 0.75는 e5-small 기준 값이었다 — Qwen3-Embedding-4B로 바뀐 뒤 실측 기준
# reranker_colab.ipynb에서 0.5로 재조정됐는데 이 모듈엔 반영이 안 돼 있었다(로컬 경로와
# 코랩 경로의 판정 기준이 갈라지면 안 되므로 동일하게 맞춘다).
VDB_MIN_SIMILARITY = 0.5

# 2026-08-19: VDB 전용 임베딩 모델은 64개 카탈로그용(KOSIS_EMBEDDING_MODEL, e5 계열)과
# 별개다 — VDB만 Qwen3-Embedding-4B(truncate_dim=1024)로 바꿨다. 용도가 완전히 분리돼
# 있어서(서로 비교되는 벡터쌍이 아님) 같은 환경변수를 재사용하면 안 된다.
_SOURCE_MODEL = os.environ.get("KOSIS_VDB_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")

_conn = None  # 커넥션은 프로세스당 한 번만 만들어 재사용


def _get_connection():
    global _conn
    if _conn is None or _conn.closed:
        db_url = os.environ.get("SUPABASE_DB_URL")
        if not db_url:
            raise VdbUnavailableError(
                "SUPABASE_DB_URL이 없습니다. .env에 Supabase 프로젝트의 Postgres 연결 문자열을 넣으세요."
            )
        _conn = psycopg2.connect(db_url)
        # 이 커넥션은 읽기 전용(SELECT만)이라 autocommit으로 둔다 — 안 그러면 매 쿼리가
        # 트랜잭션을 열어놓은 채 커밋 없이 남는다(2026-08-21, lexical_query_vdb의
        # `set pg_trgm.similarity_threshold`도 세션 단위라 커넥션 재사용 중 계속
        # 살아있어야 함 — autocommit이면 SET 자체도 별도 커밋 없이 즉시 반영된다).
        _conn.autocommit = True
        with _conn.cursor() as cur:
            cur.execute("set pg_trgm.similarity_threshold = 0.1;")
    return _conn


class VdbUnavailableError(RuntimeError):
    """Supabase(pgvector)에 연결할 수 없음 (.env의 SUPABASE_DB_URL 확인 필요)."""


def batch_query_vdb(
    query_vectors: list[list[float]], *, top_k: int = VDB_TOP_K
) -> list[list[TableCandidate]]:
    """쿼리 벡터 여러 개를 하나씩 pgvector에 조회한다(claim 개수만큼의 후보 리스트 반환).

    Supabase에 연결할 수 없으면 VdbUnavailableError를 낸다 — 이 경우 호출부가 VDB
    없이(64개 카탈로그만으로) 계속 진행할지 말지 판단해야 한다."""
    if not query_vectors:
        return []

    try:
        conn = _get_connection()
    except psycopg2.OperationalError as e:
        raise VdbUnavailableError(f"Supabase(pgvector)에 연결 못 했습니다: {e}") from e

    results: list[list[TableCandidate]] = []
    try:
        with conn.cursor() as cur:
            for query_vec in query_vectors:
                # pgvector의 `<=>`는 코사인 거리(1 - 코사인 유사도)를 돌려준다(정규화된
                # 벡터 기준) — Chroma 때와 같은 변환식(유사도 = 1 - 거리)을 그대로 쓴다.
                cur.execute(
                    f"""
                    select tbl_id, org_id, text, embedding <=> %s::vector as distance
                    from {TABLE_NAME}
                    order by embedding <=> %s::vector
                    limit %s;
                    """,
                    (query_vec, query_vec, top_k),
                )
                rows = cur.fetchall()

                candidates = []
                for tbl_id, org_id, text, dist in rows:
                    similarity = 1.0 - float(dist)
                    if similarity < VDB_MIN_SIMILARITY:
                        continue
                    candidates.append(
                        TableCandidate(
                            table_id=tbl_id,
                            table_name=text,
                            score=similarity,
                            required_slots=[],
                            source_meta=f"kosis_vdb model={_SOURCE_MODEL}",
                            org_id=org_id or None,
                        )
                    )
                results.append(candidates)
    except psycopg2.Error as e:
        # 쿼리 도중 끊기면(SSL 종료 등) 커넥션이 "aborted transaction" 상태로 남는다 —
        # .closed는 여전히 0이라 _get_connection()이 이 망가진 커넥션을 계속 재사용해서,
        # 다음 claim부터는 매번 InFailedSqlTransaction으로 즉시 죽는 문제가 실측됐다
        # (2026-08-19, rerank_local.py 로컬 실행 중). 다음 호출이 새 커넥션을 만들도록
        # 여기서 확실히 버린다.
        global _conn
        try:
            conn.close()
        except Exception:
            pass
        _conn = None
        raise VdbUnavailableError(f"Supabase(pgvector) 조회 중 연결 오류: {e}") from e

    return results


# 2026-08-21 도입: VDB(28.7만 개)에도 어휘 검색을 추가한다 — 지금까지 keyword_search는
# 64개 수동 카탈로그만 보고 VDB는 임베딩 단독이었는데, "기관명이 정확히 일치" 같은
# 신호는 임베딩 코사인 유사도보다 정확한 단어 매칭이 훨씬 더 잘 잡는다(형제 표 구분
# 실패의 상당수가 이 신호 부재 때문이었음). Postgres에 정식 BM25 확장은 없어서
# pg_trgm(문자 단위 trigram 유사도)으로 대체 — 형태소 분석 없이 한국어에도 그대로
# 적용된다. 사전 준비: `create extension pg_trgm;` +
# `create index kosis_vdb_tables_text_trgm_idx on kosis_vdb_tables using gin (text gin_trgm_ops);`
# (이미 적용됨, 2026-08-21).
# 2026-08-21(PHASE 8): 환경변수 BM25_TOP_K로 배포 시점에 조정 가능하게 함(DENSE_TOP_K와
# 동일 원칙). 기본값은 골든셋 검증에 쓴 30 유지.
LEXICAL_TOP_K = int(os.environ.get("BM25_TOP_K", "30"))


def lexical_query_vdb(query_text: str, *, top_k: int = LEXICAL_TOP_K) -> list[TableCandidate]:
    """trigram 유사도로 VDB를 텍스트 검색한다. query_text는 호출부가 준비한다 —
    claim.sentence 원문 그대로 넘길 수도, source_org/statistic_expression 등을 이어붙인
    구조화 쿼리를 넘길 수도 있다(이 함수는 무엇이 오든 그대로 유사도 비교만 한다)."""
    if not query_text or not query_text.strip():
        return []

    try:
        conn = _get_connection()
    except psycopg2.OperationalError as e:
        raise VdbUnavailableError(f"Supabase(pgvector)에 연결 못 했습니다: {e}") from e

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                select tbl_id, org_id, text, similarity(text, %s) as sim
                from {TABLE_NAME}
                where text %% %s
                order by sim desc
                limit %s;
                """,
                (query_text, query_text, top_k),
            )
            rows = cur.fetchall()
    except psycopg2.Error as e:
        global _conn
        try:
            conn.close()
        except Exception:
            pass
        _conn = None
        raise VdbUnavailableError(f"Supabase(pgvector) 텍스트 검색 중 연결 오류: {e}") from e

    return [
        TableCandidate(
            table_id=tbl_id,
            table_name=text,
            score=float(sim),
            required_slots=[],
            source_meta="kosis_vdb_lexical(pg_trgm)",
            org_id=org_id or None,
        )
        for tbl_id, org_id, text, sim in rows
    ]


# 2026-08-23 도입: 표 "제목"이 검색어와 전혀 안 겹치는 경우(원인 A — 예: "생애주기적자계정"
# 표 안에 "소비"/"노동소득" 항목이 있는데 표 제목엔 그 단어가 없음)를 보완한다. 지금까지의
# 4개 채널은 전부 표 제목 텍스트와 검색어를 비교했는데, 이 채널은 표 "항목"(agent.kosis
# .enrich_objl.fetch_item_names 참고) 하나하나를 독립적으로 임베딩해서 비교한다 — 표
# 제목에 여러 항목을 다 붙이면(예전에 시도했다 되돌린 방식) 항목 하나의 신호가 희석되지만,
# 항목을 표와 별개 행으로 저장하면 희석 없이 비교된다(실측 확인: DT_1NTA2002의 "노동소득"
# 항목 단독 유사도 0.376 vs 표 제목 전체 0.257).
#
# ⚠️ 검증된 전제: 이 채널의 쿼리 벡터는 다른 4개 채널과 같은 (블렌딩된) search_query가
# 아니라 claim.statistic_expression만 써야 한다 — 블렌딩된 검색어로 비교하면 항목
# 하나짜리 좁은 텍스트와는 안 맞아서(실측: "노동소득 민간 소비 전국"으로 비교하면
# "노동소득" 항목이 0.259로 오히려 안 좋아짐, "노동소득"만 쓰면 0.376으로 1위) 호출부가
# 반드시 별도로 좁은 쿼리 벡터를 만들어 넘겨야 한다.
ITEM_TOP_K = int(os.environ.get("ITEM_TOP_K", "10"))
ITEM_MIN_SIMILARITY = float(os.environ.get("ITEM_MIN_SIMILARITY", "0.3"))
ITEMS_TABLE_NAME = "kosis_vdb_items"


def item_query_vdb(query_vec: list[float], *, top_k: int = ITEM_TOP_K) -> list[TableCandidate]:
    """항목(item) 임베딩 인덱스에서 쿼리 벡터와 가장 가까운 항목들을 찾아, 그 항목이 속한
    표를 후보로 돌려준다. 한 표에 항목이 여러 개 걸리면 가장 높은 점수만 남긴다(표
    단위로 중복 제거 — RRF 합치기 쪽은 표 단위 후보 리스트를 기대한다).

    table_name은 kosis_vdb_tables와 조인해서 채운다(kosis_vdb_items는 항목 이름만 들고
    있음) — 인덱싱 시점에 중복 저장 안 해서 표 제목이 나중에 바뀌어도 항상 최신을 본다."""
    if not query_vec:
        return []

    try:
        conn = _get_connection()
    except psycopg2.OperationalError as e:
        raise VdbUnavailableError(f"Supabase(pgvector)에 연결 못 했습니다: {e}") from e

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                select i.tbl_id, i.org_id, i.item_name, t.text,
                       i.embedding <=> %s::vector as distance
                from {ITEMS_TABLE_NAME} i
                join {TABLE_NAME} t on t.tbl_id = i.tbl_id
                order by i.embedding <=> %s::vector
                limit %s;
                """,
                (query_vec, query_vec, top_k * 3),  # 표 단위 중복 제거 전이라 넉넉히 가져옴
            )
            rows = cur.fetchall()
    except psycopg2.Error as e:
        global _conn
        try:
            conn.close()
        except Exception:
            pass
        _conn = None
        raise VdbUnavailableError(f"Supabase(pgvector) 항목 검색 중 연결 오류: {e}") from e

    best_per_table: dict[str, tuple] = {}
    for tbl_id, org_id, item_name, table_text, dist in rows:
        similarity = 1.0 - float(dist)
        if similarity < ITEM_MIN_SIMILARITY:
            continue
        prev = best_per_table.get(tbl_id)
        if prev is None or similarity > prev[0]:
            best_per_table[tbl_id] = (similarity, org_id, item_name, table_text)

    ranked = sorted(best_per_table.items(), key=lambda kv: -kv[1][0])[:top_k]
    return [
        TableCandidate(
            table_id=tbl_id,
            table_name=table_text,
            score=similarity,
            required_slots=[],
            # 2026-08-24: score도 같이 남긴다 — reranker.py의 신뢰 게이트가 item_rank
            # "순위"만으론 노이즈를 못 거른다(표 1개만 인덱싱된 지금은 item_rank가 있으면
            # 사실상 항상 1등이라 순위 자체가 무의미). 유사도 점수로 문턱을 걸어야 한다
            # (골든셋 실측: 정답 0.3676~0.4810 vs 무관한 오탐 0.30~0.3584, 깨끗하게 분리됨).
            source_meta=f"kosis_vdb_item(matched='{item_name}', score={similarity:.4f})",
            org_id=org_id or None,
        )
        for tbl_id, (similarity, org_id, item_name, table_text) in ranked
    ]
