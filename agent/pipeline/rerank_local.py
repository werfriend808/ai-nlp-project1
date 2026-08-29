"""
agent/pipeline/rerank_local.py — notebooks/reranker_colab.ipynb의 3-1(VDB 매칭)+리랭킹
셀을 로컬에서 직접 돌리기 위한 스크립트. 사양이 높은(RAM 16GB+ 권장, 이상적으로는 GPU)
컴퓨터용이다 — 이 프로젝트 기준 컴퓨터(RAM 7.4GB)에서는 Qwen3-Embedding-4B와
bge-reranker-v2-m3 둘 다 세그멘테이션 폴트로 죽어서(2026-08-19/2026-08-13 각각 실측)
코랩으로 우회했는데, 사양이 충분하면 굳이 코랩을 거칠 필요가 없다.

세그폴트는 OS 레벨 크래시라 파이썬에서 못 잡는다 — 그래서 "로컬로 시도해보고 실패하면
코랩으로 자동 전환" 같은 로직은 애초에 구현이 불가능하다(죽으면 프로세스 자체가 그 자리에서
끝남). 대신 시작할 때 시스템 RAM만 가볍게 확인해서, 너무 적으면 경고만 띄우고 계속 진행 여부는
사용자 판단에 맡긴다.

입력: data/rerank_pending.json (export_for_rerank.py 출력 — keyword+64개 카탈로그 임베딩
병합까지 끝난 상태, VDB는 아직 안 섞임)
출력: data/rerank_results.json (resume_after_rerank.py가 그대로 읽는 포맷 — 코랩 경로로
받은 파일과 동일한 스키마이므로, 로컬/코랩 어느 쪽으로 실행했든 다음 단계는 안 갈린다)

VDB 조회 로직(쿼리 임베딩 모델·instruction prefix·유사도 하한선)과 리랭킹+RRF 융합 로직은
각각 agent/kosis/query_vdb.py, agent/mapping/reranker.py를 그대로 재사용한다 —
reranker_colab.ipynb 셀 e2921b28/dea39823과 동일한 결과가 나오게 하기 위함이다.

사용법 (프로젝트 루트에서):
    python -m agent.pipeline.rerank_local
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from agent.interfaces import Claim, TableCandidate
from agent.kosis.query_vdb import VDB_TOP_K, VdbUnavailableError, batch_query_vdb
from agent.mapping.reranker import build_retrieval_query, load_document_texts, rerank

PENDING_PATH = Path(__file__).parent.parent.parent / "data" / "rerank_pending.json"
RESULTS_PATH = Path(__file__).parent.parent.parent / "data" / "rerank_results.json"

# vdb_embedding_colab.ipynb가 만든 VDB(문서 쪽)와 짝이 맞아야 하는 쿼리 쪽 모델·설정 —
# reranker_colab.ipynb 셀 e2921b28과 반드시 동일하게 유지한다.
VDB_EMBED_MODEL_NAME = os.environ.get("KOSIS_VDB_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
VDB_EMBED_DIM = 2560
# 2026-08-27: 재구축된 VDB(kosis_vdb_tables_qwen)가 2560차원이라 쿼리 벡터도 2560으로
# 맞춰야 한다. 1024는 소실 이전 VDB 기준이며, 그대로 두면 차원 불일치로 검색이 실패한다.
VDB_QUERY_INSTRUCTION = (
    "Given a Korean news claim sentence, retrieve the KOSIS statistical table "
    "description that best matches it"
)

RAM_WARNING_THRESHOLD_GB = 16


def _check_ram() -> None:
    """시스템 RAM을 가볍게 확인해서, 부족해 보이면 경고만 띄운다(강제 차단 아님).

    세그폴트는 파이썬에서 못 잡으므로 이 체크가 안전을 보장하지 않는다 — 그냥 "이 밑이면
    위험할 수 있다"는 힌트만 준다. psutil 등 추가 의존성 없이 표준 라이브러리만 쓴다.
    """
    total_gb: float | None = None
    try:
        if sys.platform == "win32":
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_gb = stat.ullTotalPhys / (1024 ** 3)
        else:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_gb = int(line.split()[1]) / (1024 ** 2)
                        break
    except Exception:
        total_gb = None

    if total_gb is None:
        print("[RAM 확인] 시스템 메모리 크기를 확인하지 못했습니다 — 그냥 진행합니다.")
        return

    print(f"[RAM 확인] 시스템 RAM {total_gb:.1f}GB")
    if total_gb < RAM_WARNING_THRESHOLD_GB:
        print(
            f"[경고] RAM이 {RAM_WARNING_THRESHOLD_GB}GB 미만입니다 — Qwen3-Embedding-4B와 "
            "bge-reranker-v2-m3 로딩 중 세그멘테이션 폴트로 프로세스가 죽을 수 있습니다"
            "(파이썬에서 못 잡는 OS 레벨 크래시입니다). 대신 notebooks/reranker_colab.ipynb"
            "(코랩)로 실행하는 걸 권장합니다."
        )


_vdb_embed_model = None


def _get_vdb_embed_model():
    global _vdb_embed_model
    if _vdb_embed_model is None:
        from sentence_transformers import SentenceTransformer

        _vdb_embed_model = SentenceTransformer(VDB_EMBED_MODEL_NAME, truncate_dim=VDB_EMBED_DIM)
    return _vdb_embed_model


def _vdb_query_vec_for(query_text: str) -> list[float]:
    # Qwen3-Embedding 공식 권장 사용법: 쿼리 쪽에만 instruction prefix를 붙인다(문서 쪽은
    # vdb_embedding_colab.ipynb에서 prefix 없이 인코딩했음) — reranker_colab.ipynb와 동일.
    model = _get_vdb_embed_model()
    text = f"Instruct: {VDB_QUERY_INSTRUCTION}\nQuery: {query_text}"
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()


def _dict_to_candidate(d: dict) -> TableCandidate:
    return TableCandidate(
        table_id=d["table_id"],
        table_name=d["table_name"],
        score=d["score"],
        required_slots=d.get("required_slots", []),
        source_meta=d.get("source_meta"),
    )


def _candidate_to_dict(c: TableCandidate) -> dict:
    return {
        "table_id": c.table_id,
        "table_name": c.table_name,
        "score": c.score,
        "required_slots": c.required_slots,
        "source_meta": c.source_meta,
    }


def _merge_vdb_candidates(
    merged: list[TableCandidate], vdb_candidates: list[TableCandidate]
) -> list[TableCandidate]:
    """이미 keyword_rank/embedding_rank가 태그된 merged 후보에 vdb_rank만 추가로 얹는다
    (agent/mapping/reranker.py의 _merge_candidates()·reranker_colab.ipynb의
    add_vdb_to_merged()와 동일한 규칙)."""
    from dataclasses import replace

    by_id = {c.table_id: c for c in merged}
    for i, c in enumerate(vdb_candidates):
        tag = f"vdb_rank={i + 1}"
        existing = by_id.get(c.table_id)
        if existing is None:
            meta = f"{c.source_meta} | {tag}" if c.source_meta else tag
            by_id[c.table_id] = replace(c, source_meta=meta)
        else:
            by_id[c.table_id] = replace(
                existing, source_meta=f"{existing.source_meta} | {tag}"
            )
    return list(by_id.values())


def main() -> None:
    _check_ram()

    if not PENDING_PATH.exists():
        raise SystemExit(f"{PENDING_PATH}가 없습니다. 먼저 export_for_rerank.py를 실행하세요.")

    pending = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    catalog = pending["catalog"]
    document_texts = {tid: entry["embedding_text"] for tid, entry in catalog.items()}
    document_texts.update(load_document_texts())  # 64개 카탈로그 쪽은 원본 우선

    print(f"리랭킹 대상 claim {len(pending['items'])}건, 카탈로그 {len(catalog)}개 표")

    vdb_unavailable_warned = False
    output_items: list[dict] = []
    for i, item in enumerate(pending["items"]):
        claim = Claim(**item["claim"])
        merged = [_dict_to_candidate(d) for d in item.get("merged_candidates", [])]

        try:
            # 2026-08-28: 문맥 보강(앞 문장)·기관 별칭까지 붙인 질의를 쓴다 —
            # 운영 경로(batch_runner)와 같은 build_retrieval_query()를 공유해서
            # 로컬/코랩/운영 어디로 돌리든 같은 질의가 나가게 한다.
            query_vec = _vdb_query_vec_for(build_retrieval_query(claim))
            vdb_candidates = batch_query_vdb([query_vec], top_k=VDB_TOP_K)[0]
            merged = _merge_vdb_candidates(merged, vdb_candidates)
        except VdbUnavailableError as e:
            if not vdb_unavailable_warned:
                print(f"[VDB] 사용 불가({e}) — VDB 없이 64개 카탈로그 후보만으로 계속 진행합니다.")
                vdb_unavailable_warned = True

        fused = rerank(claim, merged, top_k=5, document_texts=document_texts)
        output_items.append(
            {"item_id": item["item_id"], "candidates": [_candidate_to_dict(c) for c in fused]}
        )

        if i % 10 == 0:
            print(f"리랭킹 진행: {i}/{len(pending['items'])}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"items": output_items}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[완료] {len(output_items)}건을 {RESULTS_PATH}에 저장했습니다.")
    print("다음: python -m agent.pipeline.resume_after_rerank 를 실행하세요.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
