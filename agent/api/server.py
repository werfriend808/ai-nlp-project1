"""
agent/api/server.py — URL 하나를 받아 1~8단계 파이프라인을 통째로 돌리는 실시간 검증 API.

2026-08-21 추가: 프론트엔드에 "새로운 뉴스 기사 검증하기" 버튼(URL 입력 모달)을 위한
백엔드. Qwen3-Embedding-4B(VDB 쿼리)와 bge-reranker-v2-m3(리랭커) 둘 다 GPU가 필요해서
이 서버는 AWS GPU 인스턴스에서만 돌아간다(로컬 7.4GB RAM 컴퓨터에서는 세그폴트 — 오늘
계속 확인한 그 제약과 동일).

파이프라인이 HCX 호출 여러 번 + VDB/리랭커 단계까지 포함해서 기사 하나 처리에 몇 분씩
걸릴 수 있어서(오늘 실측: HCX가 가끔 몇 분씩 응답이 안 오기도 함), 요청-응답을 동기로
묶지 않고 작업(job) 방식으로 간다: POST로 작업을 등록하면 즉시 job_id를 받고, 백그라운드
스레드에서 실제 처리가 진행되며, GET으로 상태를 폴링한다. 데모 규모(동시 사용자 소수)라
Redis/Celery 같은 별도 큐 없이 프로세스 내 dict로 충분하다고 판단 — 서버 재시작하면
진행 중이던 job 상태는 사라지지만(휘발성), 이미 처리 완료된 검증 결과 자체는 run_article()
안에서 DB(verifications.db)에 곧바로 저장되므로 데이터 유실은 없다.

사용법 (AWS GPU 서버, 프로젝트 루트에서):
    source venv/bin/activate
    uvicorn agent.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from db.fetch_article_text import fetch_article_for_verification
from agent.kosis.api_client import KosisApiClient
from agent.kosis.calculator import KosisCalculator
from agent.kosis.query_vdb import batch_query_vdb, lexical_query_vdb, VdbUnavailableError, VDB_TOP_K, LEXICAL_TOP_K
from agent.mapping.embedding_search import build_table_embedding_cache
from agent.pipeline.batch_runner import (
    DEFAULT_CLARIFY_REPLY,
    TABLE_PARAMS_PATH,
    _load_table_catalog_by_id,
    run_article,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent

app = FastAPI(title="KOSIS 뉴스 팩트체킹 실시간 검증 API")
app.add_middleware(
    CORSMiddleware,
    # 데모용 — 프론트가 어느 origin(로컬/배포)에서 오든 우선 허용. 실제 서비스로 넘어가면
    # 프론트 origin만 명시적으로 허용하도록 좁혀야 한다.
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 2026-08-21: 이 서버(AWS)와 프론트엔드 dev 서버(사용자 로컬 PC)는 서로 다른 컴퓨터다.
# _refresh_frontend_exports()가 갱신하는 JSON은 이 서버 자신의 디스크
# (frontend/public/data/)에 쓰이는데, 로컬 프론트는 로컬 자신의 디스크만 읽으므로 실시간
# 검증 결과가 로컬 화면에 자동으로 반영될 방법이 없었다 — 그 디렉터리를 이 API 서버가
# 직접 정적 파일로 서빙해서, 프론트가 검증 완료 후에는 로컬 파일 대신 이 서버에서 최신
# JSON을 바로 받아가게 한다(App.tsx의 loadData가 baseUrl 인자로 이 엔드포인트를 쓴다).
# ---------------------------------------------------------------------------
_FRONTEND_DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
_FRONTEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=_FRONTEND_DATA_DIR), name="data")


# ---------------------------------------------------------------------------
# 2026-08-21: 이 서버를 외부에 노출하려면(보안그룹 8000번 포트 개방) 인증 없이는 누구나
# URL을 넣어 요청을 날릴 수 있고, 요청 하나마다 HCX 호출(비용/쿼터)·KOSIS API 호출까지
# 태워진다 — 악용되면 팀이 공유하는 HCX/KOSIS 쿼터가 소진될 위험이 있다. HCX_API_KEY
# 자체와는 완전히 별개의(재사용 안 함) 새 키(VERIFY_API_KEY)로 보호한다 — 이 키가
# 유출되더라도 HCX 자격증명은 전혀 노출되지 않는다.
# ---------------------------------------------------------------------------
import secrets  # noqa: E402

_VERIFY_API_KEY = os.environ.get("VERIFY_API_KEY")
if not _VERIFY_API_KEY:
    # .env에 안 넣어놨으면 서버 시작할 때마다 새로 만든다 — 서버 재시작하면 키도
    # 바뀌므로(프론트에 다시 알려줘야 함), 운영에서는 .env에 고정값을 넣어두는 걸 권장한다.
    _VERIFY_API_KEY = secrets.token_urlsafe(24)
    print(f"[서버 시작] VERIFY_API_KEY가 .env에 없어서 임시로 생성함: {_VERIFY_API_KEY}")
    print("[서버 시작] .env에 VERIFY_API_KEY=<이 값>을 넣어두면 재시작해도 안 바뀜.")


def _require_api_key(x_api_key: str = Header(default=None)) -> None:
    if x_api_key != _VERIFY_API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key 헤더가 없거나 올바르지 않습니다.")


# ---------------------------------------------------------------------------
# 무거운 리소스(GPU 모델, 카탈로그 캐시)는 서버 시작 시 딱 한 번만 로드한다 — 요청마다
# 새로 로드하면(특히 Qwen3-Embedding-4B, 4B 파라미터) 요청 하나에 수 초~수십 초가 그냥
# 로딩으로 날아간다.
# ---------------------------------------------------------------------------
print("[서버 시작] Qwen3-Embedding-4B 로딩 중...")
from sentence_transformers import SentenceTransformer  # noqa: E402

_VDB_EMBED_MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=1024)
_VDB_QUERY_INSTRUCTION = (
    "Given a Korean news claim sentence, retrieve the KOSIS statistical table "
    "description that best matches it"
)

print("[서버 시작] 64개 카탈로그/임베딩 캐시 로딩 중...")
_CATALOG_BY_ID = _load_table_catalog_by_id()
_EMBEDDING_CACHE = build_table_embedding_cache()
with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
    _TABLE_PARAMS = json.load(f)
_CLIENT = KosisApiClient()
_CALCULATOR = KosisCalculator()
print("[서버 시작] 준비 완료")


# 2026-08-21: claim.search_query(claim_extractor가 같이 생성 — "{지표명} {있는 dimension만}
# {정규화 기관명}" 스타일, VDB 문서 텍스트("{기관} (연월) {표명}")와 형식을 맞춘 짧은 쿼리)가
# 있으면 그걸 쓰고, 없으면(옛 데이터·추출 실패 등) claim.sentence로 안전하게 폴백한다.
# 골든셋 실측: raw 문장 대비 이 스타일 쿼리가 Dense Recall@30 29.3%→61.0%, BM25는
# 2.4%→41.5%로 개선 확인(HyDE 검증 로그 참고) — dense/bm25 둘 다 같은 쿼리를 쓴다.
def _retrieval_query_text(claim) -> str:
    return claim.search_query or claim.sentence


def _vdb_fn(claim):
    """run_article()에 주입할 VDB dense 조회 함수 — search_query(또는 폴백으로 sentence)를
    Qwen으로 임베딩해서 Supabase(kosis_vdb_tables, 28만7천여 개)를 조회한다. VDB 연결
    자체가 안 되면(일시적 장애 등) 조용히 빈 리스트를 반환해서 keyword+64개 카탈로그만으로도
    계속 진행되게 한다."""
    text = f"Instruct: {_VDB_QUERY_INSTRUCTION}\nQuery: {_retrieval_query_text(claim)}"
    vec = _VDB_EMBED_MODEL.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    try:
        return batch_query_vdb([vec], top_k=VDB_TOP_K)[0]
    except VdbUnavailableError as e:
        print(f"[VDB] 조회 실패({e}) — VDB 없이 계속 진행")
        return []


def _bm25_fn(claim):
    """run_article()에 주입할 VDB BM25(trigram) 조회 함수 — dense와 같은 search_query를
    쓴다. VDB 연결이 안 되면 조용히 빈 리스트(dense/keyword만으로 계속 진행).

    2026-08-21 버그 수정: 지금까지 dense용 상수(VDB_TOP_K)를 그대로 재사용하고 있었다 —
    query_vdb.py가 BM25 전용 LEXICAL_TOP_K를 따로 정의해뒀는데도 여기서 안 쓰고 있어서,
    DENSE_TOP_K와 BM25_TOP_K를 환경변수로 따로 조정해도(PHASE 8) BM25 쪽은 실제로는
    dense 값을 따라갔다."""
    try:
        return lexical_query_vdb(_retrieval_query_text(claim), top_k=LEXICAL_TOP_K)
    except VdbUnavailableError as e:
        print(f"[BM25] 조회 실패({e}) — BM25 없이 계속 진행")
        return []


# ---------------------------------------------------------------------------
# 작업(job) 상태 관리. 데모 스케일(동시 요청 소수)이라 인메모리 dict + 락으로 충분하다.
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class VerifyRequest(BaseModel):
    url: str


def _refresh_frontend_exports() -> None:
    """DB에 새로 저장된 검증 결과를 프론트엔드가 읽는 정적 JSON들에 반영한다
    (db/export_json.py의 로직 재사용 + frontend/public/data로 복사, 지금까지 이 프로젝트에서
    수동으로 반복해온 절차를 자동화한 것)."""
    from db.export_json import (
        DEFAULT_ARTICLES_OUT_PATH,
        DEFAULT_DATES_OUT_PATH,
        DEFAULT_ORG_IDS_OUT_PATH,
        DEFAULT_OUT_PATH,
        export_article_dates,
        export_article_texts,
        export_table_org_ids,
        export_to_json,
    )

    export_to_json(DEFAULT_OUT_PATH)
    export_article_texts(DEFAULT_ARTICLES_OUT_PATH)
    export_article_dates(DEFAULT_DATES_OUT_PATH)
    export_table_org_ids(DEFAULT_ORG_IDS_OUT_PATH)

    mapping = {
        DEFAULT_OUT_PATH: "verifications.json",
        DEFAULT_ARTICLES_OUT_PATH: "articles.json",
        DEFAULT_DATES_OUT_PATH: "articleDates.json",
        DEFAULT_ORG_IDS_OUT_PATH: "tableOrgIds.json",
    }
    frontend_data_dir = PROJECT_ROOT / "frontend" / "public" / "data"
    for src, dst_name in mapping.items():
        shutil.copy(src, frontend_data_dir / dst_name)


def _run_job(job_id: str, url: str) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "fetching"

    fetched = fetch_article_for_verification(url)
    if fetched is None:
        with _jobs_lock:
            _jobs[job_id] = {
                "status": "failed",
                "error": "기사 본문을 가져오지 못했습니다(지원 안 되는 사이트이거나 네트워크 오류).",
            }
        return

    article = {
        "label": f"[실시간] {fetched['title'][:40]}",
        "article_title": fetched["title"],
        "article_url": fetched["url"],
        "published_date": fetched["published_date"],
        "article_text": fetched["text"],
        "clarify_reply": DEFAULT_CLARIFY_REPLY,
    }

    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["article_title"] = fetched["title"]

    try:
        results = run_article(
            article,
            _CLIENT,
            _CALCULATOR,
            _TABLE_PARAMS,
            _EMBEDDING_CACHE,
            _CATALOG_BY_ID,
            vdb_fn=_vdb_fn,
            bm25_fn=_bm25_fn,
            raise_on_stage12_error=True,
        )
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id] = {
                "status": "failed",
                "error": f"기사 분류/주장 추출 중 오류가 발생했습니다({type(e).__name__}: {e}). "
                "HCX 응답 지연/오류일 수 있으니 잠시 후 다시 시도해주세요.",
            }
        return

    try:
        _refresh_frontend_exports()
    except Exception as e:
        # export 실패해도 검증 결과 자체는 run_article() 안에서 이미 DB에 저장됐으므로
        # 데이터 유실은 아니다 — 프론트 새로고침만 다음 배치 export 때로 미뤄지는 정도.
        print(f"[export] 프론트 데이터 갱신 실패(검증 결과는 DB에 저장됨): {e}")

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "done",
            "article_title": fetched["title"],
            "claim_count": len(results),
            "results": results,
        }


@app.post("/api/verify", dependencies=[Depends(_require_api_key)])
def start_verify(req: VerifyRequest):
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued"}
    thread = threading.Thread(target=_run_job, args=(job_id, req.url), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/verify/{job_id}", dependencies=[Depends(_require_api_key)])
def get_verify_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id를 찾을 수 없습니다")
    return job


@app.get("/api/health")
def health():
    return {"status": "ok"}
