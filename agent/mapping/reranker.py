"""
agent/mapping/reranker.py — 3단계: 리랭커로 후보 재정렬

팀 계약(interfaces.py) 기준:
입력: Claim 1건 + TableCandidate 리스트 (keyword_search/embedding_search에서 모은 후보들)
출력: TableCandidate 리스트 (재정렬, top-k)

모델: BAAI/bge-reranker-v2-m3 (notebooks/reranker_model_comparison.ipynb 비교 실험 결과
채택. 568M). 분류 헤드가 달린 cross-encoder라 CrossEncoder.predict()가 query-document
쌍의 관련도 점수를 바로 반환한다 — Qwen3-Reranker처럼 "yes"/"no" 토큰 확률을 직접
계산하는 생성형 방식이 아니다.

임베딩(intfloat/multilingual-e5-large, Microsoft)과 달리 이 리랭커는 BAAI(Beijing
Academy of Artificial Intelligence) 제작이다. "임베딩/리랭커 둘 다 같은 회사(Microsoft)
제품으로 통일"하려 했으나, Microsoft가 공식 배포한 다국어 리랭커가 없어 회사 통일은
포기하고 성능/체급 검증이 끝난 bge-reranker-v2-m3를 그대로 유지하기로 함 (2026-07-31 결정).

3단계 전체 흐름:
  keyword_search 결과 + embedding_search 결과 → table_id 기준 합치기(중복 제거)
  → rerank()로 최종 top-k 재정렬

사전 준비물:
    pip install sentence-transformers
    sentence-transformers가 없거나 모델 로딩·추론에 실패하면 rerank_scores()가 None을
    반환해서, rerank()가 기존 score(키워드 매칭 점수 or 임베딩 유사도)를 그대로 정렬
    기준으로 쓰는 항등(identity) 폴백으로 자동으로 넘어간다.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading

# embedding_search.py와 동일한 이유(OpenMP 런타임 중복 로드로 인한 세그폴트, 2026-08-03 확인) —
# 이 모듈만 단독 import될 때도 안전하도록 여기에도 동일하게 설정한다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# embedding_search.py와 동일한 이유(HF Hub CDN 연결이 IPv6에서 무한 대기, 2026-08-05 확인) —
# 이 모듈만 단독 import될 때도 안전하도록 여기에도 동일하게 설정한다.
import socket  # noqa: E402

if not getattr(socket, "_ipv4_only_patched", False):
    _original_getaddrinfo = socket.getaddrinfo

    def _getaddrinfo_ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = _getaddrinfo_ipv4_only
    socket.setdefaulttimeout(30)
    socket._ipv4_only_patched = True

from dataclasses import replace
from pathlib import Path
from typing import Optional

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from agent.observability import Timer, log_event
from agent.kosis.enrich_objl import fetch_period_range
from agent.kosis.detail_cache import get_table_detail, DetailCacheUnavailableError
from agent.kosis.version_meta import get_version_meta_batch

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


RERANKER_MODEL = os.environ.get("KOSIS_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

CATALOG_PATH = Path(__file__).parent / "table_catalog.json"


def load_document_texts(catalog_path: Path = CATALOG_PATH) -> dict[str, str]:
    """table_catalog.json의 embedding_text를 tblId -> 텍스트로 매핑해서 반환한다.

    search_and_rerank()에 document_texts로 넘기면 리랭커가 표 제목이 아니라 이 풍부한
    텍스트(제목+키워드+설명)를 보고 판단한다 — 안 넘기면 table_name(짧은 제목)으로
    대체되어 성능이 크게 떨어진다. 2026-08-03에 이 프로젝트의 모든 실제 호출부
    (csv_batch_runner.py/batch_runner.py/agent_chat.py)가 이걸 안 넘기고 있던 걸
    발견해서 이 헬퍼로 통일했다 — 새로 search_and_rerank()를 호출하는 곳이 생기면
    반드시 이 함수로 document_texts를 만들어서 넘길 것.
    """
    tables = json.loads(catalog_path.read_text(encoding="utf-8"))["tables"]
    return {t["tblId"]: t["embedding_text"] for t in tables}

# 일부 환경(예: 특정 CPU/torch 조합)에서 리랭커 모델 로딩이 세그멘테이션 폴트로 죽는데,
# 이건 OS 레벨 크래시라 아래 try/except로 못 잡는다. 이럴 땐 모델 로딩 자체를 시도하지 않도록
# .env에 KOSIS_DISABLE_RERANKER=1을 넣어서 우회한다 (rerank()가 기존 score 기준 항등
# 정렬로 폴백 — keyword_search 점수 우선, 그다음 embedding_search 유사도).
#
# 2026-08-21(PHASE 8): RERANKER_ENABLED가 명시적으로 설정돼 있으면 그걸 우선한다(원래
# 아키텍처 스펙이 요청한 이름) — 안 넘기면 기존 KOSIS_DISABLE_RERANKER를 그대로 본다
# (하위 호환, 기존에 이 env var로 배포 스크립트/문서를 이미 쓰고 있을 수 있어서 대체가
# 아니라 우선순위만 얹는다).
_RERANKER_ENABLED_ENV = os.environ.get("RERANKER_ENABLED")
if _RERANKER_ENABLED_ENV is not None:
    _DISABLE_RERANKER = _RERANKER_ENABLED_ENV.strip().lower() not in ("1", "true", "yes")
else:
    _DISABLE_RERANKER = os.environ.get("KOSIS_DISABLE_RERANKER", "").strip().lower() in ("1", "true", "yes")

_reranker_singleton = None  # CrossEncoder 인스턴스 lazy 캐시 (프로세스당 1회만 로딩)


def _get_reranker():
    """리랭커(cross-encoder)를 lazy하게 로딩해서 프로세스 전체에서 재사용한다."""
    global _reranker_singleton
    if _reranker_singleton is None:
        # 2026-08-19: sentence-transformers(6.0.0)/transformers(5.x) 조합에서 CrossEncoder.predict()가
        # "AttributeError" (input_ids.ne 호출 시 BatchEncoding에 ne가 없음)로 항상 실패하는 걸
        # AWS GPU 서버에서 실측 확인 — CrossEncoder가 토크나이저 출력(BatchEncoding)을 텐서로
        # 안 풀고 그대로 model.forward()에 넘기는 버그로 보임(sentence-transformers/transformers
        # 여러 버전 조합에서 재현됨, 순정 transformers로는 정상 동작 확인). 그래서 CrossEncoder
        # 래퍼를 안 쓰고 AutoTokenizer/AutoModelForSequenceClassification을 직접 호출한다.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL, trust_remote_code=True)
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda")
            # 일부 모델(trust_remote_code 커스텀 구현)이 config 기본값으로 bfloat16을 쓰는데,
            # Turing급 GPU(T4 등)는 cuBLAS에서 bf16 GEMM을 지원하지 않아
            # "CUBLAS_STATUS_NOT_SUPPORTED"로 죽는다. fp16으로 강제 변환해서 회피한다.
            model = model.half()
        _reranker_singleton = (tokenizer, model)
    return _reranker_singleton


# ---------------------------------------------------------------------------
# 실제 리랭커 모델 연동 지점. bge-reranker-v2-m3(cross-encoder)를 로컬에서 호출한다.
# transformers가 없거나 모델 로딩·추론에 실패하면 None을 반환해서
# rerank()가 후보의 기존 score(키워드 매칭 점수 or 임베딩 유사도)를
# 그대로 정렬 기준으로 쓰는 항등(identity) 폴백으로 넘어가게 한다.
# ---------------------------------------------------------------------------
def rerank_scores(query: str, documents: list[str]) -> Optional[list[float]]:
    if not documents:
        return []
    if _DISABLE_RERANKER:
        return None

    try:
        import torch

        tokenizer, model = _get_reranker()
        device = next(model.parameters()).device
        inputs = tokenizer(
            [query] * len(documents), documents, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.view(-1).float()
        return logits.tolist()
    except Exception as exc:  # noqa: BLE001 - 로딩/추론 실패는 전부 항등 폴백 대상
        print(f"[reranker] {RERANKER_MODEL} 사용 불가({exc!r}) - 항등(identity) 정렬로 폴백합니다.")
        return None


def _tag_source_rank(cand: TableCandidate, key: str, rank: int) -> TableCandidate:
    """cand.source_meta에 "{key}={rank}" 태그를 덧붙인 복사본을 반환한다 (원본은 불변)."""
    tag = f"{key}={rank}"
    meta = f"{cand.source_meta} | {tag}" if cand.source_meta else tag
    return replace(cand, source_meta=meta)


def _merge_candidates(
    keyword_candidates: list[TableCandidate],
    embedding_candidates: list[TableCandidate],
    vdb_candidates: Optional[list[TableCandidate]] = None,
    bm25_candidates: Optional[list[TableCandidate]] = None,
    item_candidates: Optional[list[TableCandidate]] = None,
    core_vdb_candidates: Optional[list[TableCandidate]] = None,
    core_bm25_candidates: Optional[list[TableCandidate]] = None,
) -> list[TableCandidate]:
    """keyword_search/embedding_search/VDB(dense)/BM25(trigram)/항목(item) 후보를 table_id
    기준으로 합치면서, 각 소스 안에서의 순위(1부터, 입력 리스트가 이미 점수 내림차순
    정렬돼 있다고 가정)를 source_meta에 "keyword_rank=N"/"embedding_rank=N"/"vdb_rank=N"
    /"bm25_rank=N"/"item_rank=N"으로 남긴다 — 이후 _rrf_fuse()가 점수 크기가 아니라 이
    순위만으로 최종 신뢰도를 계산한다. 입력으로 받은 candidate 객체는 변형하지 않고
    dataclasses.replace로 복사본만 만든다.

    2026-08-23: item_candidates(agent.kosis.query_vdb.item_query_vdb) 추가 — 표 제목이
    검색어와 전혀 안 겹치지만 표 안의 개별 항목(예: "노동소득")은 겹치는 경우를 보완한다
    (원인 A, "생애주기적자계정" 표 사례로 실측 확인). bm25_rank와 마찬가지로 자동 신뢰
    대상은 아니다 — item_query_vdb 자체가 아직 표본이 작아 노이즈 가능성을 보수적으로
    본다.

    2026-08-25: core_vdb_candidates/core_bm25_candidates(build_core_query() 참고) 추가 —
    age/gender/region/population 수식어를 뺀 "핵심 쿼리"로 별도 검색한 결과를
    "vdb_core_rank"/"bm25_core_rank"로 남긴다. vdb_rank/bm25_rank(전체 search_query 기준)와
    똑같이 is_rrf_trusted()에서 자동 신뢰 대상은 아니다 — 여전히 임베딩/lexical 유사도
    기반이라 노이즈 가능성은 남아있고, RRF 융합에서 "여러 소스가 동의하는 표"로 자연스럽게
    올라오는 걸 노린다.

    2026-08-21: bm25_candidates(agent.kosis.query_vdb.lexical_query_vdb, pg_trgm 기반) 추가.
    keyword_rank와 달리 is_rrf_trusted()에서 자동 신뢰 대상이 아니다 — keyword_search(64개
    수동 카탈로그, 정확한 규칙 매칭)와 달리 trigram 유사도는 raw 문장 기준 실측 Recall이
    거의 0%였고(짧은 search_query를 줘야 의미 있는 신호가 됨, golden set 실측: Recall@30
    2.4%→41.5%), 노이즈 가능성이 여전히 embedding_rank/vdb_rank와 비슷한 수준이라 보수적으로
    취급한다.

    2026-08-18: "keyword=신뢰 / embedding·vdb=unverified" 이분법(그리고 그로 인한
    _promote_verified_within_top_ranks 승격 로직)을 RRF(Reciprocal Rank Fusion)로
    대체한다. 실측(울릉군 기사)에서 VDB 단독으로 찾은 표가 keyword_search가 아예 못 찾은
    진짜 정답이었던 사례가 확인됐는데("고용률(시/군/구)" claim), 기존 이분법은 이런
    경우를 소스 종류만 보고 무조건 "검증 안 됨"으로 버렸다. RRF는 각 소스에서의 순위를
    그대로 합산하므로, 여러 소스에서 상위권으로 뽑힌 표는 자연스럽게 점수가 올라가고
    한 소스에서만 낮은 순위로 잡힌 표는 낮게 남는다 — 소스별 점수 스케일 차이(코사인
    유사도 vs 키워드 매칭 점수)에 영향받지 않는 게 장점이다.
    """
    merged: dict[str, TableCandidate] = {}

    def _add(cands: list[TableCandidate], key: str) -> None:
        for i, cand in enumerate(cands):
            existing = merged.get(cand.table_id)
            if existing is None:
                merged[cand.table_id] = _tag_source_rank(cand, key, i + 1)
            else:
                # 2026-08-23: item_candidates가 다른 채널(keyword/embedding/vdb/bm25)보다
                # 뒤에 _add되면 이 분기(existing이 이미 있음)를 타는데, 여기서 cand의
                # source_meta(예: "kosis_vdb_item(matched='노동소득')")를 안 챙기고
                # existing.source_meta만 이어붙이면 어떤 항목이 매칭됐는지가 그냥 사라진다.
                # rerank()가 이 매칭 항목명을 리랭커 입력 텍스트에 동적으로 붙여야 하므로
                # (아래 _ITEM_MATCH_RE 사용부 참고) cand 쪽에 있으면 같이 이어붙여 보존한다.
                extra_match = _ITEM_MATCH_RE.search(cand.source_meta or "")
                extra = f" | {extra_match.group(0)}" if extra_match else ""
                # 2026-08-24 버그 수정: keyword_search/embedding_search(64개 카탈로그
                # 경로)는 org_id를 아예 안 채운다 — 이 두 채널이 먼저 _add되고 나서
                # VDB(org_id 있음)가 같은 표를 나중에 찾으면, 이 분기(existing 있음)가
                # existing.org_id(None)만 유지하고 cand.org_id(VDB가 채운 값)를 버리고
                # 있었다. _apply_axis_value_signal()이 org_id 없는 후보는 조회를 건너뛰는데,
                # 실측(건설업/제조업 취업자 claim)에서 keyword_search로 먼저 잡힌
                # DT_1DA7E06S_NEW가 이 버그 때문에 axis_value_rank를 못 받는 걸 확인했다.
                # existing에 org_id가 없고 cand엔 있으면 보강한다(반대로 이미 있으면 안 건드림).
                merged_org_id = existing.org_id or cand.org_id
                merged[cand.table_id] = replace(
                    existing, org_id=merged_org_id,
                    source_meta=f"{existing.source_meta} | {key}={i + 1}{extra}"
                )

    _add(keyword_candidates, "keyword_rank")
    _add(embedding_candidates, "embedding_rank")
    _add(vdb_candidates or [], "vdb_rank")
    _add(bm25_candidates or [], "bm25_rank")
    _add(item_candidates or [], "item_rank")
    _add(core_vdb_candidates or [], "vdb_core_rank")
    _add(core_bm25_candidates or [], "bm25_core_rank")
    return list(merged.values())


_RANK_TAG_RE = re.compile(
    r"\b(keyword_rank|embedding_rank|vdb_rank|bm25_rank|item_rank|vdb_core_rank|bm25_core_rank"
    r"|reranker_rank|population_rank|institution_rank|gender_rank|region_rank)=(\d+)"
)

# 2026-08-23: agent.kosis.query_vdb.item_query_vdb가 항목 매칭 후보의 source_meta에
# "kosis_vdb_item(matched='노동소득')" 형태로 남기는 걸 rerank()에서 뽑아 쓴다 — 표
# 텍스트 전체에 항목명을 다 붙이는 실험(DT_1NTA2002, 되돌림)은 리랭커 성능에 도움이
# 안 됐지만, item_rank로 실제로 매칭된 그 항목명 하나만 콕 집어 리랭커 입력에 동적으로
# 붙이는 건 다일루션 없이 정확한 신호만 전달한다는 가설로 시도한다.
_ITEM_MATCH_RE = re.compile(r"kosis_vdb_item\(matched='([^']*)'(?:,\s*score=[\d.]+)?\)")


def _parse_rrf_ranks(source_meta: Optional[str]) -> dict[str, int]:
    """source_meta 문자열에서 "{key}=N" 형태의 순위 태그를 전부 뽑아 dict로 돌려준다."""
    return {key: int(val) for key, val in _RANK_TAG_RE.findall(source_meta or "")}


# 2026-08-21(PHASE 8): 환경변수 RRF_K로 조정 가능하게 함. 기본값은 RRF 원 논문
# (Cormack et al., 2009)의 관례값 60 — 작을수록 상위 순위 후보에 더 민감해진다.
RRF_K = int(os.environ.get("RRF_K", "60"))


# 2026-08-21(PHASE 8): rerank()/search_and_rerank()가 최종적으로 몇 개를 돌려주는지 —
# 리랭커가 재정렬한 결과 개수(RERANKER_TOP_K)와 3단계가 4단계 이후로 넘기는 최종 후보
# 개수(KOSIS_TOP_K)가 지금 구현에서는 하나의 동일한 top_k 파라미터라 두 이름 중
# KOSIS_TOP_K를 대표로 쓴다(RERANKER_TOP_K를 넘기면 그것도 동일하게 인식 — 둘 다 세팅된
# 경우 KOSIS_TOP_K 우선). 기본값 5는 그동안 golden set 검증에 쓴 값 그대로.
_TOP_K_ENV = os.environ.get("KOSIS_TOP_K") or os.environ.get("RERANKER_TOP_K")
DEFAULT_TOP_K = int(_TOP_K_ENV) if _TOP_K_ENV else 5

_RERANK_RAW_RE = re.compile(r"rerank_raw=(-?\d+\.\d+)")

# 리랭커가 1위로 뽑았어도 그 확신도(시그모이드 확률)가 이 값 미만이면 신뢰하지 않는다.
# 2026-08-20 실측: 후보 풀 전체가 무관한 표들뿐이어도(예: 금융사기 적발건수 claim에
# "국제수지"만 후보로 들어온 경우) 리랭커가 그중 "그나마 나은" 하나를 1위로 뽑아버리면
# 순위만 보던 기존 게이트가 그대로 통과시켜서, "건수 vs 백만달러"처럼 단위 자체가 다른
# 비교를 "불일치"라고 확신에 차서 틀리게 답하는 사례가 나왔다(id 18/19/21/25).
# 순위(1등이냐)만이 아니라 점수 크기(정말 관련있다고 볼 만큼 확신했냐)도 같이 봐야 한다.
MIN_RERANKER_CONFIDENCE = 0.5


def is_rrf_trusted(source_meta: Optional[str]) -> bool:
    """이 후보를 최종 판정(judge)까지 진행시켜도 될 만큼 신뢰하는지 판단한다.

    2026-08-18: 기존 "unverified면 무조건 매칭없음" 게이트를 대체한다. keyword_search가
    찾았거나(전통적으로 신뢰해온 신호), 크로스 인코더 리랭커가 전체 후보 풀 중에서
    독자적으로 1위로 평가했으면(reranker_rank=1 — 코사인 유사도가 아니라 실제 문장을
    읽고 내린 판단이라 노이즈에 더 강함) 신뢰한다. 둘 다 아니면(=keyword도 못 찾고
    리랭커도 1위로 보지 않았으면) embedding/VDB 단독 저순위 후보일 가능성이 높아
    신뢰하지 않는다.

    2026-08-20: reranker_rank==1이어도 그 판단의 확신도(rerank_raw를 시그모이드한 값)가
    MIN_RERANKER_CONFIDENCE 미만이면 신뢰하지 않는다 — "후보 풀 전부가 나빴는데 그중
    제일 덜 나쁜 걸 1등 시켰을 뿐"인 경우를 걸러내기 위함(위 실측 사례 참고).

    2026-08-21: population_rank/institution_rank/gender_rank/region_rank(아래
    _apply_*_signal 함수들 참고)도 keyword_rank와 동급으로 신뢰한다 — claim에서 직접
    뽑은(또는 화이트리스트로 정규화된) 명시적 규칙 기반 신호라 임베딩/리랭커의 노이즈
    있는 유사도 판단보다 신뢰도가 높다고 보기 때문.

    2026-08-23: item_rank는 위 규칙 신호들과 다르다 — "청년"/"여성" 같은 정확한 단어가
    표에 그대로 있는지를 보는 게 아니라, 임베딩 코사인 유사도(item_query_vdb)로 판단하는
    "감으로 말하는" 신호라 bm25_rank/vdb_rank와 같은 부류다(그래서 population_rank
    화이트리스트에 그냥 끼워 넣지 않는다). 처음엔 "item_rank=1 + 다른 채널 corroboration"
    조건으로 신뢰해봤는데, 골든셋 실측(DT_1NTA2002 11건)에서 이 조건이 구조적으로 거의
    항상 거짓이었다 — item 채널은 애초에 "다른 채널이 다 놓친 걸 혼자 찾아라"고 만든
    거라, 정의상 다른 채널이 같이 찾아주는 일이 거의 없다(11건 중 corroboration 있었던
    건 0건). 대신 item_trust_score()의 유사도 점수 문턱으로 직접 신뢰한다 — 골든셋
    70건 실측: 정답 매칭 점수(0.3676~0.4810, 2건 예외 0.3277) vs 무관한 claim의 오탐
    점수(0.30~0.3584)가 깨끗하게 갈려서, 순위가 아니라 점수 자체가 신뢰할 만한
    구분자임을 확인했다.
    """
    ranks = _parse_rrf_ranks(source_meta)
    if "keyword_rank" in ranks:
        return True
    if any(k in ranks for k in ("population_rank", "institution_rank", "gender_rank", "region_rank")):
        return True
    if _item_trusted(source_meta):
        return True
    if ranks.get("reranker_rank") != 1:
        return False
    raw_match = _RERANK_RAW_RE.search(source_meta or "")
    if not raw_match:
        return False
    return _sigmoid(float(raw_match.group(1))) >= MIN_RERANKER_CONFIDENCE


_ITEM_SCORE_RE = re.compile(r"kosis_vdb_item\(matched='[^']*',\s*score=([\d.]+)\)")

# 2026-08-24 실측(70건 골든셋, 단일 표 인덱싱 기준): item 채널이 찾은 후보의 코사인
# 유사도 점수가 노이즈(무관한 claim이 우연히 매칭된 경우, 28건 실측 범위 0.30~0.3584)와
# 진짜 신호(정답 11건 중 9건, 0.3676~0.4810) 사이에 자연스러운 틈이 있다. item_rank
# "순위" 자체는 지금처럼 표 1개만 인덱싱된 상태에선 있으면 사실상 항상 1등이라 구분력이
# 없다 — 반드시 점수로 문턱을 걸어야 한다.
ITEM_TRUST_MIN_SIMILARITY = 0.36


def _item_trusted(source_meta: Optional[str]) -> bool:
    """item 채널이 찾은 후보를 점수 문턱으로 신뢰할지 판단한다(위 ITEM_TRUST_MIN_SIMILARITY
    실측 근거 참고)."""
    if not source_meta:
        return False
    m = _ITEM_SCORE_RE.search(source_meta)
    return bool(m) and float(m.group(1)) >= ITEM_TRUST_MIN_SIMILARITY


def select_trusted_candidate(candidates: list) -> Optional[TableCandidate]:
    """RRF 융합 순위 그대로, 최종적으로 채택할 후보 하나를 고른다.

    2026-08-24: candidates[0](RRF 종합 1위)만 검사하던 기존 방식 대신, "믿을 만한 첫
    후보"를 찾도록 넓혔다. 처음엔 is_rrf_trusted() 전체(keyword_rank/population_rank/
    reranker_rank==1 등)로 후보 전체를 훑어봤는데, 골든셋 70건 재검증에서 오히려
    회귀가 커졌다(틀렸는데 신뢰되는 케이스 25건→35건) — reranker_rank==1이 RRF 종합
    순위와 안 겹쳐도 무조건 신뢰하게 되면서, 원래 RRF가 주던 이중 확인(교차 인코더
    +종합 순위 둘 다 동의해야 신뢰)이 깨졌기 때문이다. 그래서 범위를 item 채널로만
    좁힌다 — item 유사도 점수는 노이즈/신호가 깨끗하게 갈린다는 게 이미 검증됐으므로
    (_item_trusted 참고) 이 조건 하나만 candidates[0] 밖에서도 예외적으로 허용한다.
    아무 조건도 못 넘으면 기존과 동일하게 candidates[0]을 반환해(안전한 폴백)
    is_rrf_trusted() 재검사가 "판단불가" 경로로 그대로 이어지게 한다."""
    if not candidates:
        return None
    top1 = candidates[0]
    if is_rrf_trusted(top1.source_meta):
        return top1
    for c in candidates[1:]:
        if _item_trusted(c.source_meta):
            return c
    return top1


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _rrf_fuse(reranker_ranked: list[TableCandidate], *, k: int = RRF_K) -> list[TableCandidate]:
    """리랭커 점수로 이미 정렬된 후보 리스트에 그 순위(reranker_rank)까지 얹어서
    RRF_score(D) = sum(1 / (k + rank_L(D)))를 계산하고(D가 등장하는 소스 리스트 L마다
    합산) 그 점수로 다시 정렬한다. 후보의 리랭커 시그모이드 점수는 버리고 RRF 점수로
    덮어쓴다 — 최종 정렬 기준을 하나로 통일하기 위함.

    2026-08-14/16에 있었던 "verified 후보 점수 보너스(+0.05)"와 "순위 기반 top-5 승격"
    시행착오(하단 git 이력 참고)를 RRF 하나로 대체한다 — 특정 소스 하나를 특별 취급하는
    대신 keyword/embedding/vdb/reranker 네 순위를 대칭적으로 합산하므로, 임베딩·VDB
    단독으로만 찾았어도 리랭커 순위까지 좋으면 자연스럽게 상위로 올라온다.
    """
    fused: list[TableCandidate] = []
    for i, cand in enumerate(reranker_ranked):
        ranks = _parse_rrf_ranks(cand.source_meta)
        ranks["reranker_rank"] = i + 1
        rrf_score = sum(1.0 / (k + r) for r in ranks.values())
        meta = f"{cand.source_meta} | reranker_rank={i + 1} rrf_score={rrf_score:.4f}"
        fused.append(replace(cand, score=rrf_score, source_meta=meta))
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


def _document_text_for(cand: TableCandidate, document_texts: Optional[dict[str, str]]) -> str:
    """리랭커에 넘길 문서 텍스트를 만든다. document_texts(표 전체 텍스트)가 기본이고,
    cand가 item_rank 채널로 매칭됐으면(source_meta에 "kosis_vdb_item(matched='...')"
    존재) 그 매칭된 항목명 하나만 " · 일치 항목: {item_name}"으로 덧붙인다.

    2026-08-23: DB에 저장된 표 텍스트 자체에 항목명을 다 붙이는 건(모든 후보, 모든 검색
    채널에 항상 영향) 다일루션으로 검증 실패했다(DT_1NTA2002 실험, 되돌림). 이건 그 대신
    item_rank로 실제 매칭된 후보에 한해서만, 그 요청(claim) 처리 동안만 메모리 상에서
    텍스트를 늘린다 — DB에는 저장되지 않고, 다른 채널로만 찾은 후보에는 전혀 영향 없다."""
    base = (document_texts or {}).get(cand.table_id, cand.table_name)
    m = _ITEM_MATCH_RE.search(cand.source_meta or "")
    return f"{base} · 일치 항목: {m.group(1)}" if m else base


def rerank(
    claim: Claim,
    candidates: list[TableCandidate],
    *,
    top_k: int = DEFAULT_TOP_K,
    document_texts: Optional[dict[str, str]] = None,
) -> list[TableCandidate]:
    """후보 TableCandidate 리스트를 리랭커로 재정렬한 뒤 RRF로 최종 융합한다.

    document_texts: table_id -> 임베딩/설명 텍스트. 넘기지 않으면 table_name으로 대체.
    """
    if not candidates:
        return []

    documents = [_document_text_for(c, document_texts) for c in candidates]
    scores = rerank_scores(claim.sentence, documents)

    if scores is None:
        # 리랭커 모델을 못 쓰는 상황(의존성 미설치 등) — 항등 폴백.
        # keyword_search도 못 찾고 리랭커 판단도 없는 후보(embedding/VDB 단독 저순위)는
        # score 크기만으로 정렬하면 노이즈가 앞설 수 있어 뒤로 민다.
        def _sort_key(c: TableCandidate) -> tuple[bool, float]:
            return (not is_rrf_trusted(c.source_meta), -c.score)

        return sorted(candidates, key=_sort_key)[:top_k]

    reranked: list[TableCandidate] = []
    for cand, score in zip(candidates, scores):
        # 2026-08-24 버그 수정(experiment/pipeline-redesign): 여기서 TableCandidate를
        # 필드 나열로 새로 만들면서 org_id를 안 넣고 있었다 — dataclasses.replace()가
        # 아니라 생성자를 직접 호출해서, 명시 안 한 필드는 전부 기본값(None)으로 리셋됨.
        # 이 함수(rerank())는 모든 후보가 예외 없이 거쳐가는 단계라, VDB에서 org_id가
        # 정상적으로 채워져 온 후보(DB에는 287,498건 전부 org_id 있음, 실측 확인)조차
        # 여기를 지나가면 org_id가 사라졌다 — _build_dynamic_kosis_slots()(5단계, VDB
        # 전용 표의 동적 슬롯매핑에 org_id 필수)와 _apply_period_coverage_filter()(org_id
        # 없으면 그냥 통과시키는 조용한 폴백)가 이 버그 때문에 계속 무력화되고 있었다.
        reranked.append(
            TableCandidate(
                table_id=cand.table_id,
                table_name=cand.table_name,
                score=_sigmoid(score),
                required_slots=cand.required_slots,
                source_meta=f"{cand.source_meta} | rerank_raw={score:.3f}",
                org_id=cand.org_id,
            )
        )

    reranked.sort(key=lambda c: c.score, reverse=True)
    fused = _rrf_fuse(reranked)
    return fused[:top_k]


# 표 카탈로그에 실제로 등장하는 인구집단 용어만 담았다(2026-08-21, table_catalog.json
# 전수 스캔 결과: 청년 2건, 고령 3건, 유아 1건, 어린이 1건, 외국인 2건 — 나머지 후보군
# 예: 청소년/노인/여성/남성/장애인 등)은 이 카탈로그엔 아예 없어서 넣어봐야 항상 매칭
# 실패라 의미가 없다. 새 표가 추가되면 이 리스트도 같이 늘려야 한다.
_POPULATION_TERMS = ["청년", "고령", "유아", "어린이", "외국인"]


def _apply_population_signal(
    claim: Claim, candidates: list[TableCandidate], document_texts: Optional[dict[str, str]]
) -> list[TableCandidate]:
    """claim 문장(+ population 필드)에 인구집단 용어가 있으면, 그 용어를 표 설명에
    명시적으로 포함하는 후보에 population_rank=1을 붙인다(keyword_rank와 동급으로
    is_rrf_trusted가 신뢰함).

    2026-08-21: "청년층 고용률"/"청년 취업자" 같은 claim이 청년 전용 표(DT_1DA7102S)
    대신 성별 경제활동인구총괄(DT_1DA7001S, 둘 다 고용률·실업률을 담고 있어 임베딩·
    리랭커 모두 구분 못 함)로 잘못 매칭되는 걸 golden set으로 실측 확인(A040-02/03,
    reranker on/off 둘 다 오답 — notebooks 골든셋 비교 로그 참고). keyword_search가
    이미 "청년"을 SYNONYMS로 다루고 있을 수도 있지만, 후보 풀에 두 "형제 표"가 같이
    들어왔을 때 어느 쪽이 이기는지까지는 못 가려서 이 단계가 별도로 필요하다.

    새 후보를 발굴하지 않는다(발견은 keyword/embedding/vdb 몫) — 이미 merge된 후보
    중에서 이 명시적 신호로 형제 표만 가려낸다."""
    text_source = f"{claim.population or ''} {claim.sentence}"
    matched_terms = [t for t in _POPULATION_TERMS if t in text_source]
    if not matched_terms:
        return candidates

    tagged = []
    for c in candidates:
        doc = (document_texts or {}).get(c.table_id, c.table_name)
        if any(t in doc for t in matched_terms):
            tagged.append(_tag_source_rank(c, "population_rank", 1))
        else:
            tagged.append(c)
    return tagged


# 2026-08-21(PHASE 5): institution/gender/region도 population과 같은 원리로 소프트
# 부스팅한다 — claim에 값이 있을 때만, 하드 필터 없이. claim.source_org는 옛 명칭("통계청")
# 이나 포털명("KOSIS")으로 뽑히는 경우가 실측 확인돼서(HyDE 재테스트 로그 참고), 표
# 텍스트("국가데이터처 ...")와 그대로 비교하면 매칭 실패한다 — 화이트리스트로 정규화한
# 뒤에만 신호로 쓴다(정규화 실패하면 신호 자체를 생략, "모르는 기관"을 억지로 우기지 않음).
#
# 2026-08-21/22: "국가데이터처"를 화이트리스트에서 뺐다 — 골든셋 41건 재평가에서 Recall@1이
# 65.9%→51.2%로 떨어지는 회귀가 실측 확인됐다(institution_rank가 켜진 12건 중 10건이 오히려
# 오답을 1등으로 밀어올림). 원인: population 용어(청년/고령 등)와 달리 "국가데이터처"는
# KOSIS 자체를 운영하는 기관이라 후보 표 설명에 너무 흔하게 등장한다 — 특히 VDB
# text(기관명+연도명+표명 포맷)는 발행기관을 항상 앞에 붙이므로, source_org가 "통계청"으로
# 뽑히는(가장 흔한 케이스, 이번 골든셋 12/13건) 순간 상관없는 후보들까지 대거
# institution_rank=1(keyword_rank와 동급 신뢰)을 받아버렸다 — 이미 되돌린 _apply_granularity_signal
# (월별/연도별)과 정확히 같은 실패 패턴(도메인 특이성 없는 용어로 부스팅). 나머지 기관명
# (한국은행/국세청/관세청 등)은 후보 풀에서 훨씬 드물게 등장해 이 문제가 없을 가능성이
# 높지만, 이번 골든셋엔 그 기관들이 활성화된 사례가 거의 없어 개별 검증은 못 했다 —
# 재검증 없이는 이것도 안전하다고 확정하지 않는다.
_ORG_ALIASES = {"통계청": "국가데이터처", "국가통계포털": "국가데이터처"}
_KNOWN_INSTITUTIONS = [
    "한국은행", "국세청", "기획예산처", "국민연금공단",
    "과학기술정보통신부", "한국무역협회", "한국부동산원", "한국거래소",
    "산업통상부", "고용노동부", "보건복지부", "관세청", "산림청",
]

# 2026-08-22: "한국은행 기준금리" 주장이 "예금은행 대출금리"(전혀 다른 통계, 시중은행이
# 정하는 금리)로 오매칭돼 확정 불일치까지 나버린 실제 사례 발견 — 원인은 진짜 정답표
# (VDB의 DT_2OEEO032, OECD 소스 "중앙은행 기준금리", 대한민국 분류값 포함해 실제 조회
# 확인됨)가 표 제목에 "한국은행"이 아니라 국제비교표 특유의 일반 명칭 "중앙은행"을 써서,
# 검색 쿼리("한국은행 ...")로는 후보 풀에 아예 들어오지도 못했기 때문이다("중앙은행
# 기준금리"로 직접 검색하면 1위로 잡히는 것까지 실측 확인). docs/PENDING_TABLES.md에
# "기준금리는 KOSIS Open API로 검증 불가"라고 기록돼 있었는데, 이건 이 국제비교표를 미처
# 못 찾고 내린 결론이라 재검토가 필요하다. "국가데이터처"(위 주석)처럼 기관명 자체가 아니라
# "그 기관이 흔히 다른 이름으로도 불리는 특정 케이스"만 좁게 다루므로 같은 회귀 위험이
# 작다 — 필요할 때마다 이런 특정 기관 쌍만 추가한다(일반 용어는 여기 넣지 않는다).
_INSTITUTION_ALIASES: dict[str, list[str]] = {
    "한국은행": ["중앙은행"],
}


def _normalize_institution(source_org: Optional[str]) -> Optional[str]:
    if not source_org:
        return None
    stripped = source_org.strip()
    normalized = _ORG_ALIASES.get(stripped, stripped)
    return normalized if normalized in _KNOWN_INSTITUTIONS else None


def _apply_institution_signal(
    claim: Claim, candidates: list[TableCandidate], document_texts: Optional[dict[str, str]]
) -> list[TableCandidate]:
    """claim.source_org가 화이트리스트 기관으로 정규화되면(예: "통계청"→"국가데이터처"),
    그 정규화된 기관명을 표 설명에 포함하는 후보에 institution_rank=1을 붙인다.
    정규화 안 되면(목록에 없는/모르는 기관) 신호 자체를 생략 — 하드 필터 아님, 다른
    후보를 배제하지 않는다."""
    institution = _normalize_institution(claim.source_org)
    if institution is None:
        return candidates

    doc_terms = [institution, *_INSTITUTION_ALIASES.get(institution, [])]
    tagged = []
    for c in candidates:
        doc = (document_texts or {}).get(c.table_id, c.table_name)
        if any(term in doc for term in doc_terms):
            tagged.append(_tag_source_rank(c, "institution_rank", 1))
        else:
            tagged.append(c)
    return tagged


def expand_institution_query_aliases(query: str, source_org: Optional[str]) -> str:
    """VDB dense/BM25에 보낼 검색 쿼리에 기관명 별칭을 덧붙인다(대체 아님, 추가만) —
    claim.source_org가 "한국은행"인데 정답표는 국제비교표 특유의 일반 명칭("중앙은행")을
    쓰는 경우처럼, 검색어에 없는 표현 때문에 후보 풀에 아예 못 들어오는 걸 막는다.
    원 쿼리는 그대로 두므로 기존에 "한국은행"으로 잘 맞던 다른 표(예금은행 금리 등)의
    매칭에는 영향이 없다."""
    if not source_org:
        return query
    aliases = _INSTITUTION_ALIASES.get(source_org.strip())
    if not aliases:
        return query
    extra = " ".join(a for a in aliases if a not in query)
    return f"{query} {extra}".strip() if extra else query


# 2026-08-25: search_query 구성 규칙(claim_extractor_prompt.txt) 자체가
# "[지표명] + [있는 dimension만 이어붙임] + [기관명]"이라, 흔하고 강한 인구통계 용어
# (예: "65세 이상")가 낀 claim은 그 문자열 전체를 하나의 임베딩으로 인코딩할 때 수식어가
# 핵심 지표 신호를 밀어내는 문제가 팀원 실측으로 확인됐다(예: "소비 총액 65세 이상"으로
# 검색하면 top-10이 전부 "65세 이상 고령자 OO" 인구통계 표로 채워지고, 정답표("국민이전계정"
# 계열)는 후보 풀에 아예 못 들어옴). _apply_population_signal 등 위 신호들은 이미 merge된
# 후보 중에서 재정렬만 하므로(rerank 단계) 이런 recall 실패(retrieval 단계)는 못 고친다 —
# 그래서 수식어를 뺀 "핵심 쿼리"로 별도 검색을 하나 더 돌려서(search_and_rerank의
# vdb_core_rank/bm25_core_rank) RRF 융합 풀에 합류시킨다.
def build_core_query(claim: Claim) -> Optional[str]:
    """claim.search_query에서 age/gender/region/population 수식어를 뺀, 핵심 지표
    (statistic_expression) + 정규화된 기관명만 있는 검색어를 만든다.

    dimension이 하나도 없는 claim은 core query가 기존 search_query와 사실상 같은 문자열이라
    (지표명+기관명만 있으므로) 중복 검색을 피하려고 None을 반환한다 — 호출부에서 그 경우
    core 검색 자체를 스킵한다. statistic_expression이 없으면(핵심 지표를 못 뽑은 claim)
    core query를 만들 근거가 없으므로 마찬가지로 None.

    LLM 프롬프트에 새 필드를 추가하는 대신 이미 추출된 statistic_expression/source_org로
    파이썬에서 계산한다 — 스키마 변경도, 추가 HCX 호출도 필요 없다. 다만 search_query와
    달리 statistic_expression은 KOSIS식 정규화가 안 돼 있을 수 있어(step 1 다듬기는
    search_query 생성 규칙에만 있음) 완벽히 같은 품질의 쿼리는 아니다 — 그래도 수식어에
    안 밀리는 게 더 중요한 상황(핵심 지표 없이 top-10이 통째로 엉뚱한 표로 채워지는 경우)
    에서만 보조로 쓰인다."""
    if not any([claim.age, claim.gender, claim.region, claim.population]):
        return None
    if not claim.statistic_expression:
        return None

    institution = _normalize_institution(claim.source_org)
    parts = [claim.statistic_expression, institution]
    return " ".join(p for p in parts if p)


_GENDER_TERMS = ["여성", "남성"]


def _apply_gender_signal(
    claim: Claim, candidates: list[TableCandidate], document_texts: Optional[dict[str, str]]
) -> list[TableCandidate]:
    """claim.gender가 "여성"/"남성"으로 명시돼 있으면(claim_extractor가 optional dimension으로
    뽑음 — 없으면 None), 그 단어를 표 설명에 포함하는 후보에 gender_rank=1을 붙인다."""
    if claim.gender not in _GENDER_TERMS:
        return candidates

    tagged = []
    for c in candidates:
        doc = (document_texts or {}).get(c.table_id, c.table_name)
        if claim.gender in doc:
            tagged.append(_tag_source_rank(c, "gender_rank", 1))
        else:
            tagged.append(c)
    return tagged


def _apply_region_signal(
    claim: Claim, candidates: list[TableCandidate], document_texts: Optional[dict[str, str]]
) -> list[TableCandidate]:
    """claim.region이 있으면(claim_extractor 규칙상 실제로 지리적 범위를 한정하는 표현이
    있을 때만 채워짐 — "전국"류 일반 표현은 애초에 null로 옴) 그 지역명을 표 설명에 포함하는
    후보에 region_rank=1을 붙인다."""
    if not claim.region:
        return candidates

    tagged = []
    for c in candidates:
        doc = (document_texts or {}).get(c.table_id, c.table_name)
        if claim.region in doc:
            tagged.append(_tag_source_rank(c, "region_rank", 1))
        else:
            tagged.append(c)
    return tagged


# 2026-08-21 시도했다가 되돌린 것: claim.period(예: "지난달"/"작년")로 월/분기/연 주기를
# 판정해서 표 제목에 "월별"/"연도별" 등이 있는 후보를 population_rank와 같은 방식으로
# 부스팅해봤다(DT_1J22041 "연도별 소비자물가 등락률" vs DT_1J22042 "월별 소비자물가
# 등락률"처럼 같은 통계가 주기별로 다른 tblId로 나뉘는 실제 사례가 있어서). 근데 골든셋
# 41건 재평가 결과 Recall@1이 65.9%→56.1%로 오히려 떨어져서 되돌렸다 — "월별"/"연도별"
# 같은 주기 단어는 "청년"/"고령"과 달리 도메인 특이성이 없어서 완전히 무관한 표에도
# 흔히 등장한다(예: "지난달 외식 물가" claim이 CPI와 무관한 "월별 소매판매액지수"로
# 튐, "월.분기.연간 인구동향"이 정답인 claim이 "월별 출생아 수"라는 다른 세부표로 밀림).
# 재시도한다면 주기 단어 하나만으로 부스팅하지 말고, 최소한 population_rank처럼 이미
# 주제가 맞는 후보들 사이에서만 주기로 타이브레이크하는 방식으로 범위를 좁혀야 한다.


# 2026-08-22: run_article()의 주장(claim) 처리를 스레드풀로 병렬화하면서 추가 — vdb_fn(GPU
# 임베딩 모델), embedding_fn(같은 GPU 모델 또는 그 캐시), rerank()(GPU 크로스인코더)가 전부
# 이 함수 안에서만 호출되므로, 이 함수 전체를 락으로 감싸는 게 여러 곳에 락을 흩뿌리는 것보다
# 안전하고 단순하다. 3단계(표매칭) 자체는 파이프라인에서 빠른 축이라 여기를 직렬화해도
# 손해가 적고, 진짜 느린 4~8단계(LLM 호출)는 이 락 밖에서 병렬로 풀린다.
_SEARCH_LOCK = threading.Lock()


# 2026-08-22: 위(540줄) "월별/연도별 부스팅 시도했다가 되돌림" 사례와 정확히 같은 위험이
# 있는 종류의 신호라서(구조적 표 제목 패턴 기반, 도메인 특이성 없음) 그 교훈을 그대로
# 따른다 — 점수를 직접 바꾸지 않고, 이미 리랭커가 사실상 동점으로 본 후보들 사이에서만
# "분류축이 더 단순한 쪽"으로 타이브레이크한다(예: "성/연령별 취업자" 2축 vs "산업/연령/
# 교육정도/종사지위별 취업자" 4축 — claim이 세부분류를 요구 안 하면 단순 표가 대개 정답에
# 더 가깝다, 취업자 근접중복 표 홍수 실측 사례). 반드시 골든셋으로 즉시 재검증해서 회귀
# 여부를 확인한 뒤에만 유지한다.
_COMPLEXITY_TIE_EPSILON = 0.0015  # 이 이내로 점수 차이 나는 후보끼리만 재정렬 대상
_FINE_GRAINED_MARKERS = ("소분류", "세분류")


def _table_complexity(table_name: str) -> int:
    """분류축 개수를 "/" 개수로 근사(예: "산업/연령/교육정도/종사지위별" -> 3). "소분류"/
    "세분류" 표기가 있으면 세분화된 축이라는 뜻이라 추가 패널티를 준다."""
    score = table_name.count("/")
    if any(marker in table_name for marker in _FINE_GRAINED_MARKERS):
        score += 1
    return score


def _apply_simplicity_tiebreak(candidates: list[TableCandidate]) -> list[TableCandidate]:
    if len(candidates) < 2:
        return candidates
    result = list(candidates)
    i = 0
    while i < len(result):
        j = i + 1
        while j < len(result) and result[i].score - result[j].score <= _COMPLEXITY_TIE_EPSILON:
            j += 1
        if j - i > 1:
            # 이 구간(i~j-1)은 리랭커 점수가 사실상 동점 — 원래 순서(상대 순위)는 유지한 채
            # 복잡도만 오름차순으로 안정 정렬(단순한 표가 앞으로).
            result[i:j] = sorted(result[i:j], key=lambda c: _table_complexity(c.table_name))
        i = j
    return result


# 2026-08-24: AXIS(분류축) "이름"이 아니라 그 축 "안의 실제 값"을 후보 표의 실제
# KOSIS 코드맵과 정확한 문자열로 비교한다 — "취업자" 3형제 표 충돌 사례로 실측 검증
# 완료(notebooks 없이 직접 조회): claim에 "건설업"/"제조업"처럼 축 값이 그대로 나오는데,
# 이걸 AXIS "이름"("산업별") 임베딩과 비교하면 의미적으로 안 이어져서 실패한다
# (Recall 1/3). 대신 그 값이 후보 표의 code_map 라벨에 그대로 포함되는지 정확한
# 문자열로 확인하면 성공한다(연령 축만 있는 표는 확실히 배제, 산업 축 있는 표만 남음).
# AXIS 이름이 아니라 AXIS 값을 봐야 한다는 팀 제안서의 "AXIS VALUE는 FTS/Metadata"
# 원칙과 정확히 일치.
#
# 비용 주의: get_table_detail()은 캐시 미스 시 실제 KOSIS API 호출이라, 후보 전체가
# 아니라 상위 _AXIS_VALUE_TOP_N개까지만 확인한다(population_rank 등 다른 시그널과
# 달리 document_texts만으로는 안 되고 반드시 API/캐시 조회가 필요하기 때문).
_AXIS_VALUE_TOP_N = 20
_AXIS_LABEL_PREFIX_RE = re.compile(r"^[A-Za-z*]\s+")
_AXIS_LABEL_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _strip_axis_label_noise(label: str) -> str:
    """KOSIS 분류축 라벨("F 건설업(41~42)")의 기호/코드범위를 벗겨서 순수 이름만
    남긴다(batch_runner.py의 _strip_kosis_label_noise와 동일한 패턴 — reranker.py가
    batch_runner.py를 import하면 순환참조가 생겨서 여기 별도로 둔다)."""
    stripped = _AXIS_LABEL_SUFFIX_RE.sub("", label)
    stripped = _AXIS_LABEL_PREFIX_RE.sub("", stripped)
    return stripped or label


# 2026-08-24: 골든셋 회귀 검증용 토글 — 끄면(0) 기존 동작과 완전히 동일(신호 자체가
# 안 생기니 _apply_period_coverage_filter의 axis 우선 분기도 자동으로 안 탐, 하위
# 호환 폴백 그대로). 기본값은 켜짐(1) — 검증 스크립트에서 WITH_AXIS_VALUE=0으로
# 베이스라인을 재현한다.
_AXIS_VALUE_ENABLED = os.environ.get("WITH_AXIS_VALUE", "1").strip().lower() not in ("0", "false", "no")


def _apply_axis_value_signal(claim: Claim, candidates: list[TableCandidate]) -> list[TableCandidate]:
    """claim.population/statistic_expression의 구체적 값이 후보 표의 실제 분류축 값
    목록(KOSIS code_map)에 정확히 포함되는지 확인해서, 있으면 axis_value_rank=1을
    붙인다. org_id가 없거나 표 상세 조회가 실패하면(캐시 미스+API 오류 등) 그냥
    건너뛴다(fail-open — 다른 후보를 배제하지 않음, population_rank류와 동일 원칙)."""
    if not _AXIS_VALUE_ENABLED:
        return candidates
    terms = [t.strip() for t in (claim.population, claim.statistic_expression) if t and t.strip()]
    if not terms or not candidates:
        return candidates

    tagged = []
    for i, c in enumerate(candidates):
        if i >= _AXIS_VALUE_TOP_N or not c.org_id:
            tagged.append(c)
            continue
        try:
            detail = get_table_detail(c.table_id, c.org_id)
        except DetailCacheUnavailableError:
            tagged.append(c)
            continue
        if detail.get("status") != "ok":
            tagged.append(c)
            continue
        code_maps = detail.get("code_maps") or {}
        matched = any(
            term in _strip_axis_label_noise(label) or _strip_axis_label_noise(label) in term
            for term in terms
            for cm in code_maps.values()
            for label in cm.keys()
        )
        tagged.append(_tag_source_rank(c, "axis_value_rank", 1) if matched else c)
    return tagged


_PERIOD_YEAR_RE = re.compile(r"(19|20)\d{2}")
_PERIOD_COVERAGE_TOP_N = 5  # 상위 몇 개까지만 확인할지 — 매 claim마다 API 호출이 늘어나므로 좁게 잡음


def _extract_target_year(claim: Claim) -> Optional[int]:
    if not claim.period:
        return None
    m = _PERIOD_YEAR_RE.search(claim.period)
    return int(m.group(0)) if m else None


def _covers_year(period_range: dict, year: int) -> Optional[bool]:
    """period_range(fetch_period_range() 반환값)의 주기 중 하나라도 year를 포함하면
    True, 파싱 가능한 주기가 전부 벗어나면 False, 파싱 자체가 안 되면 None(판단 보류)."""
    checked = False
    for start, end in period_range.values():
        start_m, end_m = _PERIOD_YEAR_RE.search(start), _PERIOD_YEAR_RE.search(end)
        if not start_m or not end_m:
            continue
        checked = True
        if int(start_m.group(0)) <= year <= int(end_m.group(0)):
            return True
    return False if checked else None


# 2026-08-22: 같은 통계를 구판/신판 tblId로 KOSIS가 갈아치운 경우(예: DT_1DA7E06S가
# 2024년까지만 수록하고 끝난 반면 DT_1DA7E06S_NEW가 2013년부터 이어받아 지금까지
# 수록 — 실측 확인), 리랭커 점수만으로는 구판이 이길 때가 있었다. claim이 필요로 하는
# 연도가 후보 표의 실제 수록기간을 벗어나면(확인 가능한 경우에 한해) 순위 밖으로 민다 —
# "최신판을 무조건 우선"이 아니라 "그 표에 실제로 그 연도 데이터가 있는가"만 본다.
# 기사가 오래된 시점을 다뤄서 오히려 구판에만 있는 데이터가 필요한 경우까지 안전하게
# 처리된다. 상위 N개만 확인(비용 제한), 조회 실패/판단 불가 시엔 그대로 순위 유지(fail-open).
#
# 2026-08-24 수정: 기존엔 "RRF 종합 순위 상위 5개"만 확인했는데, 실측으로 확인된
# "취업자" 구판/신판 클러스터(30개 후보 중 11개가 산업/성별 취업자 계열 연도별 중복)에서
# 이 상위 5개 안에 신판이 안 들어있으면 필터가 아예 발동을 안 하는 구멍이 있었다.
# _apply_axis_value_signal()이 이미 axis_value_rank로 "관련 있는 후보"를 찾아뒀으면,
# 임의의 "상위 5개" 대신 그 후보 집합을 확인 대상으로 쓴다 — AXIS로 좁혀진 후보 안에서만
# 최신판을 고르는 것. axis_value_rank가 하나도 없으면(그 신호가 아예 안 켜진 claim)
# 기존과 동일하게 상위 5개로 폴백한다(하위 호환, 회귀 없음).
def _apply_period_coverage_filter(claim: Claim, candidates: list[TableCandidate]) -> list[TableCandidate]:
    year = _extract_target_year(claim)
    if year is None or not candidates:
        return candidates

    axis_matched = [c for c in candidates if _parse_rrf_ranks(c.source_meta).get("axis_value_rank") == 1]
    if axis_matched:
        matched_ids = {c.table_id for c in axis_matched}
        head = [c for c in candidates if c.table_id in matched_ids]
        tail = [c for c in candidates if c.table_id not in matched_ids]
    else:
        head, tail = candidates[:_PERIOD_COVERAGE_TOP_N], candidates[_PERIOD_COVERAGE_TOP_N:]

    kept, excluded = [], []
    for c in head:
        if not c.org_id:
            kept.append(c)
            continue
        period_range = fetch_period_range(c.org_id, c.table_id)
        covers = _covers_year(period_range, year) if period_range else None
        (kept if covers is not False else excluded).append(c)
    return kept + excluded + tail


# 2026-08-25: 원인 C(근접중복·버전분화) 중에서도 _apply_period_coverage_filter가 못
# 거르는 케이스 — 신판이 구판 기간을 통째로 포함해버리면(예: DT_1DA7E06S_NEW가
# 2013~2026 전체를 커버) 구판/신판 둘 다 "그 연도 데이터 있음"으로 통과한다. 이럴 땐
# "그 연도를 커버하는가"가 아니라 "지금도 갱신되는 표인가"로 골라야 한다.
# (STAT_ID, TBL_NM)이 둘 다 같은 후보들을 같은 시리즈의 다른 버전으로 보고, 그 중
# SEND_DE(마지막 자료 제공일)가 가장 최근인 것만 남기고 나머지는 순위 밖으로 민다 —
# KOSIS 자체 크롤 메타데이터로 이미 확인된 값이라 API 호출이 추가로 들지 않는다
# (agent/kosis/version_meta.py 참고). fail-open: 메타 조회 실패/클러스터 없음이면 그대로.
_VERSION_FRESHNESS_ENABLED = os.environ.get("WITH_VERSION_FRESHNESS", "1").strip().lower() not in ("0", "false", "no")


def _apply_version_freshness_signal(candidates: list[TableCandidate]) -> list[TableCandidate]:
    if not _VERSION_FRESHNESS_ENABLED or not candidates:
        return candidates
    meta = get_version_meta_batch([c.table_id for c in candidates])
    if not meta:
        return candidates

    clusters: dict[tuple, list[TableCandidate]] = {}
    for c in candidates:
        m = meta.get(c.table_id)
        if not m:
            continue
        key = (m["stat_id"], m["tbl_nm"])
        clusters.setdefault(key, []).append(c)

    stale_ids: set[str] = set()
    fresh_ids: set[str] = set()
    for members in clusters.values():
        with_date = [c for c in members if meta[c.table_id].get("send_de")]
        if len(with_date) < 2:
            continue  # 클러스터가 진짜 2개 이상이고 날짜 비교가 가능할 때만 판단
        freshest = max(with_date, key=lambda c: meta[c.table_id]["send_de"])
        fresh_ids.add(freshest.table_id)
        stale_ids.update(c.table_id for c in with_date if c.table_id != freshest.table_id)

    if not stale_ids:
        return candidates

    tagged = [
        _tag_source_rank(c, "version_fresh_rank", 1) if c.table_id in fresh_ids else c
        for c in candidates
    ]
    kept = [c for c in tagged if c.table_id not in stale_ids]
    demoted = [c for c in tagged if c.table_id in stale_ids]
    return kept + demoted


def search_and_rerank(
    claim: Claim,
    *,
    keyword_fn,
    embedding_fn,
    vdb_fn=None,
    bm25_fn=None,
    item_fn=None,
    top_k: int = DEFAULT_TOP_K,
    document_texts: Optional[dict[str, str]] = None,
) -> list[TableCandidate]:
    """3단계 전체 흐름: keyword_search + embedding_search(+ VDB dense +VDB BM25 +VDB 항목)
    결과를 합쳐 rerank까지 수행.

    keyword_fn, embedding_fn: 각각 keyword_search(claim), embedding_search(claim) 함수를 주입.
    vdb_fn: (선택) VDB dense 조회 함수 — 넘기면 KOSIS 표 전체(28만7천여 개)도 후보에 포함시킨다.
    bm25_fn: (선택, 2026-08-21 추가) VDB에 trigram(pg_trgm) 기반 텍스트 검색을 하는 함수 —
    agent.kosis.query_vdb.lexical_query_vdb를 claim의 search_query로 감싼 클로저를 넘긴다.
    dense가 못 잡는 "기관명/지표명이 정확히 일치" 케이스를 보완한다(골든셋 실측: HyDE 스타일
    쿼리 기준 Dense Recall@30 61.0% vs BM25 41.5% — 서로 다른 실패 패턴이라 RRF로 합치면
    상호보완 기대).
    item_fn: (선택, 2026-08-23 추가) agent.kosis.query_vdb.item_query_vdb를 감싼 클로저 —
    표 제목이 아니라 표 안의 개별 "항목" 임베딩과 비교한다(원인 A 보완). ⚠️ 이 클로저는
    다른 4개와 같은 블렌딩 검색어를 쓰면 안 되고, claim.statistic_expression만으로 쿼리
    벡터를 만들어야 한다(실측: 블렌딩 검색어로는 항목 단독 유사도가 오히려 떨어짐).
    document_texts: table_id -> 임베딩/설명 텍스트. rerank()로 그대로 전달된다(생략 시
    table_name으로 대체되는데, 리랭커가 짧은 제목만 보고 판단하게 되어 성능이 떨어진다).

    2026-08-25: vdb_fn/bm25_fn은 claim.search_query(또는 sentence) 전체를 하나의 쿼리로
    쓰는데, age/gender/region/population 수식어가 낀 claim은 그 수식어가 핵심 지표 신호를
    밀어내 정답표가 후보 풀에 아예 안 들어오는 경우가 실측 확인됐다(build_core_query() 참고).
    build_core_query()가 core query를 만들 수 있으면(수식어가 있고 statistic_expression도
    있으면) claim의 search_query만 그 core query로 바꾼 사본으로 vdb_fn/bm25_fn을 한 번 더
    호출해서 "vdb_core_rank"/"bm25_core_rank"로 합류시킨다 — vdb_fn/bm25_fn 자체는 그대로
    재사용(별도 함수 없이 dataclasses.replace로 만든 claim 사본만 넘김).
    """
    core_query = build_core_query(claim)
    core_claim = replace(claim, search_query=core_query) if core_query else None

    with _SEARCH_LOCK:
        with Timer() as t_kw:
            kw_results = keyword_fn(claim)
        with Timer() as t_emb:
            emb_results = embedding_fn(claim)
        with Timer() as t_vdb:
            vdb_results = vdb_fn(claim) if vdb_fn else []
        with Timer() as t_bm25:
            bm25_results = bm25_fn(claim) if bm25_fn else []
        with Timer() as t_item:
            item_results = item_fn(claim) if item_fn else []
        with Timer() as t_vdb_core:
            core_vdb_results = vdb_fn(core_claim) if (vdb_fn and core_claim) else []
        with Timer() as t_bm25_core:
            core_bm25_results = bm25_fn(core_claim) if (bm25_fn and core_claim) else []
        merged = _merge_candidates(
            kw_results, emb_results, vdb_results, bm25_results,
            item_results, core_vdb_results, core_bm25_results,
        )
        merged = _apply_population_signal(claim, merged, document_texts)
        merged = _apply_institution_signal(claim, merged, document_texts)
        merged = _apply_gender_signal(claim, merged, document_texts)
        merged = _apply_region_signal(claim, merged, document_texts)
        with Timer() as t_rerank:
            result = rerank(claim, merged, top_k=top_k, document_texts=document_texts)

    # GPU 락 밖에서 실행 — KOSIS API 호출이라 GPU/DB 자원과 무관하고, 락을 오래 붙잡을
    # 이유가 없다(다른 claim의 GPU 작업을 불필요하게 막지 않기 위함).
    # (2026-08-22: _apply_simplicity_tiebreak도 시도했으나 골든셋 70건 재검증에서 효과
    # 없음+1건 순위 악화(7위→9위)로 되돌림 — "월별/연도별" 부스팅 회귀 사례와 같은 종류의
    # 위험한 신호로 판단, 구현은 위에 남겨두되 호출은 뺌.)
    # 2026-08-24: axis_value_signal이 period_coverage_filter보다 먼저 실행돼야 한다 —
    # period 필터가 axis_value_rank 태그를 보고 확인 대상을 좁히기 때문(취업자 구판/신판
    # 클러스터 실측 검증, notebooks 없이 직접 조회로 확인).
    result = _apply_axis_value_signal(claim, result)
    result = _apply_period_coverage_filter(claim, result)
    result = _apply_version_freshness_signal(result)

    # 2026-08-22(관측성): 3단계 검색/리랭킹 한 번을 구조화 로그 한 줄로 남긴다 — 소스별
    # 후보 수·지연시간, 최종 1등 후보/점수를 나중에 집계할 수 있게(예: "이번 배치에서
    # BM25가 실제로 후보를 몇 건이나 보탰나", "리랭커 지연시간 분포"). log_event()는
    # 실패해도 예외를 안 올리므로 여기서 별도 try/except 불필요.
    top1 = result[0] if result else None
    log_event(
        "table_matching",
        sentence=claim.sentence[:200],
        search_query=claim.search_query,
        core_query=core_query,
        kw_count=len(kw_results), emb_count=len(emb_results),
        vdb_count=len(vdb_results), bm25_count=len(bm25_results),
        item_count=len(item_results),
        vdb_core_count=len(core_vdb_results), bm25_core_count=len(core_bm25_results),
        merged_count=len(merged), result_count=len(result),
        top1_table_id=top1.table_id if top1 else None,
        top1_score=round(top1.score, 4) if top1 else None,
        latency_ms={
            "keyword": round(t_kw.elapsed_ms, 1), "embedding": round(t_emb.elapsed_ms, 1),
            "vdb_dense": round(t_vdb.elapsed_ms, 1), "vdb_bm25": round(t_bm25.elapsed_ms, 1),
            "item": round(t_item.elapsed_ms, 1),
            "vdb_dense_core": round(t_vdb_core.elapsed_ms, 1),
            "vdb_bm25_core": round(t_bm25_core.elapsed_ms, 1),
            "rerank": round(t_rerank.elapsed_ms, 1),
        },
        reranker_enabled=not _DISABLE_RERANKER,
    )
    return result


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
    document_texts = load_document_texts()
    for c in test_claims:
        result = search_and_rerank(
            c,
            keyword_fn=keyword_search,
            embedding_fn=lambda claim: embedding_search(claim, cache=cache),
            document_texts=document_texts,
        )
        print(f"\n[{c.sentence}]")
        for r in result[:3]:
            print(f"  - {r.table_name} ({r.table_id}) score={r.score:.3f} | {r.source_meta}")
