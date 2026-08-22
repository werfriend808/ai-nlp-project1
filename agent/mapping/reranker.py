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
) -> list[TableCandidate]:
    """keyword_search/embedding_search/VDB(dense)/BM25(trigram) 후보를 table_id 기준으로
    합치면서, 각 소스 안에서의 순위(1부터, 입력 리스트가 이미 점수 내림차순 정렬돼 있다고
    가정)를 source_meta에 "keyword_rank=N"/"embedding_rank=N"/"vdb_rank=N"/"bm25_rank=N"으로
    남긴다 — 이후 _rrf_fuse()가 점수 크기가 아니라 이 순위만으로 최종 신뢰도를 계산한다.
    입력으로 받은 candidate 객체는 변형하지 않고 dataclasses.replace로 복사본만 만든다.

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
                merged[cand.table_id] = replace(
                    existing, source_meta=f"{existing.source_meta} | {key}={i + 1}"
                )

    _add(keyword_candidates, "keyword_rank")
    _add(embedding_candidates, "embedding_rank")
    _add(vdb_candidates or [], "vdb_rank")
    _add(bm25_candidates or [], "bm25_rank")
    return list(merged.values())


_RANK_TAG_RE = re.compile(
    r"\b(keyword_rank|embedding_rank|vdb_rank|bm25_rank|reranker_rank|population_rank"
    r"|institution_rank|gender_rank|region_rank)=(\d+)"
)


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
    """
    ranks = _parse_rrf_ranks(source_meta)
    if "keyword_rank" in ranks:
        return True
    if any(k in ranks for k in ("population_rank", "institution_rank", "gender_rank", "region_rank")):
        return True
    if ranks.get("reranker_rank") != 1:
        return False
    raw_match = _RERANK_RAW_RE.search(source_meta or "")
    if not raw_match:
        return False
    return _sigmoid(float(raw_match.group(1))) >= MIN_RERANKER_CONFIDENCE


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

    documents = [
        (document_texts or {}).get(c.table_id, c.table_name) for c in candidates
    ]
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
        reranked.append(
            TableCandidate(
                table_id=cand.table_id,
                table_name=cand.table_name,
                score=_sigmoid(score),
                required_slots=cand.required_slots,
                source_meta=f"{cand.source_meta} | rerank_raw={score:.3f}",
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
def _apply_period_coverage_filter(claim: Claim, candidates: list[TableCandidate]) -> list[TableCandidate]:
    year = _extract_target_year(claim)
    if year is None or not candidates:
        return candidates

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


def search_and_rerank(
    claim: Claim,
    *,
    keyword_fn,
    embedding_fn,
    vdb_fn=None,
    bm25_fn=None,
    top_k: int = DEFAULT_TOP_K,
    document_texts: Optional[dict[str, str]] = None,
) -> list[TableCandidate]:
    """3단계 전체 흐름: keyword_search + embedding_search(+ VDB dense +VDB BM25) 결과를 합쳐
    rerank까지 수행.

    keyword_fn, embedding_fn: 각각 keyword_search(claim), embedding_search(claim) 함수를 주입.
    vdb_fn: (선택) VDB dense 조회 함수 — 넘기면 KOSIS 표 전체(28만7천여 개)도 후보에 포함시킨다.
    bm25_fn: (선택, 2026-08-21 추가) VDB에 trigram(pg_trgm) 기반 텍스트 검색을 하는 함수 —
    agent.kosis.query_vdb.lexical_query_vdb를 claim의 search_query로 감싼 클로저를 넘긴다.
    dense가 못 잡는 "기관명/지표명이 정확히 일치" 케이스를 보완한다(골든셋 실측: HyDE 스타일
    쿼리 기준 Dense Recall@30 61.0% vs BM25 41.5% — 서로 다른 실패 패턴이라 RRF로 합치면
    상호보완 기대).
    document_texts: table_id -> 임베딩/설명 텍스트. rerank()로 그대로 전달된다(생략 시
    table_name으로 대체되는데, 리랭커가 짧은 제목만 보고 판단하게 되어 성능이 떨어진다).
    """
    with _SEARCH_LOCK:
        with Timer() as t_kw:
            kw_results = keyword_fn(claim)
        with Timer() as t_emb:
            emb_results = embedding_fn(claim)
        with Timer() as t_vdb:
            vdb_results = vdb_fn(claim) if vdb_fn else []
        with Timer() as t_bm25:
            bm25_results = bm25_fn(claim) if bm25_fn else []
        merged = _merge_candidates(kw_results, emb_results, vdb_results, bm25_results)
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
    result = _apply_period_coverage_filter(claim, result)

    # 2026-08-22(관측성): 3단계 검색/리랭킹 한 번을 구조화 로그 한 줄로 남긴다 — 소스별
    # 후보 수·지연시간, 최종 1등 후보/점수를 나중에 집계할 수 있게(예: "이번 배치에서
    # BM25가 실제로 후보를 몇 건이나 보탰나", "리랭커 지연시간 분포"). log_event()는
    # 실패해도 예외를 안 올리므로 여기서 별도 try/except 불필요.
    top1 = result[0] if result else None
    log_event(
        "table_matching",
        sentence=claim.sentence[:200],
        search_query=claim.search_query,
        kw_count=len(kw_results), emb_count=len(emb_results),
        vdb_count=len(vdb_results), bm25_count=len(bm25_results),
        merged_count=len(merged), result_count=len(result),
        top1_table_id=top1.table_id if top1 else None,
        top1_score=round(top1.score, 4) if top1 else None,
        latency_ms={
            "keyword": round(t_kw.elapsed_ms, 1), "embedding": round(t_emb.elapsed_ms, 1),
            "vdb_dense": round(t_vdb.elapsed_ms, 1), "vdb_bm25": round(t_bm25.elapsed_ms, 1),
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
