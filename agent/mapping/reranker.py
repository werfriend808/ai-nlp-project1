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
) -> list[TableCandidate]:
    """keyword_search/embedding_search/VDB 후보를 table_id 기준으로 합치면서, 각 소스
    안에서의 순위(1부터, 입력 리스트가 이미 점수 내림차순 정렬돼 있다고 가정)를
    source_meta에 "keyword_rank=N"/"embedding_rank=N"/"vdb_rank=N"으로 남긴다 —
    이후 _rrf_fuse()가 점수 크기가 아니라 이 순위만으로 최종 신뢰도를 계산한다.
    입력으로 받은 candidate 객체는 변형하지 않고 dataclasses.replace로 복사본만 만든다.

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
    return list(merged.values())


_RANK_TAG_RE = re.compile(r"\b(keyword_rank|embedding_rank|vdb_rank|reranker_rank|population_rank)=(\d+)")


def _parse_rrf_ranks(source_meta: Optional[str]) -> dict[str, int]:
    """source_meta 문자열에서 "{key}=N" 형태의 순위 태그를 전부 뽑아 dict로 돌려준다."""
    return {key: int(val) for key, val in _RANK_TAG_RE.findall(source_meta or "")}


RRF_K = 60  # RRF 원 논문(Cormack et al., 2009)의 관례값. 작을수록 상위 순위 후보에 더 민감해진다.


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

    2026-08-21: population_rank(아래 _apply_population_signal 참고)도 keyword_rank와
    동급으로 신뢰한다 — claim 문장에서 직접 뽑은 명시적 규칙 기반 신호라 임베딩/리랭커의
    노이즈 있는 유사도 판단보다 신뢰도가 높다고 보기 때문.
    """
    ranks = _parse_rrf_ranks(source_meta)
    if "keyword_rank" in ranks:
        return True
    if "population_rank" in ranks:
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
    top_k: int = 5,
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


def search_and_rerank(
    claim: Claim,
    *,
    keyword_fn,
    embedding_fn,
    vdb_fn=None,
    top_k: int = 5,
    document_texts: Optional[dict[str, str]] = None,
) -> list[TableCandidate]:
    """3단계 전체 흐름: keyword_search + embedding_search(+ VDB) 결과를 합쳐 rerank까지 수행.

    keyword_fn, embedding_fn: 각각 keyword_search(claim), embedding_search(claim) 함수를 주입.
    vdb_fn: (선택) VDB 조회 함수 — 넘기면 KOSIS 표 전체(28만7천여 개)도 후보에 포함시킨다.
    document_texts: table_id -> 임베딩/설명 텍스트. rerank()로 그대로 전달된다(생략 시
    table_name으로 대체되는데, 리랭커가 짧은 제목만 보고 판단하게 되어 성능이 떨어진다).
    """
    kw_results = keyword_fn(claim)
    emb_results = embedding_fn(claim)
    vdb_results = vdb_fn(claim) if vdb_fn else []
    merged = _merge_candidates(kw_results, emb_results, vdb_results)
    merged = _apply_population_signal(claim, merged, document_texts)
    return rerank(claim, merged, top_k=top_k, document_texts=document_texts)


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
