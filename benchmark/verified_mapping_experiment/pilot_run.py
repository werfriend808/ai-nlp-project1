"""
benchmark/verified_mapping_experiment/pilot_run.py
====================================================
2,500건 Verified Mapping + Retrieval Cascade 실험 — 50건 PILOT 스코프 전용 실행기.

절대 원칙(methodology.md 10번 참고):
  - production DB(SUPABASE_DB_URL)는 SELECT만.
  - production 코드 파일은 전부 import만 하고 수정하지 않는다.
  - production 로컬 SQLite(data/verifications.db)에도 쓰지 않는다
    (db.store.insert_verification을 이 스크립트는 호출하지 않음).
  - gold(70건 골든셋)는 discovery/HIGH mapping 생성에 절대 쓰지 않는다 — 여기서는
    아예 로드조차 하지 않고, leakage 체크에만 별도로 읽는다.
  - retrieval Top-1을 자동 정답으로 보지 않는다 — judge()의 실제 KOSIS 수치 비교를
    거쳐야만 HIGH.

이 스크립트는 50건 pilot 전용이다 — 2,706건 전체나 discovery/validation/evaluation
분할은 여기서 하지 않는다(사용자가 명시적으로 금지).
"""

from __future__ import annotations

import io
import csv
import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # production 모듈들이 상대경로(prompts/, agent/kosis/table_params.json 등)를 기대함

EXP_DIR = Path(__file__).resolve().parent
N_PILOT = int(os.environ.get("PILOT_N", "50"))  # 스모크 테스트 시 PILOT_N=2 등으로 축소 가능
SEED = 20260830  # batch_runner 기본 seed(42)와 겹치지 않게 별도 지정 — 재현 가능한 무작위 표본

# --- production 코드 (전부 import만, 수정 없음) -----------------------------------
from agent.preprocessing import classifier as classifier_mod
from agent.preprocessing import claim_extractor as claim_extractor_mod
from agent.preprocessing.source_filter import resolve_claim_sources, filter_verifiable_claims
from agent.mapping.keyword_search import keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import (
    search_and_rerank, load_document_texts, _parse_rrf_ranks, is_rrf_trusted,
    expand_institution_query_aliases, _DISABLE_RERANKER,
)
from agent.orchestrator.calc_type_router import route_calc_type, _mentions_foreign_country
from agent.kosis.api_client import KosisApiClient, KosisApiError
from agent.kosis.calculator import KosisCalculator, CalculationError
from agent.verdict.judge import judge, JudgeError
from agent.interfaces import Claim, dense_query_text
import agent.preprocessing.hcx_client as hcx_client_mod

from agent.pipeline.batch_runner import (
    _load_csv_rows, _row_to_article, _load_table_catalog_by_id, _dedup_claims_by_sentence,
    run_stage_4, run_stage_5_6, select_itm_id, select_dimension_values,
    TABLE_PARAMS_PATH, DATA_CSV_PATH,
)

EVAL_SET_PATH = ROOT / "benchmark" / "search_experiment" / "eval_set.json"


# ---------------------------------------------------------------------------
# HCX / KOSIS 호출 계측 (production 파일 수정 없이, 모듈 attribute만 런타임에 감쌈)
# ---------------------------------------------------------------------------

class CallCounter:
    def __init__(self):
        self.counts: dict[str, int] = {}
        self.total_ms: dict[str, float] = {}
        self.errors: dict[str, int] = {}

    def record(self, tag: str, elapsed_ms: float, ok: bool):
        self.counts[tag] = self.counts.get(tag, 0) + 1
        self.total_ms[tag] = self.total_ms.get(tag, 0.0) + elapsed_ms
        if not ok:
            self.errors[tag] = self.errors.get(tag, 0) + 1

    def summary(self) -> dict:
        return {
            tag: {
                "calls": self.counts[tag],
                "total_ms": round(self.total_ms[tag], 1),
                "avg_ms": round(self.total_ms[tag] / self.counts[tag], 1),
                "errors": self.errors.get(tag, 0),
            }
            for tag in self.counts
        }


HCX_COUNTER = CallCounter()
KOSIS_COUNTER = CallCounter()


def _wrap_hcx(tag: str, fn):
    def wrapped(*args, **kwargs):
        t0 = time.perf_counter()
        ok = True
        try:
            return fn(*args, **kwargs)
        except Exception:
            ok = False
            raise
        finally:
            HCX_COUNTER.record(tag, (time.perf_counter() - t0) * 1000, ok)

    return wrapped


def install_hcx_instrumentation():
    """classifier/claim_extractor/slot_filler는 각자 `call_hcx`를 자기 모듈 네임스페이스로
    직접 import해서 갖고 있어서(`from .hcx_client import call_hcx`), hcx_client 모듈 쪽
    속성만 바꿔서는 이미 바인딩된 이름에 영향이 없다 — 소비하는 모듈 쪽 속성을 각각 감싼다.
    judge.py는 함수 안에서 매번 새로 `from ... import call_hcx`를 하므로 hcx_client 쪽만
    감싸면 충분하다."""
    from agent.orchestrator import slot_filler as slot_filler_mod

    classifier_mod.call_hcx = _wrap_hcx("classify", classifier_mod.call_hcx)
    claim_extractor_mod.call_hcx = _wrap_hcx("claim_extract", claim_extractor_mod.call_hcx)
    slot_filler_mod.call_hcx = _wrap_hcx("slot_filler", slot_filler_mod.call_hcx)
    hcx_client_mod.call_hcx = _wrap_hcx("judge_or_other", hcx_client_mod.call_hcx)


class CountingKosisApiClient(KosisApiClient):
    def _request(self, params):
        t0 = time.perf_counter()
        ok = True
        try:
            return super()._request(params)
        except Exception:
            ok = False
            raise
        finally:
            KOSIS_COUNTER.record("kosis_api", (time.perf_counter() - t0) * 1000, ok)


# ---------------------------------------------------------------------------
# 확장 taxonomy(spec 4번) — 규칙 기반 재분류. LLM 재호출도, 새 프롬프트도 아님.
# production claim_type(규모/증감률/비교/전망) + route_calc_type() 결과 + 원문 정규식
# 신호만으로 numeric/comparative/superlative/trend/threshold/qualitative를 매긴다.
# ---------------------------------------------------------------------------

import re

_TREND_RE = re.compile(r"(\d+)\s*(년|개월|분기)\s*(연속|째)|연속\s*(증가|감소|상승|하락)")
_THRESHOLD_OP = {"초과", "미만"}


def extended_claim_type(claim: Claim, calc_type: Optional[str]) -> tuple[str, bool]:
    """(extended_type, verifiable) 반환. verifiable=False면 애초에 정량 검증 대상이 아님
    (production도 동일 판단 — 전망/해외는 route_calc_type이 이미 None)."""
    if claim.claim_type == "전망":
        return "qualitative", False
    if _mentions_foreign_country(claim.population, claim.region, claim.comparison_target):
        return "qualitative", False
    if calc_type is None:
        return "qualitative", False
    if calc_type in ("최댓값검증", "최솟값검증"):
        return "superlative", True
    if _TREND_RE.search(claim.sentence or ""):
        return "trend", True
    if calc_type in ("증감", "증감률"):
        return "comparative", True
    if calc_type == "단순조회":
        if claim.comparison_operator in _THRESHOLD_OP:
            return "threshold", True
        return "numeric", True
    return "numeric", True


_COMPARISON_OP_MAP = {
    "증가": "increase", "감소": "decrease", "동일": "eq", "초과": "gt", "미만": "lt",
}


def normalize_comparison_operator(claim: Claim, extended_type: str) -> Optional[str]:
    if extended_type == "trend":
        return "trend"
    if claim.comparison_operator:
        return _COMPARISON_OP_MAP.get(claim.comparison_operator)
    return None


# ---------------------------------------------------------------------------
# 50건 표본 로드 + leakage 체크
# ---------------------------------------------------------------------------

def load_pilot_articles(n: int = N_PILOT, seed: int = SEED) -> list[dict]:
    rows = _load_csv_rows(DATA_CSV_PATH)
    # _load_csv_rows는 이미 원본 순서를 유지한다 — row_index를 셔플 전에 부여해서
    # article_id가 "원본 CSV에서 몇 번째 True 행이었는지"를 그대로 보존하게 한다.
    for i, r in enumerate(rows):
        r["_row_index"] = i
    random.Random(seed).shuffle(rows)
    picked = rows[:n]
    articles = []
    for r in picked:
        art = _row_to_article(r)
        art["article_id"] = f"row{r['_row_index']}"
        articles.append(art)
    return articles


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def check_leakage(pilot_articles: list[dict]) -> list[dict]:
    """70건 골든셋 sentence가 pilot 기사 본문에 부분 문자열로 포함되는지 검사한다
    (methodology.md 5번 참고 — eval_set.json에 URL/제목이 없어 문장 포함 여부로 대체).
    gold는 이 함수에서조차 읽지 않는다(sentence/claim_id만 사용) — mapping 생성에 gold를
    쓰지 않는다는 원칙을 leakage 체크 자체에도 지킨다."""
    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    hits = []
    for art in pilot_articles:
        normalized_article = _normalize_ws(art["article_text"])
        for row in eval_set:
            sent = row.get("sentence") or ""
            if not sent:
                continue
            if _normalize_ws(sent) in normalized_article:
                hits.append({
                    "article_id": art["article_id"],
                    "article_title": art["article_title"],
                    "eval_claim_id": row["claim_id"],
                    "eval_sentence": sent,
                })
    return hits


# ---------------------------------------------------------------------------
# claim 1건 처리: retrieval -> (top1) slot fill -> KOSIS 조회 -> judge -> confidence
# ---------------------------------------------------------------------------

def _candidate_row(claim_id: str, rank: int, cand) -> dict:
    ranks = _parse_rrf_ranks(cand.source_meta)
    return {
        "claim_id": claim_id,
        "retrieval_rank": rank,
        "table_id": cand.table_id,
        "table_name": cand.table_name,
        "org_id": cand.org_id,
        "rrf_score": round(cand.score, 6),
        "dense_rank": ranks.get("vdb_rank"),
        "bm25_rank": ranks.get("bm25_rank"),
        "keyword_rank": ranks.get("keyword_rank"),
        "embedding_rank": ranks.get("embedding_rank"),
        "reranker_rank": ranks.get("reranker_rank"),
        "is_rrf_trusted": is_rrf_trusted(cand.source_meta),
        "source_meta": cand.source_meta,
    }


def enrich_candidates_with_db(candidates_rows: list[dict], db_conn) -> None:
    """kosis_vdb_tables_qwen에서 organization/period 메타를 배치로 채운다(SELECT만)."""
    table_ids = sorted({r["table_id"] for r in candidates_rows})
    if not table_ids or db_conn is None:
        return
    with db_conn.cursor() as cur:
        cur.execute(
            "select table_id, institution_name, topic, period_start, period_end "
            "from kosis_vdb_tables_qwen where table_id = any(%s)",
            (table_ids,),
        )
        meta = {row[0]: {"organization": row[1], "topic": row[2],
                          "period_start": row[3], "period_end": row[4]} for row in cur.fetchall()}
    for r in candidates_rows:
        m = meta.get(r["table_id"], {})
        r["organization"] = m.get("organization")
        r["topic"] = m.get("topic")
        r["period_start"] = m.get("period_start")
        r["period_end"] = m.get("period_end")


CONFIDENCE_UNKNOWN_NO_CLAIM_TYPE = "UNKNOWN"


def classify_confidence(stage: str, judge_verdict: Optional[str]) -> tuple[str, str]:
    """(confidence, reason). methodology.md 6번에서 이 함수 하나로 고정한다고 명시함 —
    2,706건 확장 시에도 이 규칙을 재조정하지 않을 계획."""
    if stage == "skipped_qualitative":
        return "UNKNOWN", "claim_type=전망 또는 해외 국가 포함 — 정량 검증 대상 아님"
    if stage == "no_candidate":
        return "UNKNOWN", "retrieval 후보 없음(Dense+BM25+RRF 전부 매칭 실패)"
    if stage == "untrusted_top1":
        return "LOW", "최상위 후보가 RRF 기준으로 신뢰도 낮음(keyword 미발견+리랭커 비신뢰)"
    if stage == "no_calc_route":
        return "UNKNOWN", "calc_type_router가 규칙 기반 라우팅 불가 판정"
    if stage == "slot_fill_incomplete":
        return "MEDIUM", "후보는 있으나 슬롯 채우기/되묻기 미해결(기간·지역 등 불명확)"
    if stage == "kosis_fetch_failed":
        return "MEDIUM", "후보+슬롯은 있으나 KOSIS 실측값 조회 실패/미지원(표 구조 불일치 등)"
    if stage == "verified":
        if judge_verdict == "일치":
            return "HIGH", "실제 KOSIS 수치와 독립 검증 결과 일치"
        if judge_verdict == "불일치":
            return "LOW", "구조적으로는 후보가 그럴듯하나 실제 KOSIS 수치와 불일치(mapping conflict형)"
        return "MEDIUM", "실제 KOSIS 수치까지 조회했으나 judge()가 판단불가(주제 불일치/모호 등)"
    return "UNKNOWN", f"미분류 단계: {stage}"


def process_claim(
    article: dict, claim: Claim, claim_id: str,
    table_params: dict, catalog_by_id: dict, doc_texts: dict,
    vdb_fn, bm25_fn, client: KosisApiClient, calculator: KosisCalculator,
    db_conn,
) -> tuple[dict, list[dict], Optional[dict]]:
    """claim 1건을 3~8단계(축소판)로 처리한다. production insert_verification은 호출하지
    않는다(로컬 SQLite에 실험 데이터가 섞이는 걸 피하기 위함, methodology.md 3번).
    반환: (claim_record, candidate_rows, verified_mapping_row_or_None)"""
    calc_type = route_calc_type(claim)
    ext_type, verifiable = extended_claim_type(claim, calc_type)

    claim_record = {
        "claim_id": claim_id,
        "article_id": article["article_id"],
        "article_title": article["article_title"],
        "article_url": article.get("article_url"),
        "published_date": str(article["published_date"]),
        "sentence": claim.sentence,
        "prev_sentence": claim.prev_sentence,
        "production_claim_type": claim.claim_type,
        "extended_claim_type": ext_type,
        "verifiable": verifiable,
        "metric": claim.statistic_expression,
        "value": claim.value,
        "unit": claim.unit,
        "period": claim.period,
        "comparison_period": claim.comparison_target,
        "comparison_operator": normalize_comparison_operator(claim, ext_type),
        "comparison_value": claim.comparison_value,
        "region": claim.region,
        "population": claim.population,
        "gender": claim.gender,
        "age": claim.age,
        "organization": claim.source_org,
        "routed_calc_type": calc_type,
    }

    if not verifiable:
        confidence, reason = classify_confidence("skipped_qualitative", None)
        claim_record.update(confidence=confidence, confidence_reason=reason,
                             mapping_status="not_applicable", num_candidates=0)
        return claim_record, [], None

    try:
        candidates = search_and_rerank(
            claim, keyword_fn=keyword_search,
            embedding_fn=lambda c: embedding_search(c, cache=doc_texts["_emb_cache"]),
            vdb_fn=vdb_fn, bm25_fn=bm25_fn, top_k=20,
            document_texts=doc_texts["texts"],
        )
    except Exception as e:
        claim_record.update(confidence="UNKNOWN",
                             confidence_reason=f"retrieval 실패: {type(e).__name__}: {e}",
                             mapping_status="error", num_candidates=0)
        return claim_record, [], None

    if not candidates:
        confidence, reason = classify_confidence("no_candidate", None)
        claim_record.update(confidence=confidence, confidence_reason=reason,
                             mapping_status="no_mapping_fallback_empty", num_candidates=0)
        return claim_record, [], None

    candidate_rows = [_candidate_row(claim_id, i + 1, c) for i, c in enumerate(candidates)]
    enrich_candidates_with_db(candidate_rows, db_conn)
    claim_record["num_candidates"] = len(candidates)
    claim_record["top1_table_id"] = candidates[0].table_id
    claim_record["top1_table_name"] = candidates[0].table_name

    top = candidates[0]
    if top.source_meta and not is_rrf_trusted(top.source_meta):
        confidence, reason = classify_confidence("untrusted_top1", None)
        claim_record.update(confidence=confidence, confidence_reason=reason,
                             mapping_status="no_mapping_fallback_untrusted")
        return claim_record, candidate_rows, None

    try:
        slots = run_stage_4(
            claim.sentence, article.get("clarify_reply"), article["published_date"],
            table_id=top.table_id, table_params=table_params,
            catalog_by_id=catalog_by_id, claim_region=claim.region,
        )
    except Exception as e:
        claim_record.update(confidence="MEDIUM",
                             confidence_reason=f"슬롯 채우기 예외: {type(e).__name__}: {e}",
                             mapping_status="slot_fill_error")
        return claim_record, candidate_rows, None

    if slots is None:
        confidence, reason = classify_confidence("slot_fill_incomplete", None)
        claim_record.update(confidence=confidence, confidence_reason=reason,
                             mapping_status="no_mapping_fallback_slotfail")
        return claim_record, candidate_rows, None

    if calc_type is None:
        confidence, reason = classify_confidence("no_calc_route", None)
        claim_record.update(confidence=confidence, confidence_reason=reason,
                             mapping_status="not_applicable")
        return claim_record, candidate_rows, None
    slots["calc_type"] = calc_type

    selected_itm = select_itm_id(top.table_id, claim, table_params)
    if selected_itm:
        slots["itm_id"] = selected_itm
    dim_values = select_dimension_values(top.table_id, claim, table_params, slots)
    if dim_values:
        slots.update(dim_values)

    try:
        computed = run_stage_5_6(
            top.table_id, slots, table_params, client, calculator,
            comparison_target=claim.comparison_target, claim_sentence=claim.sentence,
            article_year=article["published_date"].year, org_id=top.org_id, claim=claim,
        )
    except (KosisApiError, CalculationError, KeyError) as e:
        confidence, reason = classify_confidence("kosis_fetch_failed", None)
        claim_record.update(confidence=confidence, confidence_reason=f"{reason} ({type(e).__name__}: {e})",
                             mapping_status="no_mapping_fallback_kosiserror")
        return claim_record, candidate_rows, None
    except Exception as e:
        confidence, reason = classify_confidence("kosis_fetch_failed", None)
        claim_record.update(confidence=confidence, confidence_reason=f"{reason} (예상외 예외 {type(e).__name__}: {e})",
                             mapping_status="no_mapping_fallback_error")
        return claim_record, candidate_rows, None

    if computed is None:
        confidence, reason = classify_confidence("kosis_fetch_failed", None)
        claim_record.update(confidence=confidence, confidence_reason=reason,
                             mapping_status="no_mapping_fallback_kosisnone")
        return claim_record, candidate_rows, None

    try:
        verdict = judge(
            claim, computed, prd_se=slots.get("prd_se"),
            article_date=str(article["published_date"]), matched_table_name=top.table_name,
        )
    except JudgeError as e:
        confidence, reason = classify_confidence("kosis_fetch_failed", None)
        claim_record.update(confidence=confidence,
                             confidence_reason=f"judge() 실패({type(e).__name__}: {e}), 실측값은 조회됨",
                             mapping_status="judge_error",
                             computed_value=computed.raw_value, computed_unit=computed.unit,
                             computed_period=computed.period)
        return claim_record, candidate_rows, None

    confidence, reason = classify_confidence("verified", verdict.verdict)
    claim_record.update(
        confidence=confidence, confidence_reason=reason,
        judge_verdict=verdict.verdict, judge_gap_type=verdict.gap_type, judge_reason=verdict.reason,
        computed_value=computed.raw_value, computed_unit=computed.unit, computed_period=computed.period,
        mapping_status="independent_verification_hit" if confidence == "HIGH"
        else ("mapping_conflict" if verdict.verdict == "불일치" else "no_mapping_fallback_inconclusive"),
    )

    mapping_row = None
    if confidence == "HIGH":
        mapping_row = {
            "claim_id": claim_id,
            "concept": f"{claim.statistic_expression or claim.population or ext_type}"
                       f"|{claim.region or 'ALL'}|{top.table_id}",
            "claim_text": claim.sentence,
            "table_id": top.table_id,
            "table_name": top.table_name,
            "metric": claim.statistic_expression,
            "organization": claim.source_org,
            "period": computed.period,
            "region": claim.region,
            "axis": {"gender": claim.gender, "age": claim.age},
            "confidence": "HIGH",
            "mapping_source": "first_time_discovery",  # 이번 pilot엔 재사용 mapping이 없음
            "independent_verification": True,
            "verification_evidence": {
                "retrieval_evidence": {
                    "top1_source_meta": top.source_meta,
                    "num_candidates": len(candidates),
                },
                "metadata_evidence": {
                    "organization_claim": claim.source_org,
                    "period_claim": claim.period,
                },
                "numeric_evidence": {
                    "claim_value": claim.value, "computed_value": computed.raw_value,
                    "unit": computed.unit, "period": computed.period,
                },
                "judge_reason": verdict.reason,
            },
        }
    return claim_record, candidate_rows, mapping_row


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def build_vdb_bm25_fns():
    from agent.kosis.query_vdb import batch_query_vdb, bm25_query_vdb, lexical_query_vdb, VdbUnavailableError, VDB_TOP_K, LEXICAL_TOP_K
    from sentence_transformers import SentenceTransformer

    print("[준비] Qwen3-Embedding-4B 로딩 중...")
    t0 = time.perf_counter()
    vdb_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560)
    print(f"[준비] 로딩 완료 ({time.perf_counter() - t0:.1f}s)")
    vdb_instruction = (
        "Given a Korean news claim sentence, retrieve the KOSIS statistical table "
        "description that best matches it"
    )

    def _retrieval_query_text(claim) -> str:
        base = claim.search_query or dense_query_text(claim)
        return expand_institution_query_aliases(base, claim.source_org)

    def vdb_fn(claim):
        text = f"Instruct: {vdb_instruction}\nQuery: {_retrieval_query_text(claim)}"
        vec = vdb_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        try:
            return batch_query_vdb([vec], top_k=VDB_TOP_K)[0]
        except VdbUnavailableError:
            return []

    def bm25_fn(claim):
        query_text = _retrieval_query_text(claim)
        try:
            return bm25_query_vdb(query_text, top_k=LEXICAL_TOP_K)
        except VdbUnavailableError:
            pass
        try:
            return lexical_query_vdb(query_text, top_k=LEXICAL_TOP_K)
        except VdbUnavailableError:
            return []

    return vdb_fn, bm25_fn


def main():
    t_start = time.perf_counter()
    install_hcx_instrumentation()

    print(f"[설정] _DISABLE_RERANKER(CE 비활성화 여부) = {_DISABLE_RERANKER}")
    assert _DISABLE_RERANKER is True, "이번 실험은 CE 없는 RRF-only 기준이어야 한다(spec 전제)"

    print(f"[1/6] 50건 표본 로드 (seed={SEED})")
    articles = load_pilot_articles()
    print(f"  -> {len(articles)}건 로드")

    print("[2/6] 70건 골든셋 leakage 체크")
    leakage_hits = check_leakage(articles)
    print(f"  -> leakage 후보 {len(leakage_hits)}건")

    print("[3/6] production 리소스 준비 (table_params/catalog/embedding_cache/DB 연결)")
    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)
    catalog_by_id = _load_table_catalog_by_id()
    embedding_cache = build_table_embedding_cache()
    doc_texts_map = load_document_texts()
    doc_texts = {"_emb_cache": embedding_cache, "texts": doc_texts_map}

    db_conn = None
    try:
        import psycopg2
        db_conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
        db_conn.autocommit = True
    except Exception as e:
        print(f"[경고] DB 연결 실패({e}) — 후보 organization/period 보강 없이 진행")

    client = CountingKosisApiClient()
    calculator = KosisCalculator()

    print("[4/6] VDB(dense)/BM25 함수 준비 (GPU)")
    vdb_fn, bm25_fn = build_vdb_bm25_fns()

    claims_out, candidates_out, mappings_out = [], [], []
    stage_timers = {"classify": 0.0, "extract": 0.0, "retrieval_and_verify": 0.0}
    n_articles_relevant = 0
    n_claims_total = 0
    failed_articles = []

    print(f"[5/6] {len(articles)}건 기사 처리 시작")
    for idx, article in enumerate(articles):
        print(f"\n--- [{idx + 1}/{len(articles)}] {article['article_id']} {article['article_title'][:40]!r}")
        try:
            t0 = time.perf_counter()
            cls_result = classifier_mod.classify(article["article_text"])
            stage_timers["classify"] += time.perf_counter() - t0
            if not cls_result.label:
                print("  [1단계] 무관 판정 -> 스킵")
                continue
            n_articles_relevant += 1

            t0 = time.perf_counter()
            claims = claim_extractor_mod.extract_claims(article["article_text"])
            claims = claim_extractor_mod.recover_missed_claims(article["article_text"], claims)
            claims = claim_extractor_mod.strip_title_prefix_from_claims(claims, article.get("article_title"))
            claims = _dedup_claims_by_sentence(claims)
            claims = resolve_claim_sources(claims, cls_result.reason)
            claims = filter_verifiable_claims(claims)
            stage_timers["extract"] += time.perf_counter() - t0
            print(f"  [2단계] claim {len(claims)}건 추출(필터 후)")

            for c_idx, claim in enumerate(claims):
                claim_id = f"{article['article_id']}-c{c_idx}"
                n_claims_total += 1
                t0 = time.perf_counter()
                try:
                    claim_rec, cand_rows, mapping_row = process_claim(
                        article, claim, claim_id, table_params, catalog_by_id, doc_texts,
                        vdb_fn, bm25_fn, client, calculator, db_conn,
                    )
                except Exception as e:
                    traceback.print_exc()
                    claim_rec = {
                        "claim_id": claim_id, "article_id": article["article_id"],
                        "sentence": claim.sentence, "confidence": "UNKNOWN",
                        "confidence_reason": f"처리 중 예상외 예외: {type(e).__name__}: {e}",
                        "mapping_status": "error",
                    }
                    cand_rows, mapping_row = [], None
                stage_timers["retrieval_and_verify"] += time.perf_counter() - t0
                claims_out.append(claim_rec)
                candidates_out.extend(cand_rows)
                if mapping_row:
                    mappings_out.append(mapping_row)
                print(f"    claim[{c_idx}] confidence={claim_rec.get('confidence')} "
                      f"({claim_rec.get('confidence_reason', '')[:60]})")
        except Exception as e:
            traceback.print_exc()
            failed_articles.append({"article_id": article["article_id"], "error": str(e)})

    total_elapsed = time.perf_counter() - t_start

    print("[6/6] 결과 저장")
    with open(EXP_DIR / "pilot_claims.jsonl", "w", encoding="utf-8") as f:
        for row in claims_out:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with open(EXP_DIR / "pilot_candidates.jsonl", "w", encoding="utf-8") as f:
        for row in candidates_out:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with open(EXP_DIR / "pilot_verified_mappings.jsonl", "w", encoding="utf-8") as f:
        for row in mappings_out:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    run_stats = {
        "n_articles_sampled": len(articles),
        "n_articles_relevant": n_articles_relevant,
        "n_claims_total": n_claims_total,
        "n_failed_articles": len(failed_articles),
        "failed_articles": failed_articles,
        "leakage_hits": leakage_hits,
        "stage_timers_sec": {k: round(v, 1) for k, v in stage_timers.items()},
        "total_elapsed_sec": round(total_elapsed, 1),
        "hcx_calls": HCX_COUNTER.summary(),
        "kosis_calls": KOSIS_COUNTER.summary(),
        "confidence_distribution": {},
        "extended_claim_type_distribution": {},
        "mapping_status_distribution": {},
    }
    for row in claims_out:
        conf = row.get("confidence", "UNKNOWN")
        run_stats["confidence_distribution"][conf] = run_stats["confidence_distribution"].get(conf, 0) + 1
        et = row.get("extended_claim_type", "unknown")
        run_stats["extended_claim_type_distribution"][et] = run_stats["extended_claim_type_distribution"].get(et, 0) + 1
        ms = row.get("mapping_status", "unknown")
        run_stats["mapping_status_distribution"][ms] = run_stats["mapping_status_distribution"].get(ms, 0) + 1

    with open(EXP_DIR / "pilot_run_stats.json", "w", encoding="utf-8") as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== 완료 ===")
    print(json.dumps(run_stats, ensure_ascii=False, indent=2, default=str))

    if db_conn is not None:
        db_conn.close()


if __name__ == "__main__":
    main()
