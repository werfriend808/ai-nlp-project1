"""
benchmark/verified_mapping_experiment/pilot20_run.py
======================================================
Verified Mapping + Retrieval Cascade — 20건 Pilot (Discovery 10 / Evaluation 10).

이 파일은 50건 pilot(pilot_run.py)을 "처음부터 다시 만들지 않고" 그대로 import해서
재사용한다(CallCounter/install_hcx_instrumentation/CountingKosisApiClient/
extended_claim_type/classify_confidence/_candidate_row/enrich_candidates_with_db/
build_vdb_bm25_fns/load_pilot_articles 헬퍼 등 전부 `import pilot_run as pr`로 재사용).
이번 20건 pilot에서 **새로 추가되는 부분만** 이 파일에 작성한다:

  1. KOSIS 21분 정체 재발 방지(가장 중요, 지난 실패 직접 원인):
     agent/orchestrator/slot_filler.py:49의 `requests.post(URL, headers=HEADERS,
     json=payload)`가 timeout 인자를 전혀 넘기지 않는 걸 발견했다(agent/kosis/api_client.py는
     이미 KOSIS_TIMEOUT=10s 기본값이 있고, agent/preprocessing/hcx_client.py도
     DEFAULT_HARD_TIMEOUT_SECONDS=120s 하드 타임아웃이 있어 이 둘은 원인이 아닐 가능성이
     크다 — grep으로 agent/ 전체의 requests.get/post 호출을 확인, timeout 없는 유일한
     런타임 경로가 slot_filler.py였다). production 파일은 수정하지 않고, 이 스크립트
     프로세스 안에서만 전역 `requests.post`에 "timeout 인자가 없을 때만 기본값을 채우는"
     안전망을 설치한다(다른 모든 호출은 이미 명시적 timeout을 넘기므로 영향 없음).
  2. claim extraction: benchmark/prompt_optimization_experiment/optimized_prompt.txt +
     recovery 조건부 게이트 + max_recovery_rounds=1 + article-level workers=2
     (prompt_optimization_experiment/run_experiment.py의 Config B 오케스트레이션을
     그대로 복제 — 그 스크립트 자체는 계측 코드가 섞여 있어 import하지 않고, B설정 로직만
     복제해서 씀).
  3. Discovery(10)/Evaluation(10) 분리, claim 유형 다양성 고려한 deterministic 20건 선정.
  4. Mapping Reuse Test: Evaluation claim마다 baseline(A) vs mapping-assisted(B) 비교,
     R@1/10/50/100 계산.

절대 원칙(methodology.md 재확인): production DB는 SELECT만, production 코드 파일 수정
금지(전부 import/런타임 monkeypatch만), 로컬 SQLite(data/verifications.db)에 쓰지 않음,
70건 골든셋 gold는 discovery mapping 생성에 사용 안 함(leakage 체크도 이번 20건은 생략 —
methodology.md에 사유 명시), retrieval Top-1 자동 정답 간주 금지.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

# ---------------------------------------------------------------------------
# 0. KOSIS/HCX 정체 방지 안전망 — production import보다 먼저 설치해야 한다.
# ---------------------------------------------------------------------------
import requests as _requests_mod  # noqa: E402

_ORIG_REQUESTS_POST = _requests_mod.post
_SLOT_FILLER_TIMEOUT_SEC = 25  # spec 지시(20~30초) 중간값


def _timeout_guarded_post(*args, **kwargs):
    kwargs.setdefault("timeout", _SLOT_FILLER_TIMEOUT_SEC)
    return _ORIG_REQUESTS_POST(*args, **kwargs)


_requests_mod.post = _timeout_guarded_post

# R@100까지 측정하려면 dense/BM25 채널이 원시 후보를 100개 이상 반환해야 한다(환경변수만
# 바꾼다 — 코드 변경 아님, agent/kosis/query_vdb.py의 VDB_TOP_K/LEXICAL_TOP_K가 각각
# DENSE_TOP_K/BM25_TOP_K 환경변수를 읽는다. 이 두 모듈 상수는 import 시점에 한 번 평가되므로
# 반드시 agent.kosis.query_vdb를 아직 아무도 import하지 않은 지금 시점에 설정해야 한다).
os.environ.setdefault("DENSE_TOP_K", "100")
os.environ.setdefault("BM25_TOP_K", "100")

# 2026-08-30(스모크 테스트로 실측 발견 — 이번 pilot의 또 다른 "무한 대기" 위험):
# SentenceTransformer("Qwen/Qwen3-Embedding-4B", ...)가 모델이 이미 로컬 캐시(7.6GB,
# ~/.cache/huggingface)에 완전히 있는데도, huggingface_hub이 매 파일마다 원격 etag를
# 재검증하려 시도하며 이 sandbox 네트워크 환경에서 550초+ 멈췄다(2건 스모크 테스트로 실측
# 확인 — "Loading weights: 100%"까지는 찍히고 그 다음 줄이 580초 타임아웃까지 안 나옴).
# HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1로 원격 검증 자체를 끄면 캐시만으로 즉시
# 로드된다(같은 조건 재현 테스트: 5.1초). production 파일은 건드리지 않고 이 프로세스
# 환경변수만 설정 — 이미 캐시된 모델을 오프라인으로 쓰는 것뿐이라 결과에 영향 없음(같은
# 가중치, 같은 truncate_dim).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pilot_run as pr  # noqa: E402  (50건 pilot 스크립트 재사용 — production 아님)
from agent.preprocessing.claim_candidate_scanner import find_missed_candidates  # noqa: E402

TOP_K_RETRIEVAL = 100  # R@1/10/50/100 측정을 위해 fused 후보를 최대 100개까지 받는다.

# ---------------------------------------------------------------------------
# 1. 20건 deterministic 선정 (claim 유형 다양성, 단순랜덤 금지)
# ---------------------------------------------------------------------------
# 아래 정규식들은 "선정 단계의 사전 필터"일 뿐이다 — 이 정규식으로 기사를 6개 버킷에
# 미리 배정해서 버킷별 쿼터를 원본 CSV 순서대로(무작위 없이) 채운다. 실제 claim 유형은
# HCX 추출 + extended_claim_type() 실측으로 pilot20_report.md에 별도 보고한다(사전 필터의
# 정확도를 주장하지 않음 — 어디까지나 "다양한 기사를 고르기 위한 결정론적 신호").
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_QUAL_WORDS_RE = re.compile(
    r"(역대\s*(최고|최대|최저|최소)|사상\s*(최고|최대|최저|최소|최다)|역대급|최고치|최저치|"
    r"지속적으로\s*(증가|감소|상승|하락))"
)
_TREND_RE = re.compile(
    r"(\d+)\s*(년|개월|분기)\s*(연속|째)|연속\s*(증가|감소|상승|하락|상승세|하락세)"
)
_THRESHOLD_RE = re.compile(r"\d[\d,\.]*\s*(%|퍼센트|명|건|원|배)?\s*(을|를)?\s*(초과|미만|이상|이하)")
_SUPERLATIVE_RE = re.compile(r"(역대\s*(최고|최대)|사상\s*(최고|최대)|최고치|최대치|최저치|최다)")
_COMPARATIVE_RE = re.compile(r"(전년\s*대비|전월\s*대비|전분기\s*대비|작년\s*보다|지난해\s*보다|보다\s*\d)")
_NUMERIC_RE = re.compile(r"\d[\d,\.]*\s*(%|퍼센트|명|건|원|배|위)")

# (버킷명, 쿼터) — 우선순위 순서(희귀한 유형부터 먼저 채움). 합계 20.
BUCKET_QUOTAS = [
    ("qualitative_no_number", 4),  # 숫자없는 통계주장(스펙이 특별히 주의하라고 명시한 유형)
    ("threshold", 3),
    ("trend", 3),
    ("superlative", 3),
    ("comparative", 4),
    ("numeric", 3),
]


def classify_bucket(article_text: str) -> Optional[str]:
    for s in _SENT_SPLIT_RE.split(article_text):
        if _QUAL_WORDS_RE.search(s) and not re.search(r"\d", s):
            return "qualitative_no_number"
    if _THRESHOLD_RE.search(article_text):
        return "threshold"
    if _TREND_RE.search(article_text):
        return "trend"
    if _SUPERLATIVE_RE.search(article_text):
        return "superlative"
    if _COMPARATIVE_RE.search(article_text):
        return "comparative"
    if _NUMERIC_RE.search(article_text):
        return "numeric"
    return None


def select_pilot20_articles() -> list[dict]:
    """단순 랜덤 셔플(50건 pilot 방식) 대신, 버킷별 쿼터를 원본 CSV 순서대로 채우는
    deterministic 선정 — 무작위 요소가 전혀 없다(같은 data_set.csv면 항상 같은 20건).
    discovery/evaluation split은 버킷 등장 순서 전체를 관통하는 전역 인덱스로 번갈아
    배정해서(버킷 내부가 아니라 선정 순서 전체 기준) 정확히 10/10이 되도록 한다."""
    rows = pr._load_csv_rows(pr.DATA_CSV_PATH)
    for i, r in enumerate(rows):
        r["_row_index"] = i

    quotas = dict(BUCKET_QUOTAS)
    picked_by_bucket: dict[str, list[dict]] = {b: [] for b, _ in BUCKET_QUOTAS}
    for r in rows:
        bucket = classify_bucket(r["기사 본문 전체"])
        if bucket is None or bucket not in quotas:
            continue
        if len(picked_by_bucket[bucket]) >= quotas[bucket]:
            continue
        picked_by_bucket[bucket].append(r)
        if all(len(picked_by_bucket[b]) >= q for b, q in BUCKET_QUOTAS):
            break

    pick_order = []  # 버킷 우선순위 + CSV 순서 (split 배정용 — 전역 인덱스가 여기 순서 기준)
    bucket_labels = {}
    for bucket, _ in BUCKET_QUOTAS:
        for r in picked_by_bucket[bucket]:
            pick_order.append(r)
            bucket_labels[r["_row_index"]] = bucket

    split_assignment = {
        r["_row_index"]: ("discovery" if i % 2 == 0 else "evaluation")
        for i, r in enumerate(pick_order)
    }

    articles = []
    for r in sorted(pick_order, key=lambda r: r["_row_index"]):
        art = pr._row_to_article(r)
        art["article_id"] = f"row{r['_row_index']}"
        art["selection_bucket"] = bucket_labels[r["_row_index"]]
        art["split"] = split_assignment[r["_row_index"]]
        articles.append(art)
    return articles


# ---------------------------------------------------------------------------
# 2. claim extraction — 최적화 프롬프트 + 조건부 recovery(1라운드) + workers=2
#    (prompt_optimization_experiment/run_experiment.py Config B 오케스트레이션 복제)
# ---------------------------------------------------------------------------
OPTIMIZED_PROMPT_PATH = ROOT / "benchmark" / "verified_mapping_experiment" / "optimized_prompt.txt"
OPTIMIZED_PROMPT_TEXT = OPTIMIZED_PROMPT_PATH.read_text(encoding="utf-8")


def install_optimized_extraction():
    ce = pr.claim_extractor_mod
    ce._load_prompt_template = lambda path=None: OPTIMIZED_PROMPT_TEXT


def extract_claims_optimized(article_text: str) -> list:
    ce = pr.claim_extractor_mod
    claims = ce.extract_claims(article_text)
    local_text = ce.dedupe_repeated_sentences(article_text)
    already = [c.sentence for c in claims]
    missed_now = find_missed_candidates(local_text, already)

    should_recover = True
    if not claims and not missed_now:
        should_recover = False
    elif claims and not missed_now:
        should_recover = False

    if should_recover:
        local_claims = claims
        for _ in range(1):  # max_recovery_rounds=1 (B설정, run_experiment.py 실측 근거)
            before = len(local_claims)
            local_claims = ce._recover_missed_claims_once(
                local_text, local_claims, model=ce.MODEL, max_tokens=2048, temperature=0.0
            )
            if len(local_claims) == before:
                break
        claims = ce.attach_prev_sentence(local_text, local_claims)
    else:
        claims = ce.attach_prev_sentence(local_text, claims)
    return claims


def run_extraction_stage(articles: list[dict]) -> dict[str, dict]:
    """1~2단계를 article-level 스레드풀(workers=2)로 병렬 처리(grounding 2번 — 4/8은 429
    에러 다발로 탈락, 2가 실측 최적치). 3단계 이후(retrieval+KOSIS 검증, GPU/DB 접근)는
    원래 pilot_run.py처럼 순차 처리한다."""
    results: dict[str, dict] = {}

    def _work(article):
        cls_result = pr.classifier_mod.classify(article["article_text"])
        if not cls_result.label:
            return article["article_id"], {"relevant": False, "claims": []}
        claims = extract_claims_optimized(article["article_text"])
        claims = pr.claim_extractor_mod.strip_title_prefix_from_claims(claims, article.get("article_title"))
        claims = pr._dedup_claims_by_sentence(claims)
        claims = pr.resolve_claim_sources(claims, cls_result.reason)
        claims = pr.filter_verifiable_claims(claims)
        return article["article_id"], {"relevant": True, "claims": claims}

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="article") as ex:
        futs = {ex.submit(_work, art): art for art in articles}
        for fut in as_completed(futs):
            article = futs[fut]
            try:
                aid, res = fut.result()
            except Exception as e:
                traceback.print_exc()
                aid, res = article["article_id"], {"relevant": False, "claims": [], "error": str(e)}
            results[aid] = res
    return results


# ---------------------------------------------------------------------------
# 3. claim 1건을 특정 table_id 후보에 대해 검증하는 일반화 함수
#    (pilot_run.py::process_claim의 stage4 이후 로직을 "임의 table_id"에 대해 쓸 수 있게
#    일반화한 것 — production run_stage_4/run_stage_5_6/judge 호출은 완전히 동일)
# ---------------------------------------------------------------------------

def verify_table_for_claim(
    article: dict, claim, calc_type: str, table_id: str, table_name: str, org_id: Optional[str],
    table_params: dict, catalog_by_id: dict, client, calculator,
) -> dict:
    base = {"table_id": table_id, "table_name": table_name, "judge_verdict": None,
            "judge_gap_type": None, "judge_reason": None,
            "computed_value": None, "computed_unit": None, "computed_period": None}
    try:
        slots = pr.run_stage_4(
            claim.sentence, article.get("clarify_reply"), article["published_date"],
            table_id=table_id, table_params=table_params,
            catalog_by_id=catalog_by_id, claim_region=claim.region,
        )
    except Exception as e:
        conf, reason = pr.classify_confidence("kosis_fetch_failed", None)
        return {**base, "confidence": conf, "confidence_reason": f"슬롯 채우기 예외: {type(e).__name__}: {e}"}

    if slots is None:
        conf, reason = pr.classify_confidence("slot_fill_incomplete", None)
        return {**base, "confidence": conf, "confidence_reason": reason}

    slots["calc_type"] = calc_type
    selected_itm = pr.select_itm_id(table_id, claim, table_params)
    if selected_itm:
        slots["itm_id"] = selected_itm
    dim_values = pr.select_dimension_values(table_id, claim, table_params, slots)
    if dim_values:
        slots.update(dim_values)

    try:
        computed = pr.run_stage_5_6(
            table_id, slots, table_params, client, calculator,
            comparison_target=claim.comparison_target, claim_sentence=claim.sentence,
            article_year=article["published_date"].year, org_id=org_id, claim=claim,
        )
    except (pr.KosisApiError, pr.CalculationError, KeyError) as e:
        conf, reason = pr.classify_confidence("kosis_fetch_failed", None)
        return {**base, "confidence": conf, "confidence_reason": f"{reason} ({type(e).__name__}: {e})"}
    except Exception as e:
        conf, reason = pr.classify_confidence("kosis_fetch_failed", None)
        return {**base, "confidence": conf,
                "confidence_reason": f"{reason} (예상외 예외 {type(e).__name__}: {e})"}

    if computed is None:
        conf, reason = pr.classify_confidence("kosis_fetch_failed", None)
        return {**base, "confidence": conf, "confidence_reason": reason}

    try:
        verdict = pr.judge(
            claim, computed, prd_se=slots.get("prd_se"),
            article_date=str(article["published_date"]), matched_table_name=table_name,
        )
    except pr.JudgeError as e:
        conf, reason = pr.classify_confidence("kosis_fetch_failed", None)
        return {**base, "confidence": conf,
                "confidence_reason": f"judge() 실패({type(e).__name__}: {e}), 실측값은 조회됨",
                "computed_value": computed.raw_value, "computed_unit": computed.unit,
                "computed_period": computed.period}

    conf, reason = pr.classify_confidence("verified", verdict.verdict)
    return {**base, "confidence": conf, "confidence_reason": reason,
            "judge_verdict": verdict.verdict, "judge_gap_type": verdict.gap_type,
            "judge_reason": verdict.reason,
            "computed_value": computed.raw_value, "computed_unit": computed.unit,
            "computed_period": computed.period}


def _base_claim_record(article: dict, claim, claim_id: str, ext_type: str, verifiable: bool,
                        calc_type: Optional[str]) -> dict:
    return {
        "claim_id": claim_id,
        "article_id": article["article_id"],
        "split": article["split"],
        "selection_bucket": article["selection_bucket"],
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
        "comparison_operator": pr.normalize_comparison_operator(claim, ext_type),
        "comparison_value": claim.comparison_value,
        "region": claim.region,
        "population": claim.population,
        "gender": claim.gender,
        "age": claim.age,
        "organization": claim.source_org,
        "routed_calc_type": calc_type,
    }


# ---------------------------------------------------------------------------
# 4. Discovery claim 처리 — baseline만(재사용 가능한 기존 mapping이 아직 없음), HIGH만 저장
# ---------------------------------------------------------------------------

def process_discovery_claim(article, claim, claim_id, table_params, catalog_by_id, doc_texts,
                             vdb_fn, bm25_fn, client, calculator, db_conn):
    calc_type = pr.route_calc_type(claim)
    ext_type, verifiable = pr.extended_claim_type(claim, calc_type)
    rec = _base_claim_record(article, claim, claim_id, ext_type, verifiable, calc_type)

    if not verifiable:
        conf, reason = pr.classify_confidence("skipped_qualitative", None)
        rec.update(confidence=conf, confidence_reason=reason, mapping_status="not_applicable", num_candidates=0)
        return rec, [], None

    try:
        candidates = pr.search_and_rerank(
            claim, keyword_fn=pr.keyword_search,
            embedding_fn=lambda c: pr.embedding_search(c, cache=doc_texts["_emb_cache"]),
            vdb_fn=vdb_fn, bm25_fn=bm25_fn, top_k=TOP_K_RETRIEVAL,
            document_texts=doc_texts["texts"],
        )
    except Exception as e:
        rec.update(confidence="UNKNOWN", confidence_reason=f"retrieval 실패: {type(e).__name__}: {e}",
                    mapping_status="error", num_candidates=0)
        return rec, [], None

    if not candidates:
        conf, reason = pr.classify_confidence("no_candidate", None)
        rec.update(confidence=conf, confidence_reason=reason,
                    mapping_status="no_mapping_fallback_empty", num_candidates=0)
        return rec, [], None

    cand_rows = [pr._candidate_row(claim_id, i + 1, c) for i, c in enumerate(candidates)]
    pr.enrich_candidates_with_db(cand_rows, db_conn)
    rec["num_candidates"] = len(candidates)
    top = candidates[0]
    rec["top1_table_id"], rec["top1_table_name"] = top.table_id, top.table_name

    if top.source_meta and not pr.is_rrf_trusted(top.source_meta):
        conf, reason = pr.classify_confidence("untrusted_top1", None)
        rec.update(confidence=conf, confidence_reason=reason, mapping_status="no_mapping_fallback_untrusted")
        return rec, cand_rows, None

    if calc_type is None:
        conf, reason = pr.classify_confidence("no_calc_route", None)
        rec.update(confidence=conf, confidence_reason=reason, mapping_status="not_applicable")
        return rec, cand_rows, None

    result = verify_table_for_claim(article, claim, calc_type, top.table_id, top.table_name, top.org_id,
                                     table_params, catalog_by_id, client, calculator)
    rec.update(
        confidence=result["confidence"], confidence_reason=result["confidence_reason"],
        judge_verdict=result["judge_verdict"], judge_gap_type=result["judge_gap_type"],
        judge_reason=result["judge_reason"],
        computed_value=result["computed_value"], computed_unit=result["computed_unit"],
        computed_period=result["computed_period"],
        mapping_status=(
            "independent_verification_hit" if result["confidence"] == "HIGH"
            else ("mapping_conflict" if result["judge_verdict"] == "불일치" else "no_mapping_fallback_inconclusive")
        ),
    )

    mapping_row = None
    if result["confidence"] == "HIGH":
        mapping_row = {
            "claim_id": claim_id,
            "source_article_id": article["article_id"],
            "concept": f"{claim.statistic_expression or claim.population or ext_type}"
                       f"|{claim.region or 'ALL'}|{top.table_id}",
            "claim_text": claim.sentence,
            "table_id": top.table_id,
            "table_name": top.table_name,
            "org_id": top.org_id,
            "metric": claim.statistic_expression,
            "organization": claim.source_org,
            "period": result["computed_period"],
            "region": claim.region,
            "axis": {"gender": claim.gender, "age": claim.age},
            "confidence": "HIGH",
            "mapping_source": "first_time_discovery",
            "independent_verification": True,
            "verification_evidence": {
                "retrieval_evidence": {"top1_source_meta": top.source_meta, "num_candidates": len(candidates)},
                "metadata_evidence": {"organization_claim": claim.source_org, "period_claim": claim.period},
                "numeric_evidence": {"claim_value": claim.value, "computed_value": result["computed_value"],
                                      "unit": result["computed_unit"], "period": result["computed_period"]},
                "judge_reason": result["judge_reason"],
            },
        }
    return rec, cand_rows, mapping_row


# ---------------------------------------------------------------------------
# 5. Mapping 적용 가능성 판정 (새 synonym dictionary 아님 — 기존 retrieval이 이미 이 표를
#    후보로 여겼는지(top-100 안에 있는지)를 게이트로 쓰고, organization은 느슨한 substring
#    비교만 한다. 실제 적용 여부의 최종 판단은 반드시 KOSIS 재조회+judge() 재검증을 통과해야
#    한다 — 이 함수는 "검증해볼 후보를 고르는 것"이지 "정답으로 확정하는 것"이 아니다.)
# ---------------------------------------------------------------------------

def find_applicable_mapping(claim, candidates, discovery_mappings: list[dict]) -> Optional[dict]:
    cand_ids = {c.table_id for c in candidates}
    for m in discovery_mappings:
        if m["table_id"] not in cand_ids:
            continue
        if m.get("organization") and claim.source_org:
            mo, co = m["organization"], claim.source_org
            if mo[:2] not in co and co[:2] not in mo:
                continue
        return m
    return None


# ---------------------------------------------------------------------------
# 6. Evaluation claim 처리 — baseline(A) vs mapping-assisted(B), R@k 계산용 gold 산정
# ---------------------------------------------------------------------------

def process_evaluation_claim(article, claim, claim_id, table_params, catalog_by_id, doc_texts,
                              vdb_fn, bm25_fn, client, calculator, db_conn, discovery_mappings):
    calc_type = pr.route_calc_type(claim)
    ext_type, verifiable = pr.extended_claim_type(claim, calc_type)
    rec = _base_claim_record(article, claim, claim_id, ext_type, verifiable, calc_type)

    if not verifiable:
        rec.update(confidence="UNKNOWN", confidence_reason="claim_type=전망 또는 해외 국가 포함 — 정량 검증 대상 아님",
                    mapping_status="not_applicable", num_candidates=0, ab_status="skipped_not_verifiable")
        return rec, [], None

    try:
        candidates = pr.search_and_rerank(
            claim, keyword_fn=pr.keyword_search,
            embedding_fn=lambda c: pr.embedding_search(c, cache=doc_texts["_emb_cache"]),
            vdb_fn=vdb_fn, bm25_fn=bm25_fn, top_k=TOP_K_RETRIEVAL,
            document_texts=doc_texts["texts"],
        )
    except Exception as e:
        rec.update(confidence="UNKNOWN", confidence_reason=f"retrieval 실패: {type(e).__name__}: {e}",
                    mapping_status="error", num_candidates=0, ab_status="retrieval_error")
        return rec, [], None

    if not candidates:
        rec.update(confidence="UNKNOWN", confidence_reason="retrieval 후보 없음", mapping_status="no_mapping_fallback_empty",
                    num_candidates=0, ab_status="no_candidates")
        return rec, [], None

    cand_rows = [pr._candidate_row(claim_id, i + 1, c) for i, c in enumerate(candidates)]
    pr.enrich_candidates_with_db(cand_rows, db_conn)
    rec["num_candidates"] = len(candidates)

    baseline_top1 = candidates[0]
    rec["baseline_top1_table_id"] = baseline_top1.table_id
    rec["baseline_top1_table_name"] = baseline_top1.table_name

    if calc_type is None:
        rec.update(confidence="UNKNOWN", confidence_reason="calc_type_router가 규칙 기반 라우팅 불가 판정",
                    mapping_status="not_applicable", ab_status="no_calc_route")
        return rec, cand_rows, None

    baseline_trusted = not baseline_top1.source_meta or pr.is_rrf_trusted(baseline_top1.source_meta)
    if baseline_trusted:
        baseline_result = verify_table_for_claim(
            article, claim, calc_type, baseline_top1.table_id, baseline_top1.table_name, baseline_top1.org_id,
            table_params, catalog_by_id, client, calculator,
        )
    else:
        conf, reason = pr.classify_confidence("untrusted_top1", None)
        baseline_result = {"table_id": baseline_top1.table_id, "table_name": baseline_top1.table_name,
                            "confidence": conf, "confidence_reason": reason, "judge_verdict": None,
                            "computed_value": None, "computed_unit": None, "computed_period": None}

    mapping_hit = find_applicable_mapping(claim, candidates, discovery_mappings)
    mapping_applied = False
    mapping_reject_reason = None
    assisted_result = baseline_result
    assisted_table_id = baseline_top1.table_id

    if mapping_hit is not None and mapping_hit["table_id"] == baseline_top1.table_id:
        mapping_applied = True  # mapping이 이미 baseline과 동일한 표를 가리킴(재확인 완료)
    elif mapping_hit is not None and baseline_result["confidence"] == "HIGH":
        # 2026-08-30 회귀 수정(pilot20_report.md §10, row15-c0): baseline이 이미 독립
        # 검증으로 HIGH인데, mapping 후보가 "그럴듯하지만 실제로는 틀린" 다른 표로
        # 재검증까지 HIGH를 받아버려서 assisted/gold를 덮어쓰는 사례가 있었다(R@1 1→0).
        # 원칙: baseline이 이미 HIGH면 mapping 후보를 재검증조차 하지 않는다 — KOSIS 재조회
        # 비용도 아끼고, "정답을 다른 정답 같은 표로 바꿔치기"할 여지 자체를 차단한다.
        mapping_reject_reason = (
            f"baseline이 이미 confidence=HIGH — mapping 후보({mapping_hit['table_id']})는 "
            f"재검증하지 않고 무시(baseline 보호 규칙, 2026-08-30 회귀 수정)"
        )
    elif mapping_hit is not None:
        probe_cand = next(c for c in candidates if c.table_id == mapping_hit["table_id"])
        probe = verify_table_for_claim(
            article, claim, calc_type, probe_cand.table_id, probe_cand.table_name, probe_cand.org_id,
            table_params, catalog_by_id, client, calculator,
        )
        if probe["confidence"] == "HIGH":
            mapping_applied = True
            assisted_result = probe
            assisted_table_id = probe_cand.table_id
        else:
            mapping_reject_reason = (
                f"mapping 후보({mapping_hit['table_id']}) 재검증 confidence={probe['confidence']}"
                f"(judge={probe.get('judge_verdict')}) — 현재 claim의 period/region/실제값과 충돌 또는 "
                f"검증불가로 mapping 무시, baseline 유지"
            )

    # gold 산정: baseline이 HIGH면 baseline이 gold. baseline이 HIGH가 아니고 mapping이 적용돼
    # HIGH로 독립 재검증됐으면 mapping 쪽이 gold(= mapping이 baseline이 놓친 걸 찾아낸 사례).
    # 둘 다 아니면 이 claim은 gold 없음(R@k 분모에서 제외, no_gold로 별도 집계).
    if baseline_result["confidence"] == "HIGH":
        gold_table_id, gold_source = baseline_top1.table_id, "baseline_verified"
    elif mapping_applied and assisted_result["confidence"] == "HIGH" and assisted_table_id != baseline_top1.table_id:
        gold_table_id, gold_source = assisted_table_id, "mapping_verified"
    else:
        gold_table_id, gold_source = None, "unverifiable"

    baseline_ranks = {c.table_id: i + 1 for i, c in enumerate(candidates)}
    if mapping_applied and assisted_table_id != baseline_top1.table_id:
        assisted_order = [assisted_table_id] + [c.table_id for c in candidates if c.table_id != assisted_table_id]
    else:
        assisted_order = [c.table_id for c in candidates]
    assisted_ranks = {tid: i + 1 for i, tid in enumerate(assisted_order)}

    def _recall_at(ranks, k, gold):
        if gold is None:
            return None
        r = ranks.get(gold)
        return 1 if (r is not None and r <= k) else 0

    ks = (1, 10, 50, 100)
    baseline_recall = {k: _recall_at(baseline_ranks, k, gold_table_id) for k in ks}
    assisted_recall = {k: _recall_at(assisted_ranks, k, gold_table_id) for k in ks}

    # 개선/악화 판정 (spec 8번 핵심 질문)
    outcome = "neutral_no_mapping"
    if mapping_hit is None:
        outcome = "no_mapping_applicable"
    elif baseline_result["confidence"] == "HIGH" and not mapping_applied:
        outcome = "mapping_skipped_baseline_protected"
    elif not mapping_applied:
        outcome = "mapping_rejected_conflict"
    elif assisted_table_id == baseline_top1.table_id:
        outcome = "mapping_confirms_baseline"
    # "mapping_overrode_correct_baseline_possible_harm"(구버전): baseline이 HIGH일 때
    # mapping_applied=True로 다른 표를 덮어쓸 수 있던 경로였으나, 위 baseline-보호 규칙으로
    # baseline이 HIGH면 mapping_applied가 True가 될 수 없어졌으므로 이 분기는 도달 불가능
    # 해져서 제거함(2026-08-30).
    elif assisted_result["confidence"] == "HIGH":
        outcome = "mapping_rescued_claim_improvement"
    else:
        outcome = "mapping_applied_no_change"

    rec.update(
        confidence=baseline_result["confidence"], confidence_reason=baseline_result["confidence_reason"],
        judge_verdict=baseline_result.get("judge_verdict"),
        computed_value=baseline_result.get("computed_value"), computed_unit=baseline_result.get("computed_unit"),
        computed_period=baseline_result.get("computed_period"),
        mapping_hit_table_id=(mapping_hit["table_id"] if mapping_hit else None),
        mapping_applied=mapping_applied,
        mapping_reject_reason=mapping_reject_reason,
        assisted_top1_table_id=assisted_table_id,
        assisted_confidence=assisted_result["confidence"],
        assisted_judge_verdict=assisted_result.get("judge_verdict"),
        gold_table_id=gold_table_id, gold_source=gold_source,
        baseline_recall=baseline_recall, assisted_recall=assisted_recall,
        ab_outcome=outcome, ab_status="scored" if gold_table_id else "no_gold",
    )
    verification_pair = {"claim_id": claim_id, "baseline": baseline_result, "assisted": assisted_result,
                          "gold_table_id": gold_table_id, "gold_source": gold_source, "outcome": outcome}
    return rec, cand_rows, verification_pair


# ---------------------------------------------------------------------------
# 7. main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    pr.install_hcx_instrumentation()
    install_optimized_extraction()

    print(f"[설정] _DISABLE_RERANKER = {pr._DISABLE_RERANKER}")
    assert pr._DISABLE_RERANKER is True, "이번 실험은 CE 없는 RRF-only 기준이어야 한다"
    print(f"[설정] slot_filler requests.post 기본 timeout={_SLOT_FILLER_TIMEOUT_SEC}s 안전망 설치 완료")
    print(f"[설정] DENSE_TOP_K={os.environ.get('DENSE_TOP_K')} BM25_TOP_K={os.environ.get('BM25_TOP_K')} "
          f"(R@100 측정용, env-only, 코드 변경 없음)")

    # 스모크 테스트용(선택): PILOT20_SMOKE_LIMIT 환경변수를 주면 discovery/evaluation 각각
    # 앞에서부터 N건만 처리하고 산출물 파일명에 접두사를 붙여 본 실행 결과와 분리한다.
    # 기본(미설정)이면 20건 전체 — 본 실행 동작은 그대로다.
    _smoke_limit = os.environ.get("PILOT20_SMOKE_LIMIT")
    _smoke_limit = int(_smoke_limit) if _smoke_limit else None
    _out_prefix = f"pilot20_smoke{_smoke_limit}_" if _smoke_limit else "pilot20_"

    print("[1/7] 20건 deterministic 선정 (claim 유형 다양성)")
    articles = select_pilot20_articles()
    bucket_counts = {}
    split_counts = {"discovery": 0, "evaluation": 0}
    for a in articles:
        bucket_counts[a["selection_bucket"]] = bucket_counts.get(a["selection_bucket"], 0) + 1
        split_counts[a["split"]] += 1
    print(f"  -> {len(articles)}건, bucket={bucket_counts}, split={split_counts}")
    with open(EXP_DIR / f"{_out_prefix}articles.json", "w", encoding="utf-8") as f:
        json.dump(
            [{"article_id": a["article_id"], "article_title": a["article_title"],
              "article_url": a.get("article_url"), "published_date": str(a["published_date"]),
              "selection_bucket": a["selection_bucket"], "split": a["split"]} for a in articles],
            f, ensure_ascii=False, indent=2,
        )

    print("[2/7] production 리소스 준비 (table_params/catalog/embedding_cache/DB 연결)")
    with open(pr.TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)
    catalog_by_id = pr._load_table_catalog_by_id()
    embedding_cache = pr.build_table_embedding_cache()
    doc_texts_map = pr.load_document_texts()
    doc_texts = {"_emb_cache": embedding_cache, "texts": doc_texts_map}

    db_conn = None
    try:
        import psycopg2
        db_conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
        db_conn.autocommit = True
    except Exception as e:
        print(f"[경고] DB 연결 실패({e}) — 후보 organization/period 보강 없이 진행")

    client = pr.CountingKosisApiClient(timeout=25, retry=1)
    calculator = pr.KosisCalculator()

    print("[3/7] VDB(dense)/BM25 함수 준비 (GPU)")
    vdb_fn, bm25_fn = pr.build_vdb_bm25_fns()

    print("[4/7] claim extraction (article-level workers=2, 최적화 프롬프트)")
    t0 = time.perf_counter()
    extraction_results = run_extraction_stage(articles)
    extraction_elapsed = time.perf_counter() - t0
    n_relevant = sum(1 for v in extraction_results.values() if v.get("relevant"))
    n_claims_extracted = sum(len(v.get("claims", [])) for v in extraction_results.values())
    print(f"  -> {extraction_elapsed:.1f}s, 관련기사 {n_relevant}/{len(articles)}건, "
          f"claim {n_claims_extracted}건 추출")

    claims_out, candidates_out, mappings_out, verification_pairs = [], [], [], []
    discovery_mappings: list[dict] = []
    failed_articles = []

    print("[5/7] Discovery(10건) 처리 — retrieval + 검증 (순차)")
    discovery_articles = [a for a in articles if a["split"] == "discovery"]
    if _smoke_limit:
        discovery_articles = discovery_articles[:_smoke_limit]
        print(f"  [스모크] discovery {len(discovery_articles)}건으로 축소")
    for idx, article in enumerate(discovery_articles):
        res = extraction_results.get(article["article_id"], {})
        print(f"\n--- [D {idx + 1}/{len(discovery_articles)}] {article['article_id']} "
              f"{article['article_title'][:40]!r} relevant={res.get('relevant')}")
        if not res.get("relevant"):
            continue
        for c_idx, claim in enumerate(res["claims"]):
            claim_id = f"{article['article_id']}-c{c_idx}"
            try:
                rec, cand_rows, mapping_row = process_discovery_claim(
                    article, claim, claim_id, table_params, catalog_by_id, doc_texts,
                    vdb_fn, bm25_fn, client, calculator, db_conn,
                )
            except Exception as e:
                traceback.print_exc()
                rec = {"claim_id": claim_id, "article_id": article["article_id"], "split": "discovery",
                       "sentence": claim.sentence, "confidence": "UNKNOWN",
                       "confidence_reason": f"처리 중 예상외 예외: {type(e).__name__}: {e}",
                       "mapping_status": "error"}
                cand_rows, mapping_row = [], None
            claims_out.append(rec)
            candidates_out.extend(cand_rows)
            if mapping_row:
                mappings_out.append(mapping_row)
                discovery_mappings.append(mapping_row)
            print(f"    claim[{c_idx}] confidence={rec.get('confidence')} "
                  f"({str(rec.get('confidence_reason', ''))[:60]})")

    print(f"\n[5/7 완료] Discovery HIGH mapping {len(discovery_mappings)}건 생성")

    print("[6/7] Evaluation(10건) 처리 — baseline(A) vs mapping-assisted(B)")
    evaluation_articles = [a for a in articles if a["split"] == "evaluation"]
    if _smoke_limit:
        evaluation_articles = evaluation_articles[:_smoke_limit]
        print(f"  [스모크] evaluation {len(evaluation_articles)}건으로 축소")
    for idx, article in enumerate(evaluation_articles):
        res = extraction_results.get(article["article_id"], {})
        print(f"\n--- [E {idx + 1}/{len(evaluation_articles)}] {article['article_id']} "
              f"{article['article_title'][:40]!r} relevant={res.get('relevant')}")
        if not res.get("relevant"):
            continue
        for c_idx, claim in enumerate(res["claims"]):
            claim_id = f"{article['article_id']}-c{c_idx}"
            try:
                rec, cand_rows, ver_pair = process_evaluation_claim(
                    article, claim, claim_id, table_params, catalog_by_id, doc_texts,
                    vdb_fn, bm25_fn, client, calculator, db_conn, discovery_mappings,
                )
            except Exception as e:
                traceback.print_exc()
                rec = {"claim_id": claim_id, "article_id": article["article_id"], "split": "evaluation",
                       "sentence": claim.sentence, "confidence": "UNKNOWN",
                       "confidence_reason": f"처리 중 예상외 예외: {type(e).__name__}: {e}",
                       "mapping_status": "error", "ab_status": "error"}
                cand_rows, ver_pair = [], None
            claims_out.append(rec)
            candidates_out.extend(cand_rows)
            if ver_pair:
                verification_pairs.append(ver_pair)
            print(f"    claim[{c_idx}] baseline={rec.get('confidence')} assisted={rec.get('assisted_confidence')} "
                  f"outcome={rec.get('ab_outcome')} gold={rec.get('gold_source')}")

    total_elapsed = time.perf_counter() - t_start

    print("[7/7] 결과 저장")
    with open(EXP_DIR / f"{_out_prefix}claims.jsonl", "w", encoding="utf-8") as f:
        for row in claims_out:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with open(EXP_DIR / f"{_out_prefix}candidates.jsonl", "w", encoding="utf-8") as f:
        for row in candidates_out:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with open(EXP_DIR / f"{_out_prefix}verified_mappings.jsonl", "w", encoding="utf-8") as f:
        for row in mappings_out:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    # --- 집계 ---
    conf_dist, ext_type_dist = {}, {}
    for row in claims_out:
        conf_dist[row.get("confidence", "UNKNOWN")] = conf_dist.get(row.get("confidence", "UNKNOWN"), 0) + 1
        et = row.get("extended_claim_type", "unknown")
        ext_type_dist[et] = ext_type_dist.get(et, 0) + 1

    eval_claims = [r for r in claims_out if r.get("split") == "evaluation" and r.get("ab_status") == "scored"]
    ks = (1, 10, 50, 100)

    def _mean_recall(rows, field, k):
        vals = [r[field][str(k)] if str(k) in r.get(field, {}) else r.get(field, {}).get(k)
                for r in rows]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    baseline_rk = {k: _mean_recall(eval_claims, "baseline_recall", k) for k in ks}
    assisted_rk = {k: _mean_recall(eval_claims, "assisted_recall", k) for k in ks}

    outcome_dist = {}
    for r in claims_out:
        if r.get("split") == "evaluation" and "ab_outcome" in r:
            outcome_dist[r["ab_outcome"]] = outcome_dist.get(r["ab_outcome"], 0) + 1

    # 실패 사례 저장 (spec 9번): confidence가 HIGH가 아닌 verifiable claim 전부
    failure_cases = []
    for r in claims_out:
        if not r.get("verifiable", False):
            continue
        conf = r.get("confidence")
        if conf == "HIGH":
            continue
        failure_cases.append({
            "article_id": r.get("article_id"), "claim_id": r.get("claim_id"),
            "split": r.get("split"), "claim": r.get("sentence"),
            "extended_claim_type": r.get("extended_claim_type"),
            "baseline_top1": r.get("top1_table_id") or r.get("baseline_top1_table_id"),
            "confidence": conf, "confidence_reason": r.get("confidence_reason"),
            "judge_verdict": r.get("judge_verdict"),
            "mapping_hit_table_id": r.get("mapping_hit_table_id"),
            "assisted_top1_table_id": r.get("assisted_top1_table_id"),
            "ab_outcome": r.get("ab_outcome"),
        })
    with open(EXP_DIR / f"{_out_prefix}failure_cases.jsonl", "w", encoding="utf-8") as f:
        for row in failure_cases:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    run_stats = {
        "n_articles_sampled": len(articles),
        "bucket_counts": bucket_counts,
        "split_counts": split_counts,
        "n_articles_relevant": n_relevant,
        "n_claims_extracted": n_claims_extracted,
        "n_claims_total_processed": len(claims_out),
        "extraction_elapsed_sec": round(extraction_elapsed, 1),
        "total_elapsed_sec": round(total_elapsed, 1),
        "confidence_distribution": conf_dist,
        "extended_claim_type_distribution": ext_type_dist,
        "n_discovery_high_mappings": len(discovery_mappings),
        "n_evaluation_claims_scored": len(eval_claims),
        "n_evaluation_claims_no_gold": sum(
            1 for r in claims_out if r.get("split") == "evaluation" and r.get("ab_status") == "no_gold"),
        "baseline_recall_at_k": baseline_rk,
        "assisted_recall_at_k": assisted_rk,
        "evaluation_outcome_distribution": outcome_dist,
        "hcx_calls": pr.HCX_COUNTER.summary(),
        "kosis_calls": pr.KOSIS_COUNTER.summary(),
        "hcx_calls_total": sum(v["calls"] for v in pr.HCX_COUNTER.summary().values()),
        "kosis_calls_total": sum(v["calls"] for v in pr.KOSIS_COUNTER.summary().values()),
        "failed_articles": failed_articles,
    }
    with open(EXP_DIR / f"{_out_prefix}run_stats.json", "w", encoding="utf-8") as f:
        json.dump(run_stats, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== 완료 ===")
    print(json.dumps(run_stats, ensure_ascii=False, indent=2, default=str))

    if db_conn is not None:
        db_conn.close()


if __name__ == "__main__":
    main()
