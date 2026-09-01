"""
benchmark/verified_mapping_819/run819.py
======================================================
Verified Mapping + Retrieval Cascade — 819건 전체 실행
(data/article_priority_institution_mentioned.json 대상, Discovery ~70% / Evaluation ~30%).

20건 pilot(pilot20_run.py, 이미 baseline-보호 회귀 수정 완료)을 "처음부터 다시 만들지
않고" 그대로 import해서 재사용한다(`import pilot20_run as p20`). 이 파일이 새로 추가하는
것은 다음뿐이다:

  1. 819건 로딩: data/article_priority_institution_mentioned.json의 row_index는
     data_set.csv **원본(2706행, 필터링 전)** DictReader 순서를 그대로 가리킨다(실측
     검증: row_index=15 -> "이창용 崔대행..." row_index=17 -> "반도체 덕에...", 전부
     '검색 구분 레이블'=True, url/title 819/819 일치, 2건만 mid-file BOM 잔재로 제목
     문자열이 다름 — _row_to_article은 title로 매칭하지 않고 직접 인덱싱하므로 무관).
     따라서 pilot20/pilot_run.py의 `_load_csv_rows`(필터링된 리스트 기준 인덱스)를 그대로
     쓰면 안 되고, 원본 순서 그대로의 csv.DictReader 결과에서 row_index로 직접 골라
     pr._row_to_article()에 넘긴다. article_id는 pilot20과 동일한 컨벤션인
     f"row{row_index}" (단, 819 대상 파일이 이미 부여한 row_index를 그대로 사용 — 새
     ID 스킴 아님).
  2. Discovery/Evaluation 70/30 deterministic split (seed=SEED_819, pilot20 seed와 별개).
  3. **체크포인트/재개**: discovery+evaluation 합쳐 5개 기사 처리마다 checkpoint.json을
     원자적으로(temp+os.replace) 저장하고, 시작 시 있으면 이어서 처리한다(discovery가
     전부 끝나야 evaluation을 시작하는 phase 경계를 지킴).
  4. A(baseline)/B(mapping-only)/C(mapping-assisted) 사후 계산: 별도 파이프라인을 다시
     돌리지 않고 evaluation.json 집계 단계에서 claims_out 필드로부터 계산한다(정의는
     methodology.md 참고).

절대 원칙은 pilot20_run.py/pilot_run.py와 동일(methodology.md에도 재확인):
production DB는 SELECT만, production 코드 파일은 import만(수정 없음), 로컬 SQLite에
안 씀, retrieval Top-1 자동 정답 간주 금지, baseline이 이미 HIGH면 mapping이 덮어쓰지
않음(pilot20_run.py에 이미 반영됨, 이 파일에서 그 규칙을 건드리지 않음).
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

EXP_DIR = Path(__file__).resolve().parent
PILOT20_DIR = ROOT / "benchmark" / "verified_mapping_experiment"
sys.path.insert(0, str(PILOT20_DIR))  # pilot20_run.py가 pilot_run.py를 상대 import하므로 필요

import pilot20_run as p20  # noqa: E402  (이미 baseline-보호 수정 완료된 20건 pilot 스크립트 재사용)
pr = p20.pr  # pilot_run.py의 production wrapper 모듈(20건 pilot이 이미 import해둔 것 재사용)

DATASET_819_PATH = ROOT / "data" / "article_priority_institution_mentioned.json"
SEED_819 = 20260830819  # pilot20 SEED(20260830)/pilot_run SEED(20260830)와 겹치지 않는 고정 seed
SPLIT_DISCOVERY_RATIO = 0.70
CHECKPOINT_EVERY_N_ARTICLES = 5
# extraction(ThreadPoolExecutor workers=2)을 이 크기 단위 배치로 나눠 실행 — 573/246건을
# 한 번에 batch extraction하면 첫 체크포인트까지 수십 분이 걸리고 크래시 시 그만큼
# 손실되므로, 체크포인트 주기(5건)에 맞춰 작은 배치로 쪼갠다(스레드풀 효율을 위해
# 체크포인트 주기의 정수배로 약간 더 크게 잡음).
EXTRACTION_BATCH_SIZE = 10

CHECKPOINT_PATH = EXP_DIR / "checkpoint.json"
RUN_STATUS_PATH = EXP_DIR / "run_status.json"
CLAIMS_JSONL_PATH = EXP_DIR / "claims.jsonl"
CANDIDATES_JSONL_PATH = EXP_DIR / "candidates.jsonl"
MAPPINGS_JSONL_PATH = EXP_DIR / "verified_mappings.jsonl"
EVALUATION_JSON_PATH = EXP_DIR / "evaluation.json"
RESULTS_CSV_PATH = EXP_DIR / "results.csv"
FAILURE_ANALYSIS_PATH = EXP_DIR / "failure_analysis.md"

# 스모크 테스트 스위치(선택) — pilot20_run.py의 PILOT20_SMOKE_LIMIT과 같은 패턴.
# RUN819_SMOKE=1이면 discovery 앞 RUN819_SMOKE_DISCOVERY(기본 2)건, evaluation 앞
# RUN819_SMOKE_EVAL(기본 1)건만 처리하고, 산출물 전부를 별도 smoke_ 접두사 경로에 써서
# 본 실행(체크포인트 포함) 상태와 절대 섞이지 않게 한다.
_SMOKE = os.environ.get("RUN819_SMOKE") == "1"
if _SMOKE:
    _SMOKE_DISCOVERY_LIMIT = int(os.environ.get("RUN819_SMOKE_DISCOVERY", "2"))
    _SMOKE_EVAL_LIMIT = int(os.environ.get("RUN819_SMOKE_EVAL", "1"))
    CHECKPOINT_PATH = EXP_DIR / "smoke_checkpoint.json"
    RUN_STATUS_PATH = EXP_DIR / "smoke_run_status.json"
    CLAIMS_JSONL_PATH = EXP_DIR / "smoke_claims.jsonl"
    CANDIDATES_JSONL_PATH = EXP_DIR / "smoke_candidates.jsonl"
    MAPPINGS_JSONL_PATH = EXP_DIR / "smoke_verified_mappings.jsonl"
    EVALUATION_JSON_PATH = EXP_DIR / "smoke_evaluation.json"
    RESULTS_CSV_PATH = EXP_DIR / "smoke_results.csv"
    FAILURE_ANALYSIS_PATH = EXP_DIR / "smoke_failure_analysis.md"


# ---------------------------------------------------------------------------
# 1. 819건 로딩 — row_index는 원본(필터링 전) CSV DictReader 순서 그대로.
# ---------------------------------------------------------------------------

def _load_raw_csv_rows(path: Path = pr.DATA_CSV_PATH) -> list[dict]:
    """pilot_run.py::_load_csv_rows와 동일한 BOM 정리 로직이지만 '검색 구분 레이블'
    필터를 걸지 않는다(원본 순서 보존 목적, 실측 검증: article_priority_institution_
    mentioned.json의 row_index가 바로 이 필터링 전 순서를 가리킴)."""
    with open(path, encoding="utf-8-sig") as f:
        text = f.read().replace("﻿", "")
    return list(csv.DictReader(io.StringIO(text)))


def load_819_articles() -> list[dict]:
    with open(DATASET_819_PATH, encoding="utf-8") as f:
        targets = json.load(f)
    raw_rows = _load_raw_csv_rows()
    articles = []
    for entry in targets:
        ri = entry["row_index"]
        row = raw_rows[ri]
        art = pr._row_to_article(row)
        art["article_id"] = f"row{ri}"
        art["selection_bucket"] = None  # 819 규모에서는 버킷 다양성 선정을 하지 않음(over-engineering, spec 명시)
        art["matched_institutions"] = entry.get("matched_institutions")
        articles.append(art)
    return articles


def split_819_articles(articles: list[dict]) -> tuple[list[dict], list[dict]]:
    """deterministic 70/30 split. 정렬 순서(발행일 등)와 상관되지 않도록 셔플을 먼저 하고
    앞 70%를 discovery, 뒤 30%를 evaluation으로 배정한다."""
    shuffled = list(articles)
    random.Random(SEED_819).shuffle(shuffled)
    n_discovery = round(len(shuffled) * SPLIT_DISCOVERY_RATIO)
    discovery_ids = {a["article_id"] for a in shuffled[:n_discovery]}
    for a in articles:
        a["split"] = "discovery" if a["article_id"] in discovery_ids else "evaluation"
    discovery = [a for a in articles if a["split"] == "discovery"]
    evaluation = [a for a in articles if a["split"] == "evaluation"]
    return discovery, evaluation


# ---------------------------------------------------------------------------
# 2. 체크포인트 (원자적 쓰기: 임시파일 -> os.replace)
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, obj, *, compact: bool = False) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        if compact:
            # checkpoint.json은 819건 끝무렵 candidates_out만 수십만 행(claim당 top-100)에
            # 달할 수 있어 5개 기사마다 통째로 다시 쓴다 — indent=2는 디스크/CPU 낭비가
            # 크므로 이 파일만 압축 포맷으로 쓴다(사람이 읽는 용도가 아니라 재개용 상태
            # 저장소이므로 가독성보다 크기/속도 우선).
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"), default=str)
        else:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def load_checkpoint() -> Optional[dict]:
    if not CHECKPOINT_PATH.exists():
        return None
    with open(CHECKPOINT_PATH, encoding="utf-8") as f:
        return json.load(f)


def fresh_state() -> dict:
    return {
        "completed_article_ids": [],
        "claims_out": [],
        "candidates_out": [],
        "mappings_out": [],
        "verification_pairs": [],
        "discovery_mappings": [],
        "current_phase": "discovery",
        "last_updated": None,
    }


def save_checkpoint(state: dict, n_total: int, n_discovery: int, n_evaluation: int,
                     t_start: float, process_started_at_iso: str) -> None:
    now = datetime.now(timezone.utc)
    state["last_updated"] = now.isoformat()
    _atomic_write_json(CHECKPOINT_PATH, state, compact=True)
    _atomic_write_jsonl(CLAIMS_JSONL_PATH, state["claims_out"])
    _atomic_write_jsonl(CANDIDATES_JSONL_PATH, state["candidates_out"])
    _atomic_write_jsonl(MAPPINGS_JSONL_PATH, state["mappings_out"])

    n_done = len(state["completed_article_ids"])
    elapsed = time.perf_counter() - t_start  # 이번 프로세스 인스턴스 기준(재시작 시 0부터 다시 잼)
    rate = elapsed / n_done if n_done else None  # 이번 프로세스 인스턴스에서 실측한 건당 평균 초
    eta_sec = rate * (n_total - n_done) if rate else None
    status = {
        "process_started_at": process_started_at_iso,
        "last_updated": state["last_updated"],
        "current_phase": state["current_phase"],
        "n_articles_total": n_total,
        "n_articles_discovery": n_discovery,
        "n_articles_evaluation": n_evaluation,
        "n_articles_completed": n_done,
        "n_claims_so_far": len(state["claims_out"]),
        "n_discovery_high_mappings_so_far": len(state["discovery_mappings"]),
        "process_elapsed_sec": round(elapsed, 1),
        "sec_per_article_observed_this_process": round(rate, 2) if rate else None,
        "eta_seconds_remaining_at_this_rate": round(eta_sec, 0) if eta_sec else None,
        "eta_completion_at_this_rate": (
            datetime.fromtimestamp(now.timestamp() + eta_sec, tz=timezone.utc).isoformat()
            if eta_sec else None
        ),
    }
    _atomic_write_json(RUN_STATUS_PATH, status)


# ---------------------------------------------------------------------------
# 3. main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()
    process_started_at_iso = datetime.now(timezone.utc).isoformat()
    pr.install_hcx_instrumentation()
    p20.install_optimized_extraction()

    print(f"[설정] _DISABLE_RERANKER = {pr._DISABLE_RERANKER}")
    assert pr._DISABLE_RERANKER is True, "이번 실험은 CE 없는 RRF-only 기준이어야 한다"
    print(f"[설정] DENSE_TOP_K={os.environ.get('DENSE_TOP_K')} BM25_TOP_K={os.environ.get('BM25_TOP_K')}")
    if _SMOKE:
        print(f"[설정] SMOKE 모드: discovery<= {_SMOKE_DISCOVERY_LIMIT}건, evaluation<= {_SMOKE_EVAL_LIMIT}건, "
              f"체크포인트/산출물 전부 smoke_ 접두사 분리")

    print("[1/6] 819건 로딩 + deterministic 70/30 split")
    all_articles = load_819_articles()
    discovery_articles, evaluation_articles = split_819_articles(all_articles)
    if _SMOKE:
        discovery_articles = discovery_articles[:_SMOKE_DISCOVERY_LIMIT]
        evaluation_articles = evaluation_articles[:_SMOKE_EVAL_LIMIT]
    print(f"  -> 전체 {len(all_articles)}건, discovery {len(discovery_articles)}건, "
          f"evaluation {len(evaluation_articles)}건 (seed={SEED_819})")
    articles_by_id = {a["article_id"]: a for a in (discovery_articles + evaluation_articles)}
    n_total = len(discovery_articles) + len(evaluation_articles)

    print("[2/6] 체크포인트 확인")
    state = load_checkpoint()
    if state is None:
        state = fresh_state()
        print("  -> 체크포인트 없음, 처음부터 시작")
    else:
        print(f"  -> 체크포인트 발견: phase={state['current_phase']}, "
              f"완료 {len(state['completed_article_ids'])}건, "
              f"HIGH mapping {len(state['discovery_mappings'])}건, last_updated={state['last_updated']}")

    print("[3/6] production 리소스 준비 (table_params/catalog/embedding_cache/DB 연결)")
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

    print("[4/6] VDB(dense)/BM25 함수 준비 (GPU)")
    vdb_fn, bm25_fn = pr.build_vdb_bm25_fns()

    completed = set(state["completed_article_ids"])
    checkpoint_counter = 0  # 마지막 체크포인트 이후 새로 완료된 기사 수(discovery+evaluation 합산)

    def _maybe_checkpoint(force: bool = False):
        nonlocal checkpoint_counter
        if force or checkpoint_counter >= CHECKPOINT_EVERY_N_ARTICLES:
            save_checkpoint(state, n_total, len(discovery_articles), len(evaluation_articles),
                             t_start, process_started_at_iso)
            checkpoint_counter = 0
            n_done = len(state["completed_article_ids"])
            print(f"[체크포인트] phase={state['current_phase']} {n_done}/{n_total}, "
                  f"HIGH mapping {len(state['discovery_mappings'])}건, "
                  f"claims {len(state['claims_out'])}건")

    def _process_article(article: dict, extraction_result: dict, phase: str):
        """claim 단위 처리 + 결과 누적. article은 완료 시 completed에 추가."""
        nonlocal checkpoint_counter
        aid = article["article_id"]
        if not extraction_result.get("relevant"):
            completed.add(aid)
            state["completed_article_ids"] = sorted(completed)
            checkpoint_counter += 1
            return
        for c_idx, claim in enumerate(extraction_result["claims"]):
            claim_id = f"{aid}-c{c_idx}"
            try:
                if phase == "discovery":
                    rec, cand_rows, mapping_row = p20.process_discovery_claim(
                        article, claim, claim_id, table_params, catalog_by_id, doc_texts,
                        vdb_fn, bm25_fn, client, calculator, db_conn,
                    )
                    ver_pair = None
                else:
                    rec, cand_rows, ver_pair = p20.process_evaluation_claim(
                        article, claim, claim_id, table_params, catalog_by_id, doc_texts,
                        vdb_fn, bm25_fn, client, calculator, db_conn, state["discovery_mappings"],
                    )
                    mapping_row = None
            except Exception as e:
                traceback.print_exc()
                rec = {"claim_id": claim_id, "article_id": aid, "split": phase,
                       "sentence": claim.sentence, "confidence": "UNKNOWN",
                       "confidence_reason": f"처리 중 예상외 예외: {type(e).__name__}: {e}",
                       "mapping_status": "error", "ab_status": "error" if phase == "evaluation" else None}
                cand_rows, mapping_row, ver_pair = [], None, None
            state["claims_out"].append(rec)
            state["candidates_out"].extend(cand_rows)
            if mapping_row:
                state["mappings_out"].append(mapping_row)
                state["discovery_mappings"].append(mapping_row)
            if ver_pair:
                state["verification_pairs"].append(ver_pair)
        completed.add(aid)
        state["completed_article_ids"] = sorted(completed)
        checkpoint_counter += 1

    def _run_phase(phase_articles: list[dict], phase: str, step_label: str):
        """extraction을 EXTRACTION_BATCH_SIZE건씩 작은 배치로 나눠 실행한다(573/246건을
        한 번에 batch extraction하면 첫 체크포인트가 나오기까지 수십 분이 걸리고, 그 사이
        크래시하면 이미 끝낸 extraction이 통째로 날아간다 — 스모크 테스트 이후 실측
        발견, 5건 체크포인트 취지에 맞게 추출도 잘게 쪼갠다)."""
        remaining = [a for a in phase_articles if a["article_id"] not in completed]
        print(f"  -> 남은 {phase} 기사 {len(remaining)}/{len(phase_articles)}건")
        for batch_start in range(0, len(remaining), EXTRACTION_BATCH_SIZE):
            batch = remaining[batch_start:batch_start + EXTRACTION_BATCH_SIZE]
            print(f"  -> claim extraction 배치 실행 중... ({batch_start + 1}~{batch_start + len(batch)}/{len(remaining)})")
            extraction_results = p20.run_extraction_stage(batch)
            for article in batch:
                res = extraction_results.get(article["article_id"], {"relevant": False, "claims": []})
                _process_article(article, res, phase)
                _maybe_checkpoint()
        _maybe_checkpoint(force=True)

    print("[5/6] Discovery 처리 (resume-aware, article-level extraction workers=2)")
    if state["current_phase"] == "discovery":
        _run_phase(discovery_articles, "discovery", "5/6")
        state["current_phase"] = "evaluation"
        save_checkpoint(state, n_total, len(discovery_articles), len(evaluation_articles),
                         t_start, process_started_at_iso)
        print(f"[5/6 완료] Discovery 전체 완료, HIGH mapping {len(state['discovery_mappings'])}건 -> phase=evaluation")
    else:
        print(f"  -> 이미 discovery 완료됨(phase={state['current_phase']}), 스킵")

    print("[6/6] Evaluation 처리 (resume-aware) — baseline(A) vs mapping-assisted(C)")
    if state["current_phase"] == "evaluation":
        _run_phase(evaluation_articles, "evaluation", "6/6")
        state["current_phase"] = "done"
        save_checkpoint(state, n_total, len(discovery_articles), len(evaluation_articles),
                         t_start, process_started_at_iso)
        print("[6/6 완료] Evaluation 전체 완료 -> phase=done")
    elif state["current_phase"] == "done":
        print("  -> 이미 완료됨(phase=done), 집계만 재실행")
    else:
        print(f"  -> 아직 discovery 단계 미완료(phase={state['current_phase']}) — 비정상 상태, 스킵")

    if db_conn is not None:
        db_conn.close()

    if state["current_phase"] != "done":
        print(f"[중단] phase={state['current_phase']}로 종료 — 재실행 시 체크포인트에서 이어서 처리됩니다.")
        return

    print("\n=== 819건 전체 완료 — 최종 집계 ===")
    write_final_outputs(state, articles_by_id, t_start)


# ---------------------------------------------------------------------------
# 4. 최종 집계 (A/B/C 사후 계산, evaluation.json/results.csv/failure_analysis.md)
#    정의는 methodology.md와 동일 문구:
#      A(baseline)      = claims_out[*].baseline_recall
#      C(assisted)      = claims_out[*].assisted_recall (baseline-보호 규칙 반영됨)
#      B(mapping-only)  = mapping_hit_table_id가 존재하는 evaluation claim 부분집합에서의
#                         assisted_recall (mapping_hit이 없는 claim은 B의 분모에서 제외)
# ---------------------------------------------------------------------------

def write_final_outputs(state: dict, articles_by_id: dict, t_start: float) -> None:
    claims_out = state["claims_out"]
    ks = (1, 10, 50, 100)

    def _mean_recall(rows, field, k):
        vals = [r.get(field, {}).get(str(k), r.get(field, {}).get(k)) for r in rows]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    eval_scored = [r for r in claims_out if r.get("split") == "evaluation" and r.get("ab_status") == "scored"]
    eval_with_mapping_hit = [r for r in eval_scored if r.get("mapping_hit_table_id")]

    recall_A = {k: _mean_recall(eval_scored, "baseline_recall", k) for k in ks}
    recall_C = {k: _mean_recall(eval_scored, "assisted_recall", k) for k in ks}
    recall_B = {k: _mean_recall(eval_with_mapping_hit, "assisted_recall", k) for k in ks}

    conf_dist, ext_type_dist, outcome_dist = {}, {}, {}
    for r in claims_out:
        conf_dist[r.get("confidence", "UNKNOWN")] = conf_dist.get(r.get("confidence", "UNKNOWN"), 0) + 1
        et = r.get("extended_claim_type", "unknown")
        ext_type_dist[et] = ext_type_dist.get(et, 0) + 1
        if r.get("split") == "evaluation" and "ab_outcome" in r:
            outcome_dist[r["ab_outcome"]] = outcome_dist.get(r["ab_outcome"], 0) + 1

    evaluation_summary = {
        "n_articles_total": len(articles_by_id),
        "n_claims_total": len(claims_out),
        "n_evaluation_claims_scored": len(eval_scored),
        "n_evaluation_claims_with_mapping_hit": len(eval_with_mapping_hit),
        "n_evaluation_claims_no_gold": sum(
            1 for r in claims_out if r.get("split") == "evaluation" and r.get("ab_status") == "no_gold"),
        "confidence_distribution": conf_dist,
        "extended_claim_type_distribution": ext_type_dist,
        "evaluation_outcome_distribution": outcome_dist,
        "recall_A_baseline_at_k": recall_A,
        "recall_B_mapping_only_at_k": recall_B,
        "recall_C_mapping_assisted_at_k": recall_C,
        "n_discovery_high_mappings": len(state["discovery_mappings"]),
        "hcx_calls_total": sum(v["calls"] for v in pr.HCX_COUNTER.summary().values()),
        "kosis_calls_total": sum(v["calls"] for v in pr.KOSIS_COUNTER.summary().values()),
        "total_elapsed_sec": round(time.perf_counter() - t_start, 1),
    }
    _atomic_write_json(EVALUATION_JSON_PATH, evaluation_summary)

    import csv as _csv
    fields = ["claim_id", "article_id", "split", "extended_claim_type", "confidence",
              "baseline_top1_table_id", "assisted_top1_table_id", "mapping_hit_table_id",
              "mapping_applied", "gold_table_id", "gold_source", "ab_outcome", "ab_status"]
    with open(RESULTS_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in claims_out:
            w.writerow(r)

    failure_rows = [
        r for r in claims_out
        if r.get("verifiable", False) and r.get("confidence") != "HIGH"
    ]
    lines = [
        "# failure_analysis.md — 819건 Verified Mapping 실험 실패/UNKNOWN 사례 정성 분석",
        "",
        f"검증 가능(verifiable=True)했지만 confidence != HIGH인 claim: {len(failure_rows)}건 "
        f"(전체 claim {len(claims_out)}건 중).",
        "",
        "## confidence별 건수",
    ]
    fail_conf_dist: dict[str, int] = {}
    for r in failure_rows:
        fail_conf_dist[r.get("confidence", "UNKNOWN")] = fail_conf_dist.get(r.get("confidence", "UNKNOWN"), 0) + 1
    for k, v in sorted(fail_conf_dist.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}건")
    lines.append("")
    lines.append("## 사례 샘플 (최대 30건, confidence_reason별 대표)")
    seen_reasons: set[str] = set()
    for r in failure_rows:
        reason = (r.get("confidence_reason") or "")[:50]
        if reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        lines.append(
            f"- [{r.get('claim_id')}] confidence={r.get('confidence')} "
            f"reason={r.get('confidence_reason')!r} sentence={str(r.get('sentence'))[:80]!r}"
        )
        if len(seen_reasons) >= 30:
            break
    FAILURE_ANALYSIS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(evaluation_summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
