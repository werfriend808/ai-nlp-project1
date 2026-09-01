"""3단계(표 매칭) 값 기준 재평가 — 팀원 제안 1번.

tblId 문자열 완전일치 대신, top-K 후보 표들의 실제 KOSIS 조회값이 골든셋 kosis_value와
맞는지로 MISS를 재채점한다. 근접중복/버전분화(원인 C)로 인한 "가짜 미스"를 걸러내서
진짜 표매칭 성능이 얼마나 되는지 드러내는 게 목적 — 기존 stage3_verify_report.md의
MISS 47건(고유 34건) 중 rank=None인 것만 대상으로 한다.

방법: 각 후보 표에 대해 _build_dynamic_kosis_slots()로 슬롯을 만들어 KOSIS를 직접
조회한다. golden claim_value/kosis_value(실측) 중 하나와 허용오차 내로 맞으면 "값 일치"로
본다 — 팀원이 건설업 사례를 수작업으로 확인한 것과 같은 원리를 자동화한 것.

v2(2026-08-23, 값 재검증 요청으로 실행하며 발견/수정): v1은 두 가지 이유로 실질적인
검증을 거의 못 하고 있었다.
1) period 포맷 버그 — kosis_period를 str()만 해서(예: "2024-04-01 00:00:00") period
   슬롯에 그대로 넣었는데, KOSIS는 표 주기(Y/M/Q)에 맞는 자릿수 문자열만 받는다
   (api_client._validate_period_format 참고). 포맷이 안 맞으면 요청 자체가 매번
   실패해서(이번 재실행에서 KOSIS 조회 실패 157건) 대부분의 후보가 애초에 조회조차
   안 되고 있었다. _format_period()로 표의 prd_se를 먼저 확인해 자릿수를 맞춘다.
2) 두 시점 계산 누락 — "15만명 급감" 같은 증감형 claim은 golden claim_value/
   kosis_value(실측)에 변화량(예: 150000)이 기록되는데, v1은 한 시점(그것도 현재가
   아니라 kosis_period=비교기준 시점) 원본값만 조회해서 비교했다. 원본 레벨값(수백만
   단위)과 변화량은 스케일 자체가 달라 절대 못 맞는다. 이제 claim의 현재 시점(period)과
   기준 시점(kosis_period) 둘 다 조회해서, 원본값 매칭에 이어 두 값의 차이(diff)도
   golden과 비교한다.

사용법: python -m notebooks.verify_stage3_value_reeval
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2

from agent.interfaces import Claim
from agent.kosis.api_client import KosisApiClient, KosisApiError
from agent.kosis.calculator import KosisCalculator, CalculationError
from agent.kosis.detail_cache import get_table_detail, DetailCacheUnavailableError
from agent.kosis.query_vdb import batch_query_vdb, lexical_query_vdb, VDB_TOP_K, LEXICAL_TOP_K
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.keyword_search import keyword_search
from agent.mapping.reranker import search_and_rerank, expand_institution_query_aliases
from agent.pipeline.batch_runner import _load_table_catalog_by_id, _build_dynamic_kosis_slots

GOLDEN_PATH = Path(__file__).parent / "골든셋_통합.xlsx"
REPORT_PATH = Path(__file__).parent / "stage3_value_reeval_report.md"
TOP_K = 10
CANDIDATES_TO_CHECK = 5  # 비용 상한 — top-K 중 상위 5개만 실제 KOSIS 조회 시도

_ORG_NORMALIZE = {"통계청": "국가데이터처", "KOSIS": None}


def _build_search_query(row: pd.Series) -> str:
    stat_parts = []
    stat_expr = row.get("statistic_expression")
    if isinstance(stat_expr, str) and stat_expr.strip():
        stat_parts.append(stat_expr.strip())
    for col in ("population", "region"):
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            stat_parts.append(val.strip())
    org_parts = []
    org = row.get("source_org")
    if isinstance(org, str) and org.strip():
        normalized = _ORG_NORMALIZE.get(org.strip(), org.strip())
        if normalized:
            org_parts.append(normalized)
    parts = stat_parts + org_parts
    return " ".join(parts) if parts else str(row.get("sentence(원문 그대로)", ""))


def _extract_numbers(text) -> list[float]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    nums = re.findall(r"-?\d+\.?\d*", str(text).replace(",", ""))
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except ValueError:
            pass
    return out


# 2026-08-23 실측 발견: 건설업 취업자 케이스(gold=DT_1DA7E26S, golden=150000)를 1위 후보
# DT_1DA7E06S_NEW로 직접 조회해보니 현재/기준 시점 raw_value가 각각 1947.5/2097.6이고
# 단위가 "천명"이었다 — 차이(150.1)는 사실상 골든셋 값(150,000명)과 거의 정확히 일치하는데
# (150.1 x 1000 = 150,100), 이 스크립트가 raw_value를 단위 그대로(천명 단위 숫자) 골든셋의
# 원 단위(명) 숫자와 비교하고 있어서 "안 맞다"고 오판했다 — 지금까지 "가짜 미스 0건"으로
# 나온 게 실은 검색/조회가 다 맞았는데 이 단위 환산 누락 때문에 전부 놓치고 있었을 가능성이
# 크다. KosisApiResponse.unit에 실제 단위가 있으니 비교 전에 여기서 배율을 곱해준다.
_UNIT_SCALE = {
    "천명": 1_000, "천원": 1_000, "천달러": 1_000, "천호": 1_000, "천가구": 1_000, "천세대": 1_000,
    "백만원": 1_000_000, "백만달러": 1_000_000,
    "억원": 100_000_000, "억달러": 100_000_000,
    "조원": 1_000_000_000_000,
}


def _scaled_value(raw_value: Optional[float], unit: Optional[str]) -> Optional[float]:
    if raw_value is None:
        return None
    return raw_value * _UNIT_SCALE.get(unit or "", 1)


def _values_match(fetched: float, golden_candidates: list[float], rel_tol: float = 0.02) -> bool:
    """fetched 값이 golden_candidates(claim_value/comparison_value/kosis_value) 중
    하나와 상대오차 2% 이내로 맞으면 True. 절대오차 0.05도 같이 허용(0에 가까운 값 대비)."""
    for g in golden_candidates:
        if g == 0:
            if abs(fetched) < 0.05:
                return True
            continue
        if abs(fetched - g) / abs(g) <= rel_tol:
            return True
    return False


def _format_period(date_val, prd_se: Optional[str]) -> Optional[str]:
    """pandas 시점 값(Timestamp 등) -> KOSIS가 받는 시점 문자열.

    표의 prd_se(연=Y/월=M/분기=Q)에 맞춰 자릿수를 맞춘다(api_client._validate_period_format
    참고) — 이 필드를 str()만 해서 그대로 넘기면(예: "2024-04-01 00:00:00") 자릿수 검증에서
    매번 거부돼 조회 자체가 안 된다(2026-08-23 발견). prd_se가 Y/M/Q가 아니면(F=격년,
    D=일단위 등) 아직 규칙이 없어 스킵(None) — 조용히 틀린 값을 넘기지 않는다."""
    if date_val is None or (isinstance(date_val, float) and pd.isna(date_val)):
        return None
    try:
        ts = pd.Timestamp(date_val)
    except (ValueError, TypeError):
        return None
    if prd_se == "Y":
        return f"{ts.year}"
    if prd_se == "M":
        return f"{ts.year}{ts.month:02d}"
    if prd_se == "Q":
        q = (ts.month - 1) // 3 + 1
        return f"{ts.year}{q:02d}"
    return None


def main() -> None:
    df7 = pd.read_excel(GOLDEN_PATH, sheet_name="7단계_판정목록")
    df2 = pd.read_excel(GOLDEN_PATH, sheet_name="2단계_claim목록")
    df7 = df7.merge(
        df2[["claim_id", "statistic_expression", "population", "region", "source_org"]],
        on="claim_id", how="left",
    )
    ids_stripped = df7["matched_table_id(3단계)"].astype(str).str.strip()
    evalable = df7[ids_stripped != "없음"].reset_index(drop=True)

    catalog_by_id = _load_table_catalog_by_id()
    document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

    # 2026-08-23 발견: keyword_search/embedding_search(64개 카탈로그 경로)가 만드는
    # TableCandidate는 org_id를 아예 안 채운다(agent/mapping/keyword_search.py,
    # embedding_search.py 둘 다 org_id 관련 코드 자체가 없음) — VDB 경로(query_vdb.py)만
    # org_id를 채운다. 그래서 이 스크립트 첫 실행에서 235건이 전부 "org_id 없어 스킵"으로
    # 날아갔다(_build_dynamic_kosis_slots가 org_id 없으면 바로 포기). catalog_by_id에
    # orgId가 있으면 그걸 먼저 쓰고, 없으면(카탈로그 밖 VDB 전용 표) DB에서 직접 채운다.
    _org_id_conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    _org_id_cache: dict[str, Optional[str]] = {}

    def _resolve_org_id(table_id: str) -> Optional[str]:
        if table_id in _org_id_cache:
            return _org_id_cache[table_id]
        org_id = None
        cat_entry = catalog_by_id.get(table_id)
        if cat_entry and cat_entry.get("orgId"):
            org_id = cat_entry["orgId"]
        else:
            cur = _org_id_conn.cursor()
            cur.execute("select org_id from kosis_vdb_tables where tbl_id = %s", (table_id,))
            row = cur.fetchone()
            cur.close()
            if row:
                org_id = row[0]
        _org_id_cache[table_id] = org_id
        return org_id

    from sentence_transformers import SentenceTransformer

    print("Qwen3-Embedding-4B 로딩 중...")
    vdb_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=1024)
    vdb_instruction = (
        "Given a Korean news claim sentence, retrieve the KOSIS statistical table "
        "description that best matches it"
    )

    def _retrieval_query_text(claim) -> str:
        base = claim.search_query or claim.sentence
        return expand_institution_query_aliases(base, claim.source_org)

    def vdb_fn(claim):
        text = f"Instruct: {vdb_instruction}\nQuery: {_retrieval_query_text(claim)}"
        vec = vdb_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        return batch_query_vdb([vec], top_k=VDB_TOP_K)[0]

    def bm25_fn(claim):
        return lexical_query_vdb(_retrieval_query_text(claim), top_k=LEXICAL_TOP_K)

    client = KosisApiClient()
    calculator = KosisCalculator()

    from collections import Counter

    fake_miss = []  # (sentence, gold_id, matched_candidate_id, fetched_value, golden_value)
    true_miss = []
    already_ok = 0
    skipped_no_org = 0
    fetch_errors = 0
    error_categories: Counter = Counter()  # 임시 진단용(2026-08-23) — 에러 유형 분포 확인

    report_lines = ["# 3단계 값 기준 재평가 결과\n\n"]

    for i, row in evalable.iterrows():
        gold_id = str(row["matched_table_id(3단계)"]).strip()
        sentence = str(row["sentence(원문 그대로)"])
        gold_ids = [g.strip() for g in gold_id.split(",")]

        org = row.get("source_org")
        period_val = row.get("period")
        # 2026-08-23 실측 발견: population/statistic_expression/region을 안 채우고
        # 있었다 — _build_dynamic_kosis_slots() 내부의 map_claim_slots()가 이 필드들로
        # "건설업"/"제조업"처럼 표의 구체적인 분류축 코드를 골라내는데, 비어있으면 어느
        # 항목을 조회해야 할지 못 정해서 엉뚱한 값(또는 ALL 집계)을 가져온다. 실측 재현:
        # 건설업 취업자 케이스(gold=DT_1DA7E26S)를 1위 후보 DT_1DA7E06S_NEW에 population을
        # 수동으로 "건설업"이라고 채워서 조회하니 실제 값(150.1천명=150,100명)이 golden
        # (150,000명)과 거의 정확히 일치했는데, 이 필드가 비어있으면 같은 조회를 해도
        # 엉뚱한 산업 코드가 잡혀 값이 안 맞았다.
        pop = row.get("population")
        stat_expr = row.get("statistic_expression")
        region = row.get("region")
        claim = Claim(
            sentence=sentence,
            claim_type=str(row.get("claim_type") or "규모"),
            period=str(period_val) if pd.notna(period_val) else None,
            source_org=org.strip() if isinstance(org, str) and org.strip() else None,
            search_query=_build_search_query(row),
            population=pop.strip() if isinstance(pop, str) and pop.strip() else None,
            statistic_expression=stat_expr.strip() if isinstance(stat_expr, str) and stat_expr.strip() else None,
            region=region.strip() if isinstance(region, str) and region.strip() else None,
        )

        try:
            candidates = search_and_rerank(
                claim, keyword_fn=keyword_search,
                embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
                vdb_fn=vdb_fn, bm25_fn=bm25_fn, top_k=TOP_K, document_texts=document_texts,
            )
        except Exception as e:
            print(f"[{i + 1}/{len(evalable)}] [검색실패] {sentence[:40]} -> {e}")
            continue

        result_ids = [c.table_id for c in candidates]
        rank = next((idx + 1 for idx, rid in enumerate(result_ids) if rid in gold_ids), None)

        if rank is not None:
            already_ok += 1
            continue  # 이미 rank로 hit — 재평가 대상 아님

        # --- MISS만 값 기준 재평가 ---
        kosis_value = row.get("kosis_value(실측)")
        golden_targets = []
        golden_targets += _extract_numbers(kosis_value)
        golden_targets += _extract_numbers(row.get("claim_value"))
        golden_targets += _extract_numbers(row.get("comparison_value"))
        golden_targets = list(dict.fromkeys(golden_targets))  # 중복 제거, 순서 유지

        kosis_period_raw = row.get("kosis_period")  # 비교 기준(base) 시점 원본
        current_period_raw = period_val  # claim의 현재 시점 원본(위에서 이미 읽음)

        if not golden_targets or not (pd.notna(kosis_period_raw) or pd.notna(current_period_raw)):
            true_miss.append((sentence, gold_id, "골든셋에 비교할 값/시점 없음"))
            print(f"[{i + 1}/{len(evalable)}] [진짜미스-비교불가] {sentence[:40]}")
            time.sleep(0.2)
            continue

        matched = None
        for cand in candidates[:CANDIDATES_TO_CHECK]:
            org_id = cand.org_id or _resolve_org_id(cand.table_id)
            if not org_id:
                skipped_no_org += 1
                continue

            try:
                detail = get_table_detail(cand.table_id, org_id)
            except DetailCacheUnavailableError:
                continue
            if detail.get("status") != "ok":
                continue
            prd_se = detail.get("prd_se")

            # 현재/기준 두 시점을 각각 조회 — "단순조회"형 claim은 한쪽만 맞아도 되고,
            # "증감률"형 claim은 두 값의 차이(diff)를 golden과 비교해야 하므로 둘 다 시도한다.
            fetched: list[tuple[str, float]] = []
            for label, raw_date in (("current", current_period_raw), ("base", kosis_period_raw)):
                if not pd.notna(raw_date):
                    continue
                period_str = _format_period(raw_date, prd_se)
                if not period_str:
                    continue
                generic_slots = {"period": period_str}
                dynamic = _build_dynamic_kosis_slots(cand.table_id, org_id, claim, generic_slots)
                if dynamic is None:
                    continue
                kosis_slots, base_req = dynamic
                try:
                    resp = client.call_dynamic(cand.table_id, kosis_slots, base_req)
                except KosisApiError as e:
                    fetch_errors += 1
                    msg = str(e)
                    if "건입니다" in msg:
                        error_categories["다차원_미확정(축 여러개 ALL)"] += 1
                    elif "40000" in msg or "err=31" in msg or "err=21" in msg:
                        error_categories["40000셀_초과"] += 1
                    elif "PRD_DE" in msg:
                        error_categories["시점_불일치"] += 1
                    else:
                        error_categories["기타:" + msg[:40]] += 1
                    continue
                except Exception as e:
                    fetch_errors += 1
                    error_categories["기타예외:" + type(e).__name__] += 1
                    continue
                scaled = _scaled_value(resp.raw_value, resp.unit)
                if scaled is not None:
                    fetched.append((label, scaled))
                time.sleep(0.15)

            cand_match = None
            for _label, val in fetched:
                if _values_match(val, golden_targets):
                    cand_match = val
                    break
            if cand_match is None and len(fetched) == 2:
                diff = abs(fetched[0][1] - fetched[1][1])
                if _values_match(diff, golden_targets):
                    cand_match = diff

            if cand_match is not None:
                matched = (cand.table_id, cand_match)
                break

        if matched:
            fake_miss.append((sentence, gold_id, matched[0], matched[1], golden_targets))
            print(f"[{i + 1}/{len(evalable)}] [가짜미스] gold={gold_id} 값일치후보={matched[0]}({matched[1]}) golden={golden_targets} | {sentence[:40]}")
        else:
            true_miss.append((sentence, gold_id, f"top{CANDIDATES_TO_CHECK} 중 값 일치 후보 없음"))
            print(f"[{i + 1}/{len(evalable)}] [진짜미스] gold={gold_id} golden={golden_targets} | {sentence[:40]}")

        time.sleep(0.3)

    n_total = len(evalable)
    n_rank_ok = already_ok
    n_fake = len(fake_miss)
    n_true = len(true_miss)
    corrected_recall = (n_rank_ok + n_fake) / n_total if n_total else 0.0

    print("\n=== 최종 요약 ===")
    print(f"평가 대상: {n_total}건")
    print(f"기존 rank 기준 hit: {n_rank_ok}건")
    print(f"MISS 중 값 일치(가짜 미스): {n_fake}건")
    print(f"MISS 중 값도 불일치(진짜 미스): {n_true}건")
    print(f"보정 Recall@{TOP_K}(값 기준): {n_rank_ok + n_fake}/{n_total} = {corrected_recall:.1%}")
    print(f"(참고: KOSIS 조회 실패 {fetch_errors}건, org_id 없어 스킵 {skipped_no_org}건)")
    print("=== 에러 유형 분포(진단용) ===")
    for cat, cnt in error_categories.most_common(15):
        print(f"  {cat}: {cnt}건")

    report_lines.append(f"- 평가 대상: {n_total}건\n")
    report_lines.append(f"- 기존 rank 기준 hit: {n_rank_ok}건\n")
    report_lines.append(f"- MISS 중 값 일치(가짜 미스): {n_fake}건\n")
    report_lines.append(f"- MISS 중 값도 불일치(진짜 미스): {n_true}건\n")
    report_lines.append(f"- 보정 Recall@{TOP_K}(값 기준): {n_rank_ok + n_fake}/{n_total} = {corrected_recall:.1%}\n\n")
    report_lines.append("## 가짜 미스(값은 맞았음)\n\n")
    for sentence, gold_id, cand_id, val, golden in fake_miss:
        report_lines.append(f"- {sentence} (gold={gold_id}, 후보={cand_id}, 조회값={val}, golden={golden})\n")
    report_lines.append("\n## 진짜 미스\n\n")
    for sentence, gold_id, reason in true_miss:
        report_lines.append(f"- {sentence} (gold={gold_id}): {reason}\n")
    REPORT_PATH.write_text("".join(report_lines), encoding="utf-8")
    print(f"리포트 -> {REPORT_PATH}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
