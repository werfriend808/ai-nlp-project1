"""
agent/pipeline/pipeline_1_4.py — 1→2→3→4단계 파이프라인 연결 (5~8단계는 여기서 다루지 않음)

⚠️ batch_runner.py(1~8단계, 실전2 준비하면서 임시로 만든 버전)와의 관계:
    batch_runner.py의 4단계(run_stage_4)는 옛날 fill_slots+clarify만 쓰고 있어서,
    agent_chat.py(실전2)에서 새로 만든 표별 슬롯 힌트(코드 기반 1차/LLM 2차), 나이대
    다중코드, 극값(코로나 이후/N년 만에/역대) 감지 로직이 전혀 반영돼 있지 않다.
    이 모듈은 batch_runner.py를 고치는 대신, agent_chat.py에 이미 쌓인 성숙한 로직을
    그대로 재사용해서 1~4단계만 새로 연결한다. 5~8단계가 완성되면 그때 이 모듈 위에
    이어붙이거나, batch_runner.py를 완전히 대체한다 (2026-08-04 결정).

⚠️ 실시간 대화(agent_chat.py)와 다른 점 — 되묻기(clarify) 처리:
    agent_chat.py는 사용자가 바로 답해주는 걸 전제로 만들어졌지만, 이 모듈은 실제 기사
    데이터를 사람 개입 없이 배치로 돌린다. 되묻기가 필요한 상황(period/region/calc_type
    미해결)이 생기면 가짜 답변으로 채우지 않고 "4단계_미해결"로 정직하게 표시하고 다음
    주장으로 넘어간다 — 1~4단계가 실제로 얼마나 자동완성되는지 왜곡 없이 측정하기 위함
    (2026-08-04 사용자 결정).

실행:
    python -m agent.pipeline.pipeline_1_4          # ARTICLES 시나리오
    python -m agent.pipeline.pipeline_1_4 --csv    # data_set.csv 실제 기사 샘플
"""

from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from agent.interfaces import Claim, TableCandidate
from agent.preprocessing.classifier import classify, ClassifierError
from agent.preprocessing.claim_extractor import extract_claims, ClaimExtractorError
from agent.mapping.keyword_search import keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import search_and_rerank, load_document_texts
from agent.orchestrator.slot_filler import fill_slots
from agent.orchestrator.agent_chat import (
    _extract_table_dimension_hints,
    _extract_demographic_hints,
    _extract_table_dimension_hints_llm,
    _normalize_short_reply,
    _normalize_year_expression,
    _extract_month_period,
    _wants_comparison,
    _resolve_decade_age_codes,
    _resolve_since_event_start_year,
    _N_YEARS_SINCE_RE,
    _ALL_TIME_RE,
    _REGION_LIKE_DIMENSIONS,
    build_kosis_slots,
)

TABLE_PARAMS_PATH = Path(__file__).parent.parent / "kosis" / "table_params.json"
DATA_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "data_set.csv"


@dataclass
class Stage4Result:
    """주장 1건이 1~4단계를 거친 결과 — 5단계로 넘길 수 있는지 판단하는 최종 산출물."""

    claim: Claim
    status: str  # "3단계_매칭없음" | "3단계_매칭_불충분" | "4단계_미해결" | "5단계_진행가능"
    table: Optional[TableCandidate] = None
    slots: dict = field(default_factory=dict)  # generic: period/region/calc_type
    dimension_hints: dict = field(default_factory=dict)  # 표별 세부 축: age/gender/...
    missing_slots: list[str] = field(default_factory=list)
    special_resolution: Optional[str] = None  # 5단계가 참고할 특수 처리 힌트
    kosis_slots: Optional[dict] = None  # 5단계 api_client에 그대로 넘길 수 있는 최종 슬롯


def _needs_region_for_table(table_id: str, table_params: dict) -> bool:
    """ChatSession._needs_region()과 동일한 판단 — code_map이 "전국" 하나뿐인 표는
    지역을 물어봐도 의미가 없으므로 필수로 안 본다."""
    dims = table_params.get(table_id, {}).get("dimensions", {})
    for dim_name in _REGION_LIKE_DIMENSIONS:
        dim = dims.get(dim_name)
        if dim and len(set(dim.get("code_map", {}).values())) > 1:
            return True
    return False


def _detect_special_resolution(claim_sentence: str, table_id: str, table_params: dict) -> Optional[str]:
    """5단계가 단일 KosisApiResponse 대신 다중코드/시계열 조회를 써야 하는 경우를 표시한다.
    실제 조회는 여기서 하지 않고(그건 5단계 몫), 패턴 감지만 한다."""
    age_code_map = table_params.get(table_id, {}).get("dimensions", {}).get("age", {}).get("code_map", {})
    if age_code_map and _resolve_decade_age_codes(claim_sentence, age_code_map) is not None:
        return "나이대_다중코드"
    if _resolve_since_event_start_year(claim_sentence) is not None:
        return "극값_이벤트기준"
    if _N_YEARS_SINCE_RE.search(claim_sentence):
        return "극값_N년만에"
    if _ALL_TIME_RE.search(claim_sentence):
        return "극값_역대"
    return None


def resolve_claim_1_to_4(
    claim: Claim,
    article_date: date,
    *,
    embedding_cache: dict,
    document_texts: dict,
    table_params: dict,
) -> Stage4Result:
    """주장 1건에 대해 3단계(표 매핑) + 4단계(슬롯필링)를 수행한다. 1·2단계는 기사 단위라
    run_article_1_4()에서 먼저 처리되고, 이 함수는 그 결과로 나온 Claim 하나를 받는다."""

    # --- 3단계: 통계표 매핑 ---
    candidates = search_and_rerank(
        claim,
        keyword_fn=keyword_search,
        embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
        top_k=3,
        document_texts=document_texts,
    )
    if not candidates:
        return Stage4Result(claim=claim, status="3단계_매칭없음")

    top = candidates[0]
    if top.source_meta and "unverified" in top.source_meta:
        return Stage4Result(claim=claim, status="3단계_매칭_불충분", table=top)

    # --- 4단계: 슬롯필링 ---
    # 4-1) 표별 세부 축 힌트: 코드 기반(1차) + 동의어(청년 등) + LLM 폴백(2차)
    dimension_hints = _extract_table_dimension_hints(claim.sentence, top.table_id, table_params)
    demographic_hints = _extract_demographic_hints(claim.sentence)
    combined_hints = {**demographic_hints, **dimension_hints}

    all_dims = table_params.get(top.table_id, {}).get("dimensions", {})
    unresolved_dims = [d for d in all_dims if d not in combined_hints]
    llm_hints = _extract_table_dimension_hints_llm(claim.sentence, top.table_id, table_params, unresolved_dims)
    combined_hints = {**llm_hints, **combined_hints}

    # 4-2) generic 슬롯(period/region/calc_type): agent_chat.py와 동일한 정규화 순서
    normalized = _normalize_short_reply(_normalize_year_expression(claim.sentence))
    slots = fill_slots(normalized, {}, article_date)

    month_period = _extract_month_period(claim.sentence)
    if month_period:
        slots["period"] = month_period

    # 4-3) 되묻기 미해결 판정: agent_chat.py의 tool_reask 우선순위와 동일한 필요조건
    needs_region = _needs_region_for_table(top.table_id, table_params)
    needs_calc_type = _wants_comparison(claim.sentence)

    missing_slots = []
    if not slots.get("period"):
        missing_slots.append("period")
    if needs_region and not slots.get("region"):
        missing_slots.append("region")
    if needs_calc_type and not slots.get("calc_type"):
        missing_slots.append("calc_type")

    special_resolution = _detect_special_resolution(claim.sentence, top.table_id, table_params)

    if missing_slots:
        return Stage4Result(
            claim=claim,
            status="4단계_미해결",
            table=top,
            slots=slots,
            dimension_hints=combined_hints,
            missing_slots=missing_slots,
            special_resolution=special_resolution,
        )

    kosis_slots = build_kosis_slots(top.table_id, {**slots, **combined_hints}, table_params)
    return Stage4Result(
        claim=claim,
        status="5단계_진행가능",
        table=top,
        slots=slots,
        dimension_hints=combined_hints,
        missing_slots=[],
        special_resolution=special_resolution,
        kosis_slots=kosis_slots,
    )


def run_article_1_4(
    article: dict,
    *,
    embedding_cache: dict,
    document_texts: dict,
    table_params: dict,
    verbose: bool = True,
) -> list[Stage4Result]:
    """기사 1건을 1단계(분류)→2단계(주장추출)→주장별 3~4단계까지 돌린다.
    1단계에서 무관 판정되면 빈 리스트를 반환한다."""
    if verbose:
        print(f"\n{'=' * 60}")
        print(article.get("label", article.get("article_title", "(제목 없음)")))

    try:
        cls_result = classify(article["article_text"])
    except ClassifierError as e:
        if verbose:
            print(f"[1단계] 분류 실패 ({e}) → 기사 스킵")
        return []

    if not cls_result.label:
        if verbose:
            print(f"[1단계] 무관한 기사로 판정(score={cls_result.score:.2f}) → 스킵")
        return []

    try:
        claims = extract_claims(article["article_text"])
    except ClaimExtractorError as e:
        if verbose:
            print(f"[2단계] 주장 추출 실패 ({e}) → 기사 스킵")
        return []

    if verbose:
        print(f"[1단계] 관련 기사 판정(score={cls_result.score:.2f})")
        print(f"[2단계] 주장 {len(claims)}건 추출")

    results = []
    for claim in claims:
        result = resolve_claim_1_to_4(
            claim,
            article["published_date"],
            embedding_cache=embedding_cache,
            document_texts=document_texts,
            table_params=table_params,
        )
        results.append(result)
        if verbose:
            _print_claim_result(result)
    return results


def _print_claim_result(r: Stage4Result) -> None:
    print(f"  - \"{r.claim.sentence}\" → {r.status}", end="")
    if r.table:
        print(f" | 표={r.table.table_name}({r.table.table_id})", end="")
    if r.missing_slots:
        print(f" | 미해결={r.missing_slots}", end="")
    if r.special_resolution:
        print(f" | 특수처리={r.special_resolution}", end="")
    print()


def load_articles_from_csv(path: Path = DATA_CSV_PATH, n: int = 15, seed: int = 42) -> list[dict]:
    # data_set.csv 맨 앞에 BOM(﻿)이 붙어있어 encoding="utf-8"로 열면 첫 컬럼명이
    # "﻿기사제목"으로 읽혀 row.get("기사제목")이 항상 빈 문자열을 반환하던 버그.
    # "utf-8-sig"는 BOM을 자동으로 벗겨내서 컬럼명이 정상적으로 "기사제목"이 된다.
    with open(path, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("검색 구분 레이블", "").strip().lower() == "true"]

    random.Random(seed).shuffle(rows)

    articles = []
    for row in rows[:n]:
        try:
            y, m, d = (int(v) for v in row["작성일"].split("-"))
            published = date(y, m, d)
        except (KeyError, ValueError):
            published = date(2025, 1, 1)
        title = row.get("기사제목", "")
        articles.append(
            {
                "label": f"[data_set.csv] {title[:40]}",
                "article_title": title,
                "published_date": published,
                "article_text": row["기사 본문 전체"][:3000],
            }
        )
    return articles


def print_summary(all_results: list[Stage4Result]) -> None:
    print(f"\n{'=' * 60}")
    print("1~4단계 파이프라인 완성도 요약")
    print(f"{'=' * 60}")

    total = len(all_results)
    print(f"\n표 매핑 시도된 주장: {total}건")
    if total == 0:
        return

    counts: dict[str, int] = {}
    for r in all_results:
        counts[r.status] = counts.get(r.status, 0) + 1

    for status in ("5단계_진행가능", "4단계_미해결", "3단계_매칭_불충분", "3단계_매칭없음"):
        n = counts.get(status, 0)
        print(f"  {status}: {n}건 ({n / total * 100:.1f}%)")

    missing_breakdown: dict[str, int] = {}
    for r in all_results:
        for slot in r.missing_slots:
            missing_breakdown[slot] = missing_breakdown.get(slot, 0) + 1
    if missing_breakdown:
        print(f"\n[미해결 슬롯 분포] {missing_breakdown}")

    special_breakdown: dict[str, int] = {}
    for r in all_results:
        if r.special_resolution:
            special_breakdown[r.special_resolution] = special_breakdown.get(r.special_resolution, 0) + 1
    if special_breakdown:
        print(f"[특수 처리 감지] {special_breakdown} (5단계에서 다중코드/시계열 조회 필요)")


def main(use_csv_sample: bool = False, csv_n: int = 15) -> None:
    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)

    embedding_cache = build_table_embedding_cache()
    document_texts = load_document_texts()

    if use_csv_sample:
        articles = load_articles_from_csv(n=csv_n)
    else:
        from agent.pipeline.batch_runner import ARTICLES  # 시나리오 재사용(읽기 전용)

        articles = ARTICLES

    all_results: list[Stage4Result] = []
    for article in articles:
        all_results.extend(
            run_article_1_4(
                article,
                embedding_cache=embedding_cache,
                document_texts=document_texts,
                table_params=table_params,
            )
        )

    print_summary(all_results)


if __name__ == "__main__":
    main(use_csv_sample="--csv" in sys.argv)