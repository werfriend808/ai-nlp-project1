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
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from agent.interfaces import Claim, TableCandidate
from agent.preprocessing.classifier import classify
from agent.preprocessing.claim_extractor import extract_claims
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
    _resolve_relative_month_period,
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


_NATIONAL_DEFAULT_LABELS = ("전국",)


def _has_safe_national_default(dim: dict) -> bool:
    """default_value가 "전국"류 라벨을 가리키면, 지역 언급이 없어도 조용히 그 값으로
    채워도 안전하다. 배치 파이프라인엔 되물어볼 사용자가 없으므로(agent_chat.py의 챗봇과
    달리), 안전한 기본값이 있는데도 "미해결"로 막는 건 과잉이다 — 2026-08-05, "2020년
    인구주택총조사 응답률 96.3%"처럼 전국 단위 통계인데 지역 미언급을 이유로 불필요하게
    미해결 처리되던 버그를 실제 기사 테스트에서 발견해서 수정."""
    default = dim.get("default_value")
    code_map = dim.get("code_map", {})
    labels = [label for label, code in code_map.items() if code == default]
    return any(label in _NATIONAL_DEFAULT_LABELS for label in labels)


def _needs_region_for_table(table_id: str, table_params: dict) -> bool:
    """표에 실제 지역 축(코드 2개 이상)이 있고, 그 기본값이 "전국"류 안전한 기본값이
    아닐 때만 필수로 본다. 기본값이 "전국"이면(예: DT_1B04005N) 지역 언급이 없어도
    조용히 전국으로 채워도 안전하므로 미해결로 막지 않는다."""
    dims = table_params.get(table_id, {}).get("dimensions", {})
    for dim_name in _REGION_LIKE_DIMENSIONS:
        dim = dims.get(dim_name)
        if dim and len(set(dim.get("code_map", {}).values())) > 1 and not _has_safe_national_default(dim):
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
    elif not slots.get("period"):
        # "지난 10월"/"올 5월"처럼 연도 없이 상대적으로 월만 가리키는 표현은 기사 날짜
        # 기준으로 계산해야 해서 _extract_month_period(절대 연도 표현)와 분리돼 있다.
        relative_month_period = _resolve_relative_month_period(claim.sentence, article_date)
        if relative_month_period:
            slots["period"] = relative_month_period

    special_resolution = _detect_special_resolution(claim.sentence, top.table_id, table_params)

    # 4-3) 되묻기 미해결 판정: agent_chat.py의 tool_reask 우선순위와 동일한 필요조건
    needs_region = _needs_region_for_table(top.table_id, table_params)
    # 극값(역대/이벤트기준/N년만에)이 이미 감지됐으면 "이 claim은 최댓값검증이 필요하다"는
    # 걸 이미 알고 있는 것이므로, calc_type 슬롯이 따로 안 채워졌다고 미해결로 막지 않는다
    # (2026-08-05 실제 기사에서 발견 — "10년 만에 처음이다"가 극값_N년만에는 정확히
    # 감지됐는데 calc_type만 안 채워졌다는 이유로 미해결 처리되던 모순을 수정).
    is_extremum = bool(special_resolution) and special_resolution.startswith("극값")
    needs_calc_type = _wants_comparison(claim.sentence) and not is_extremum

    missing_slots = []
    if not slots.get("period"):
        missing_slots.append("period")
    if needs_region and not slots.get("region"):
        missing_slots.append("region")
    if needs_calc_type and not slots.get("calc_type"):
        missing_slots.append("calc_type")

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
) -> dict:
    """기사 1건을 1단계(분류)→2단계(주장추출)→주장별 3~4단계까지 돌린다.

    반환값은 {"article_title", "classify_score", "skip_stage", "claim_results"} 형태.
    skip_stage가 None이 아니면 1~2단계에서 걸러진 것(claim_results는 빈 리스트).
    claim_results는 Stage4Result 리스트 — 기존 all_results 취합(print_summary용)과
    JSONL 구조화 출력(분석용) 양쪽에서 그대로 재사용한다.
    """
    title = article.get("label", article.get("article_title", "(제목 없음)"))
    if verbose:
        print(f"\n{'=' * 60}")
        print(title)

    base = {"article_title": title, "classify_score": None, "skip_stage": None, "claim_results": []}

    try:
        cls_result = classify(article["article_text"])
    except Exception as e:  # noqa: BLE001 - HCX 네트워크 순간 끊김 등도 이 기사 하나만
        # 스킵하고 배치 전체는 계속 진행 (2026-08-05, 실제 배치 실행 중
        # requests.exceptions.ConnectionError로 전체가 죽는 걸 실측으로 발견해서 방어 추가)
        if verbose:
            print(f"[1단계] 분류 실패 ({type(e).__name__}: {e}) → 기사 스킵")
        base["skip_stage"] = f"1단계_분류실패({type(e).__name__})"
        return base

    base["classify_score"] = cls_result.score
    if not cls_result.label:
        if verbose:
            print(f"[1단계] 무관한 기사로 판정(score={cls_result.score:.2f}) → 스킵")
        base["skip_stage"] = "1단계_무관"
        return base

    try:
        claims = extract_claims(article["article_text"])
    except Exception as e:  # noqa: BLE001 - 위와 동일한 이유
        if verbose:
            print(f"[2단계] 주장 추출 실패 ({type(e).__name__}: {e}) → 기사 스킵")
        base["skip_stage"] = f"2단계_추출실패({type(e).__name__})"
        return base

    if verbose:
        print(f"[1단계] 관련 기사 판정(score={cls_result.score:.2f})")
        print(f"[2단계] 주장 {len(claims)}건 추출")

    results = []
    for claim in claims:
        try:
            result = resolve_claim_1_to_4(
                claim,
                article["published_date"],
                embedding_cache=embedding_cache,
                document_texts=document_texts,
                table_params=table_params,
            )
        except Exception as e:  # noqa: BLE001 - HCX/KOSIS 네트워크 순간 끊김 등으로 claim
            # 하나가 실패해도 나머지 claim/기사는 계속 처리 (2026-08-05, 실제 배치 실행 중
            # ConnectionError로 전체 배치가 죽는 걸 실측으로 발견해서 방어 추가)
            if verbose:
                print(f"  - \"{claim.sentence}\" → 처리오류 ({type(e).__name__}: {e})")
            results.append(Stage4Result(claim=claim, status="처리오류"))
            continue
        results.append(result)
        if verbose:
            _print_claim_result(result)
    base["claim_results"] = results
    return base


def _result_to_json_row(article_title: str, r: Stage4Result) -> dict:
    return {
        "type": "claim",
        "article_title": article_title,
        "sentence": r.claim.sentence,
        "status": r.status,
        "table_id": r.table.table_id if r.table else None,
        "table_name": r.table.table_name if r.table else None,
        "table_source_meta": r.table.source_meta if r.table else None,
        "missing_slots": r.missing_slots,
        "special_resolution": r.special_resolution,
        "slots": r.slots,
        "dimension_hints": r.dimension_hints,
    }


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

    for status in ("5단계_진행가능", "4단계_미해결", "3단계_매칭_불충분", "3단계_매칭없음", "처리오류"):
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


def main(use_csv_sample: bool = False, csv_n: int = 15, csv_seed: int = 42, out_path: Optional[str] = None) -> None:
    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)

    embedding_cache = build_table_embedding_cache()
    document_texts = load_document_texts()

    if use_csv_sample:
        articles = load_articles_from_csv(n=csv_n, seed=csv_seed)
    else:
        from agent.pipeline.batch_runner import ARTICLES  # 시나리오 재사용(읽기 전용)

        articles = ARTICLES

    out_file = open(out_path, "w", encoding="utf-8") if out_path else None

    all_results: list[Stage4Result] = []
    try:
        for article in articles:
            article_result = run_article_1_4(
                article,
                embedding_cache=embedding_cache,
                document_texts=document_texts,
                table_params=table_params,
            )
            all_results.extend(article_result["claim_results"])

            if out_file is None:
                continue
            title = article_result["article_title"]
            if article_result["skip_stage"]:
                row = {
                    "type": "skip",
                    "article_title": title,
                    "classify_score": article_result["classify_score"],
                    "skip_stage": article_result["skip_stage"],
                }
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            else:
                for r in article_result["claim_results"]:
                    row = _result_to_json_row(title, r)
                    out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_file.flush()
    finally:
        if out_file:
            out_file.close()

    print_summary(all_results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true", help="ARTICLES 대신 data_set.csv 실제 기사 샘플 사용")
    parser.add_argument("--n", type=int, default=15, help="--csv일 때 샘플 기사 수 (기본 15)")
    parser.add_argument("--seed", type=int, default=42, help="--csv 샘플링 랜덤 시드 (기본 42, 같은 시드면 매번 동일 샘플)")
    parser.add_argument("--out", type=str, default=None, help="주장별 결과를 JSONL로 저장할 경로 (분석용, 생략 가능)")
    args = parser.parse_args()

    main(use_csv_sample=args.csv, csv_n=args.n, csv_seed=args.seed, out_path=args.out)