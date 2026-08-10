# 1→2→3→4→5→6→7→8단계 전체 CSV 순회 실행
"""
agent/pipeline/batch_runner.py — 1→2→3→4→5→6→7→8단계 전체 자동 연결 실행기

⚠️ 7·8단계 추가 (전체 통합):
    6단계까지 계산된 ComputedResult가 나오면 D의 judge()로 판정(7단계)하고,
    그 Verdict를 다시 explain()에 넘겨 사람이 읽을 수 있는 최종 설명(8단계)까지
    생성합니다. judge/explain 둘 다 실패해도(JudgeError/ExplainerError 등) 배치
    전체가 멈추지 않도록 run_stage_7_8에서 개별적으로 잡고 다음 주장/기사로 넘어갑니다.

⚠️ 이전 버전과의 차이:
    예전에는 3단계(통계표 자동 매핑)가 없어서 table_id/claim_sentence를 시나리오마다
    사람이 손으로 지정했습니다. 지금은 B의 search_and_rerank()가 완성되어 있어서
    2단계가 뽑은 Claim을 그대로 3단계에 흘려보내 table_id를 자동으로 정합니다.

    4단계(slot_filler/clarify)도 마찬가지로, generic_slots를 손으로 채워두는 대신
    D의 fill_slots()/clarify()를 실제로 호출합니다. clarify()가 되묻기 질문을
    반환하면 시나리오에 준비된 clarify_reply(사용자가 한 번 더 답했다고 가정한
    발화)로 한 번 더 채워보고, 그래도 부족하면 그 주장은 스킵합니다.

    5단계로 넘어가기 전에 D의 generic slots(period/region/calc_type, 표 구분 없이
    고정)를 C의 table_params.json에 정의된 표별 dimensions(gender/age 등)로
    변환하는 다리(build_kosis_slots)가 필요합니다 — 이 변환이 3단계 표 매핑이
    실제로 없던 시절엔 존재하지 않았던 부분입니다.

    실행 (프로젝트 루트에서):
        python -m agent.pipeline.batch_runner

⚠️ 사전 준비물:
    .env에 HCX_API_KEY, KOSIS_API_KEY 둘 다 필요합니다. 1·2·4단계는 HCX API를,
    5단계는 KOSIS API를 실제로 호출합니다 (더미 아님).

⚠️ 연결하면서 실제로 드러난 팀 간 불일치 (일부러 감추지 않고 그대로 노출시킴):
    1. B(table_catalog.json)와 C(table_params.json)가 서로 다른 표를 가리키는 경우가
       있습니다. 예: "청년 실업률" 계열 주장에 대해 B는 DT_1DA7001S(성별 경제활동인구
       총괄)를 최상위로 매칭하는데, C의 table_params.json에는 그 표가 없고 대신
       DT_1DA7102S(성/연령별 실업률)만 등록돼 있습니다. → build_kosis_slots가
       table_params.json에 없는 table_id를 만나면 None을 반환하고, 그 주장은
       "5단계 파라미터 없음"으로 표시하며 건너뜁니다.
    2. D(clarify_rules.REQUIRED_SLOTS)는 모든 표에 대해 region을 무조건 필수로
       요구하는데, 실제 KOSIS 표 중에는(DT_1DA7102S처럼) 지역 축 자체가 없는 표도
       있습니다. 이 경우도 사람이 미리 안 걸러주면 "지역이 없는 표인데 지역을
       되묻는" 상황이 그대로 재현됩니다 (아래 실행 결과 참고).
"""

from __future__ import annotations

import csv
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from agent.preprocessing.classifier import classify
from agent.preprocessing.claim_extractor import extract_claims
from agent.mapping.keyword_search import SYNONYMS, keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import search_and_rerank
from agent.orchestrator.slot_filler import fill_slots
from agent.orchestrator.clarify import clarify
from agent.kosis.api_client import KosisApiClient, KosisApiError
from agent.kosis.calculator import KosisCalculator, CalculationError
from agent.verdict.judge import judge, JudgeError
from agent.explain.explainer import explain, ExplainerError
from agent.interfaces import ComputedResult, Explanation, Verdict
from db.store import insert_verification, make_result_id

TABLE_PARAMS_PATH = Path(__file__).parent.parent / "kosis" / "table_params.json"
TABLE_CATALOG_PATH = Path(__file__).parent.parent / "mapping" / "table_catalog.json"
# "data.csv"라는 존재하지 않는 파일을 가리키고 있어서 --csv 모드가 항상 FileNotFoundError로
# 실패했음 (실제 파일명은 data_set.csv, data/ 디렉터리에 이것만 있음) — 여기서 다시 고침.
DATA_CSV_PATH = Path(__file__).parent.parent.parent / "data" / "data_set.csv"
# ARTICLES의 시나리오들이 공통으로 쓰는 되묻기 답변 — region/period/calc_type을 한 번에
# 채워주는 발화라, CSV에서 무작위로 뽑은 실제 기사에도 동일하게 재사용한다.
DEFAULT_CLARIFY_REPLY = "전국 기준으로 작년 대비 증감률 알려줘"


def _load_table_catalog_by_id(path: Path = TABLE_CATALOG_PATH) -> dict[str, dict]:
    """table_catalog.json(3단계 B가 관리)을 tblId 기준으로 인덱싱 — statistic_category
    필드를 매칭된 table_id로 조회하기 위한 용도(새 추출 없이 기존 데이터 재사용)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {t["tblId"]: t for t in data.get("tables", [])}


def _normalize_statistic_name(expression: Optional[str]) -> Optional[str]:
    """3단계 keyword_search.py의 기존 SYNONYMS 사전을 재사용해서 기사 표현을 정규화된
    통계 용어로 바꾼다 (새 LLM 호출 없음). 매칭 안 되면 원래 표현 그대로 둔다."""
    if not expression:
        return None
    for raw_term, mapped_terms in SYNONYMS.items():
        if raw_term in expression:
            return mapped_terms[0]
    return expression


def _clean_scraped_article_text(title: str, raw_text: str, max_len: int = 3000) -> str:
    """실제 스크랩 기사(CSV의 '기사 본문 전체')는 신문사 내비게이션 메뉴가 본문 앞에
    반복적으로 붙어있어서(광고/관련기사 텍스트까지 합치면 2만자 넘는 경우도 있음),
    그대로 HCX에 넘기면 "40003 Context length exceeded"로 거부당한다 (실제 재현됨).

    기사제목의 앞부분을 raw_text 안에서 찾아 그 위치부터 잘라내는 방식으로 내비게이션
    잡음을 건너뛰고, 이후 max_len자만 남겨서 컨텍스트 길이를 안전하게 유지한다.
    제목을 못 찾으면(예외 케이스) 그냥 앞에서부터 max_len자를 쓴다.
    """
    anchor = title[:12].strip()
    idx = raw_text.find(anchor) if anchor else -1
    start = idx if idx >= 0 else 0
    return raw_text[start : start + max_len]


def load_articles_from_csv(
    path: Path = DATA_CSV_PATH, n: int = 15, seed: int = 42
) -> list[dict]:
    """data_set.csv에서 '검색 구분 레이블'이 True인 기사 중 n건을 무작위 샘플링해서
    ARTICLES와 같은 형식(label/published_date/article_text/clarify_reply)으로 변환한다.

    Day3 "전체 통합" 체크리스트가 요구하는 "전체 CSV 또는 대표 샘플" 실행을 위한 것.
    손으로 쓴 ARTICLES(알려진 이슈 재현용)와 달리, 실제 기사라 어떤 clarify 질문이
    나올지 미리 알 수 없어서 여러 슬롯을 한 번에 답하는 범용 문구를 그대로 재사용한다.
    """
    with open(path, encoding="utf-8") as f:
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
                "article_url": row.get("URL"),
                "published_date": published,
                "article_text": _clean_scraped_article_text(title, row["기사 본문 전체"]),
                "clarify_reply": DEFAULT_CLARIFY_REPLY,
            }
        )
    return articles


ARTICLES = [
    {
        "label": "시나리오 1 [고용/노동] 청년 실업률 — 정상 자동 연결",
        "published_date": date(2025, 1, 6),
        "article_text": (
            "6일 통계청이 발표한 고용동향에 따르면 지난달 청년 실업률이 6%에 육박한 "
            "것으로 나타났다. 청년층 취업자 수는 46개월 만에 감소로 전환했다."
        ),
        "clarify_reply": "전국 기준으로 작년 대비 증감률 알려줘",
    },
    {
        "label": "시나리오 2 [물가/CPI] 소비자물가 — 정상 자동 연결 (단, 월/연 비교 기준 차이 있음, 이슈 1-2 참고)",
        "published_date": date(2025, 2, 5),
        "article_text": (
            "5일 통계청이 발표한 소비자물가동향에 따르면 지난달 소비자물가가 전년 "
            "동월 대비 2.2% 오른 것으로 나타났다."
        ),
        "clarify_reply": "전국 기준으로 작년 대비 증감률 알려줘",
    },
    {
        "label": "시나리오 3 [인구] 주민등록인구 감소 — 정상 자동 연결 (단순 조회 경로)",
        "published_date": date(2025, 1, 3),
        "article_text": (
            "행정안전부에 따르면 지난해 12월 기준 주민등록인구는 5017만명으로 "
            "5년 연속 감소세를 이어갔다."
        ),
        "clarify_reply": "전국 기준으로 알려줘",
    },
    {
        "label": "시나리오 4 [경제성장] GDP 성장률 — 예상된 실패 (DT_200Y102는 분기 단위 전용, 이슈 1-3)",
        "published_date": date(2025, 1, 23),
        "article_text": (
            "한국은행이 발표한 국민소득(잠정)에 따르면 작년 4분기 실질 GDP는 "
            "전기 대비 0.2% 성장에 그쳤다."
        ),
        "clarify_reply": "전국 기준으로 알려줘",
    },
    {
        "label": "시나리오 5 [무역/수출입] 수출 역대 최대 — 정상 자동 연결 (단순 조회 경로)",
        "published_date": date(2025, 1, 1),
        "article_text": (
            "관세청에 따르면 지난해 수출액은 6838억달러로 역대 최대치를 기록했다."
        ),
        "clarify_reply": "전국 기준으로 알려줘",
    },
    {
        "label": "시나리오 6 [무역/수출입] 무역수지 흑자 전환 — 정상 자동 연결 (증감 경로, 단 itmId는 수출액 고정이라 실제로는 무역수지 자체가 아닌 수출액 증감으로 검증됨, 이슈 1-1)",
        "published_date": date(2025, 1, 10),
        "article_text": (
            "산업통상자원부는 지난해 무역수지가 3년 만에 흑자로 전환됐다고 밝혔다."
        ),
        "clarify_reply": "전국 기준으로 작년 대비 증감률 알려줘",
    },
    {
        "label": "시나리오 7 [부동산/주택] 집값 하락 — 예상된 실패 (DT_30404_B012는 월 단위 전용, 이슈 1-3)",
        "published_date": date(2025, 2, 1),
        "article_text": (
            "한국부동산원 조사에 따르면 지난달 전국 아파트 매매가격은 하락세를 이어갔다."
        ),
        "clarify_reply": "전국 기준으로 알려줘",
    },
    {
        "label": "시나리오 8 [출생/사망/혼인] 출생아 수 증가 — 정상 자동 연결 (증감률 경로)",
        "published_date": date(2025, 1, 26),
        "article_text": (
            "통계청에 따르면 작년 출생아 수는 23만8천명으로 전년보다 늘어난 것으로 "
            "나타났다."
        ),
        "clarify_reply": "전국 기준으로 작년 대비 증감률 알려줘",
    },
    {
        "label": "시나리오 9 [출생/사망/혼인] 혼인 건수 역대 최저 — 알려진 왜곡 (vital_item이 출생아수로 고정돼있어 실제로는 혼인 건수가 아니라 출생아수 값으로 잘못 조회됨, 이슈 1-1)",
        "published_date": date(2025, 3, 19),
        "article_text": (
            "통계청에 따르면 지난해 혼인 건수는 역대 최저치를 기록했다."
        ),
        "clarify_reply": "전국 기준으로 알려줘",
    },
    {
        "label": "시나리오 10 [무관한 기사] 통계 주장 없음 — 1단계에서 걸러져야 함 (배치 안정성 확인용)",
        "published_date": date(2025, 1, 2),
        "article_text": (
            "3일 서울 종로구의 한 상가건물에서 화재가 발생해 소방당국이 진화 작업을 "
            "벌였다. 인명피해는 없는 것으로 파악됐다."
        ),
        "clarify_reply": None,
    },
]


def build_kosis_slots(table_id: str, generic_slots: dict, table_params: dict) -> Optional[dict]:
    """D의 generic slots(period/region/calc_type, 표 구분 없이 고정)를
    C의 table_params.json에 정의된 표별 dimensions로 변환한다.

    table_params.json에 이 table_id 자체가 없으면 None을 반환한다 (B가 고른 표를
    C가 아직 조사 안 한 경우 — 위 모듈 docstring 이슈 1 참고).
    """
    if table_id not in table_params:
        return None

    base = table_params[table_id]
    kosis_slots: dict = {"period": generic_slots.get("period")}

    for dim_name, dim in base.get("dimensions", {}).items():
        # 이 표에 정의된 축(dim_name)만 채운다. generic_slots에 값이 있으면 쓰고,
        # 없으면 표의 default_value로 채운다 (예: region이 없는 표는 gender/age만 봄).
        value = generic_slots.get(dim_name)
        kosis_slots[dim_name] = value if value is not None else dim.get("default_value")

    return kosis_slots


def run_stage_4(claim_sentence: str, clarify_reply: Optional[str], article_date: date) -> Optional[dict]:
    """4단계: fill_slots + clarify. 한 번에 안 채워지면 clarify_reply로 한 번 더 시도.
    그래도 부족하면 None (되묻기 미해결 → 5단계로 못 감)을 반환한다."""
    slots = fill_slots(claim_sentence, {}, article_date)
    question = clarify(slots)
    print(f"[4단계 slot_filler] 1차 슬롯: {slots}")

    if question and clarify_reply:
        print(f"[4단계 clarify] 되묻기: \"{question}\" → (준비된 답변) \"{clarify_reply}\"")
        slots = fill_slots(clarify_reply, slots, article_date)
        question = clarify(slots)
        print(f"[4단계 slot_filler] 2차 슬롯: {slots}")

    if question:
        print(f"[4단계 clarify] 여전히 부족 → 되묻기: \"{question}\" (여기서 중단)")
        return None

    print("[4단계 clarify] 필수 슬롯 모두 채워짐 → 5단계 진행")
    return slots


def run_stage_5_6(
    table_id: str,
    generic_slots: dict,
    table_params: dict,
    client: KosisApiClient,
    calculator: KosisCalculator,
) -> Optional[ComputedResult]:
    """5·6단계. 7·8단계로 넘길 수 있도록 ComputedResult를 반환한다 (실패/스킵 시 None)."""
    kosis_slots = build_kosis_slots(table_id, generic_slots, table_params)
    if kosis_slots is None:
        print(
            f"[5단계 api_client] '{table_id}'가 table_params.json에 없음 "
            "→ C가 아직 이 표를 조사하지 않음 (알려진 갭, 스킵)"
        )
        return None

    calc_type = generic_slots.get("calc_type")
    try:
        if calc_type in ("증감", "증감률") and kosis_slots.get("period"):
            base_slots = dict(kosis_slots, period=str(int(kosis_slots["period"]) - 1))
            base_resp = client(table_id, base_slots)
            target_resp = client(table_id, kosis_slots)
            print(f"[5단계 api_client] base   = {base_resp}")
            print(f"[5단계 api_client] target = {target_resp}")

            calc_fn = calculator.compute_change_rate if calc_type == "증감률" else calculator.compute_change
            result = calc_fn(base_resp, target_resp)
            print(f"[6단계 calculator] {result}")
            return result
        else:
            resp = client(table_id, kosis_slots)
            print(f"[5단계 api_client] {resp}")
            print("[6단계 calculator] 단순 조회 (calc_type 없음/미지원) → 계산 없이 값 그대로 사용")
            return ComputedResult(calc_type="단순조회", raw_value=resp.raw_value, unit=resp.unit, period=resp.period)
    except (KosisApiError, CalculationError) as e:
        print(f"[오류] {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"[오류] {type(e).__name__}: {e}")
        return None


def run_stage_7_8(claim, top, computed: ComputedResult) -> Optional[tuple[Verdict, Optional[Explanation]]]:
    """7단계 judge + 8단계 explain. judge가 실패하면 이 주장 전체를 스킵(None)하지만,
    explain만 실패하는 경우는 Verdict는 살리고 Explanation만 None으로 반환한다 —
    DB 저장 레이어가 evidence 필드는 비어도 verification_result는 기록할 수 있게 하기 위함."""
    try:
        verdict = judge(claim, computed)
        print(f"[7단계 judge] {verdict}")
    except JudgeError as e:
        print(f"[7단계 judge] 실패 ({type(e).__name__}: {e}) → 설명 생성 스킵")
        return None
    except Exception as e:
        print(f"[7단계 judge] 실패 ({type(e).__name__}: {e}) → 설명 생성 스킵")
        return None

    explanation: Optional[Explanation] = None
    try:
        explanation = explain(claim, top, computed, verdict)
        print(f"[8단계 explain] {explanation.explanation_text}")
        if explanation.limitation:
            print(f"[8단계 explain][한계] {explanation.limitation}")
    except ExplainerError as e:
        print(f"[8단계 explain] 실패 ({type(e).__name__}: {e})")
    except Exception as e:
        print(f"[8단계 explain] 실패 ({type(e).__name__}: {e})")

    return verdict, explanation


def _lookup_statistic_category(table_id: Optional[str], catalog_by_id: dict[str, dict]) -> Optional[str]:
    """table_catalog.json(3단계 B가 관리, 이미 있는 category 필드)을 table_id로 조회.
    새 추출 없이 기존 데이터를 재사용한다."""
    if not table_id:
        return None
    entry = catalog_by_id.get(table_id)
    return entry.get("category") if entry else None


def _build_verification_record(
    *,
    article: dict,
    claim,
    top,
    generic_slots: Optional[dict],
    table_params: dict,
    computed: Optional[ComputedResult],
    verdict: Optional[Verdict],
    explanation: Optional[Explanation],
    cls_result,
    catalog_by_id: dict,
    verification_possible: str,
    ambiguity_reason: Optional[str],
) -> dict:
    """claim + 3~8단계 결과를 db/store.py의 검증 스키마 dict로 조립한다."""
    kosis_dimension = None
    if generic_slots is not None and top is not None:
        kosis_dimension = build_kosis_slots(top.table_id, generic_slots, table_params)

    calc_type = (generic_slots or {}).get("calc_type")
    article_title = article.get("article_title") or article["label"]

    return {
        "result_id": make_result_id(article_title, claim.sentence),
        "article_title": article_title,
        "article_url": article.get("article_url"),
        "claim_sentence": claim.sentence,
        "claim_type": claim.claim_type,
        "statistic_expression": claim.statistic_expression,
        "normalized_statistic_name": _normalize_statistic_name(claim.statistic_expression),
        "statistic_category": _lookup_statistic_category(top.table_id if top else None, catalog_by_id),
        "value": claim.value,
        "unit": claim.unit,
        "comparison_operator": claim.comparison_operator,
        "comparison_target": claim.comparison_target,
        "comparison_value": claim.comparison_value,
        "time_expression": claim.period,
        "reference_time": (generic_slots or {}).get("period"),
        "population": claim.population,
        "region": claim.region,
        "source_org": claim.source_org,
        "source_report": claim.source_report,
        "kosis_table_id": top.table_id if top else None,
        "kosis_table": top.table_name if top else None,
        "kosis_item": None,  # C의 table_params.json엔 아직 사람이 읽을 항목명이 없음 (알려진 갭)
        "kosis_dimension": kosis_dimension,
        "calculation_required": calc_type in ("증감", "증감률"),
        "calculation_type": computed.calc_type if computed else None,
        "verification_possible": verification_possible,
        "ambiguity_reason": ambiguity_reason,
        "verification_result": verdict.verdict if verdict else None,
        "mismatch_reason": verdict.gap_type if verdict else None,
        "evidence": explanation.explanation_text if explanation else (verdict.reason if verdict else None),
        "classifier_score": cls_result.score if cls_result else None,
    }


def run_article(
    article: dict,
    client: KosisApiClient,
    calculator: KosisCalculator,
    table_params: dict,
    embedding_cache: dict,
    catalog_by_id: dict,
) -> list[dict]:
    """기사 하나를 1~8단계까지 돌리고, 16:30~17:00 결과 검수용 레코드 리스트를 반환한다.
    각 레코드: {article, claim_sentence, table_name, verdict, gap_type, classifier_score}."""
    results: list[dict] = []

    print(f"\n{'=' * 60}")
    print(article["label"])
    print(f"기사 원문: \"{article['article_text']}\"")
    print(f"{'-' * 60}")

    try:
        cls_result = classify(article["article_text"])
        print(f"[1단계 classifier] {cls_result}")
    except Exception as e:
        print(f"[1단계 classifier] 실패 ({type(e).__name__}: {e}) → 이 기사 스킵")
        return results

    if not cls_result.label:
        print("[1단계 classifier] 무관한 기사로 판정 → 스킵")
        return results

    try:
        claims = extract_claims(article["article_text"])
        print(f"[2단계 claim_extractor] {len(claims)}개 주장 추출")
    except Exception as e:
        print(f"[2단계 claim_extractor] 실패 ({type(e).__name__}: {e}) → 이 기사 스킵")
        return results

    for claim in claims:
        print(f"{'-' * 60}")
        print(f"주장: \"{claim.sentence}\" (claim_type={claim.claim_type})")

        if claim.claim_type == "전망":
            # 미래 예측 주장은 애초에 공식 통계로 검증할 대상이 없다 — 3~8단계를 태워봐야
            # 표 매핑도 안 되고(과거 통계표에 "전망치"가 없음) API 호출만 낭비되므로,
            # calc_type 라우팅 조사에서 드러난 이 케이스(100건 실측 중 8/84)는 여기서
            # 바로 판단불가로 끝낸다 (calc_type_router.route_calc_type도 "전망"은 None을
            # 반환해 같은 원칙을 공유함).
            print("[분류] claim_type='전망'(미래 예측) → 3~8단계 건너뛰고 즉시 판단불가 처리")
            verdict = Verdict(verdict="판단불가", gap_type=None, reason="미래 예측 주장은 공식 통계로 검증 불가")
            insert_verification(
                _build_verification_record(
                    article=article, claim=claim, top=None, generic_slots=None,
                    table_params=table_params, computed=None, verdict=verdict, explanation=None,
                    cls_result=cls_result, catalog_by_id=catalog_by_id,
                    verification_possible="불가", ambiguity_reason="미래 예측 주장(claim_type=전망)은 검증 대상 아님",
                )
            )
            results.append(
                {
                    "article": article["label"],
                    "claim_sentence": claim.sentence,
                    "table_name": None,
                    "verdict": "판단불가",
                    "gap_type": None,
                    "classifier_score": cls_result.score,
                }
            )
            continue

        try:
            candidates = search_and_rerank(
                claim,
                keyword_fn=keyword_search,
                embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
                document_texts={tid: t["embedding_text"] for tid, t in catalog_by_id.items()},
            )
        except Exception as e:
            print(f"[3단계 매핑] 실패 ({type(e).__name__}: {e}) → 이 주장 스킵")
            continue

        if not candidates:
            print("[3단계 매핑] 매칭되는 표 없음 → 스킵")
            continue

        top = candidates[0]
        print(f"[3단계 매핑] 최상위 후보: {top.table_name} ({top.table_id}) score={top.score:.3f}")

        # 안전장치: keyword_search가 못 찾아서 embedding_search만으로 나온 후보는
        # source_meta에 "unverified"로 표시된다(reranker.py의 _merge_candidates 참고).
        # 지금 embedding_search는 아직 실제 임베딩 API가 아니라 해시 기반 더미라 노이즈에
        # 가까운데, 이 노이즈가 특정 표(예: DT_200Y102)로 구조적으로 쏠려서 전혀 무관한
        # 주장도 그럴듯한 score로 매칭해버리는 문제가 실제 배치 실행에서 확인됨. 검증 안
        # 된 매칭을 억지로 쓰지 않고 "매칭 없음"으로 처리한다.
        if top.source_meta and "unverified" in top.source_meta:
            print(f"[3단계 매핑] 최상위 후보가 검증 안 된 임베딩 전용 매칭(신뢰도 낮음) → 매칭 없음으로 처리")
            insert_verification(
                _build_verification_record(
                    article=article, claim=claim, top=top, generic_slots=None,
                    table_params=table_params, computed=None, verdict=None, explanation=None,
                    cls_result=cls_result, catalog_by_id=catalog_by_id,
                    verification_possible="애매",
                    ambiguity_reason="표 매칭 신뢰도가 낮아 검증 안 된 임베딩 전용 매칭임",
                )
            )
            results.append(
                {
                    "article": article["label"],
                    "claim_sentence": claim.sentence,
                    "table_name": top.table_name,
                    "verdict": "표매칭_불충분",
                    "gap_type": None,
                    "classifier_score": cls_result.score,
                }
            )
            continue

        try:
            slots = run_stage_4(claim.sentence, article.get("clarify_reply"), article["published_date"])
        except Exception as e:
            print(f"[4단계 slot_filler] 실패 ({type(e).__name__}: {e}) → 이 주장 스킵")
            continue
        if slots is None:
            continue

        computed = run_stage_5_6(top.table_id, slots, table_params, client, calculator)
        if computed is None:
            continue

        outcome = run_stage_7_8(claim, top, computed)
        if outcome is not None:
            verdict, explanation = outcome
            insert_verification(
                _build_verification_record(
                    article=article, claim=claim, top=top, generic_slots=slots,
                    table_params=table_params, computed=computed, verdict=verdict,
                    explanation=explanation, cls_result=cls_result, catalog_by_id=catalog_by_id,
                    verification_possible="가능", ambiguity_reason=None,
                )
            )
            results.append(
                {
                    "article": article["label"],
                    "claim_sentence": claim.sentence,
                    "table_name": top.table_name,
                    "verdict": verdict.verdict,
                    "gap_type": verdict.gap_type,
                    "classifier_score": cls_result.score,
                }
            )

    return results


def print_review_summary(results: list[dict]) -> None:
    """16:30~17:00 결과 검수: verdict 분포 확인 + 애매한 구간을 사람 리뷰 큐로 필터링.

    "표매칭_불충분"(3단계 신뢰도 낮은 매칭 안전장치, run_article 참고)은 진짜 판정이
    아니라서 verdict 분포(일치/불일치/판단불가)에는 안 넣고, 리뷰 큐에서 별도로 센다.
    """
    print(f"\n{'=' * 60}")
    print("결과 검수 (16:30~17:00)")
    print(f"{'=' * 60}")

    total = len(results)
    print(f"\n1~8단계 파이프라인이 처리한 주장: {total}건")
    if total == 0:
        print("(처리된 주장이 없어 분포/리뷰 큐를 만들 수 없음)")
        return

    judged = [r for r in results if r["verdict"] != "표매칭_불충분"]
    low_confidence_match = [r for r in results if r["verdict"] == "표매칭_불충분"]

    verdict_counts: dict[str, int] = {}
    for r in judged:
        verdict_counts[r["verdict"]] = verdict_counts.get(r["verdict"], 0) + 1

    print(f"\n[verdict 분포] (실제 판정까지 도달한 {len(judged)}건 기준)")
    for v in ("일치", "불일치", "판단불가"):
        n = verdict_counts.get(v, 0)
        pct = n / len(judged) * 100 if judged else 0.0
        print(f"  {v}: {n}건 ({pct:.1f}%)")
    print(f"  (표매칭 신뢰도 낮아 판정 자체를 안 한 건: {len(low_confidence_match)}건 — 아래 리뷰 큐 참고)")

    # 사람 리뷰 큐: (a) classifier score 0.4~0.6 애매 구간, (b) verdict=판단불가,
    # (c) 표매칭 신뢰도 낮음(3단계가 검증 안 된 임베딩 전용 매칭이라 판정을 안 한 경우)
    review_queue = [
        r
        for r in results
        if r["verdict"] in ("판단불가", "표매칭_불충분") or 0.4 <= r["classifier_score"] <= 0.6
    ]
    print(
        f"\n[사람 리뷰 큐] {len(review_queue)}건 "
        "(판단불가 / 표매칭 신뢰도 낮음 / classifier score 0.4~0.6)"
    )
    for r in review_queue:
        print(
            f"  - [{r['article']}] \"{r['claim_sentence']}\" "
            f"→ {r['verdict']} (score={r['classifier_score']:.2f}, gap_type={r['gap_type']})"
        )


def main(use_csv_sample: bool = False, csv_n: int = 15) -> None:
    try:
        client = KosisApiClient()
    except RuntimeError as e:
        print(f"[중단] {e}")
        return

    calculator = KosisCalculator()

    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)

    catalog_by_id = _load_table_catalog_by_id()
    embedding_cache = build_table_embedding_cache()

    articles = load_articles_from_csv(n=csv_n) if use_csv_sample else ARTICLES

    all_results: list[dict] = []
    for article in articles:
        all_results.extend(
            run_article(article, client, calculator, table_params, embedding_cache, catalog_by_id)
        )

    print_review_summary(all_results)
    print(f"\n[DB] {len(all_results)}건을 data/verifications.db에 저장했습니다.")


if __name__ == "__main__":
    main(use_csv_sample="--csv" in sys.argv)
