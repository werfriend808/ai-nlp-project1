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
import io
import json
import random
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional

from agent.preprocessing.classifier import classify
from agent.preprocessing.claim_extractor import extract_claims, recover_missed_claims
from agent.preprocessing.source_filter import resolve_claim_sources, filter_verifiable_claims
from agent.mapping.keyword_search import SYNONYMS, keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import search_and_rerank
from agent.orchestrator.slot_filler import fill_slots
from agent.orchestrator.clarify import clarify
from agent.orchestrator.calc_type_router import _mentions_foreign_country, route_calc_type
from agent.shared.extreme_value_patterns import (
    ALL_TIME_RE,
    resolve_n_years_since_start_year,
    resolve_since_event_start_year,
)
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
    # data_set.csv는 여러 차례 Excel에서 내보낸 조각을 이어붙인 흔적이 있어서, 파일 맨 앞뿐
    # 아니라 중간 행에도 BOM(U+FEFF)이 섞여 있는 경우가 실측 확인됨(2026-08-10, --csv 배치
    # 실행 중 재현 — 기사제목에 낀 BOM을 print()가 그대로 출력하려다 Windows 콘솔(cp949)
    # 인코딩 실패로 크래시). encoding="utf-8-sig"는 파일 시작 위치의 BOM만 제거하므로, 읽은
    # 텍스트 전체에서 남은 BOM을 한 번 더 제거해서 중간에 낀 것까지 잡는다.
    with open(path, encoding="utf-8-sig") as f:
        text = f.read().replace("﻿", "")
    rows = [
        r for r in csv.DictReader(io.StringIO(text))
        if r.get("검색 구분 레이블", "").strip().lower() == "true"
    ]

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
    if generic_slots.get("prd_se"):
        # 2026-08-13: 4단계(run_stage_4)가 표의 지원 주기 목록 중에서 이미 claim에 맞는
        # 걸 골라뒀으면(prd_se), 그대로 실어서 KosisApiClient까지 전달한다 — 안 넘기면
        # api_client.py가 표의 기본 주기(_default_prd_se)로 알아서 폴백한다(하위 호환).
        kosis_slots["prd_se"] = generic_slots["prd_se"]

    for dim_name, dim in base.get("dimensions", {}).items():
        # 이 표에 정의된 축(dim_name)만 채운다. generic_slots에 값이 있으면 쓰고,
        # 없으면 표의 default_value로 채운다 (예: region이 없는 표는 gender/age만 봄).
        value = generic_slots.get(dim_name)
        kosis_slots[dim_name] = value if value is not None else dim.get("default_value")

    return kosis_slots


def _infer_desired_granularity(text: Optional[str]) -> Optional[str]:
    """claim 문장에서 이 주장이 월/분기/연 중 어느 주기를 가리키는지 best-effort로 추정한다.

    3단계(prd_se 선택 로직)의 핵심 — 2026-08-13, 64개 표 중 41개(64%)가 한 표에서 여러
    주기를 동시에 지원한다는 게 실측 확인돼서, "표가 지원하는 것 중 claim이 실제로 원하는
    주기"를 골라야 한다. "분기"가 명시되면 최우선(월/연보다 구체적인 표현), 그 다음
    월/달 표현, 나머지는 판단 보류(None) — 상위 호출부가 표 주기 목록에서 기본값을 쓴다."""
    if not text:
        return None
    if "분기" in text:
        return "Q"
    if "월" in text or "달" in text:
        return "M"
    return None


def _select_prd_se(supported: object, desired: Optional[str]) -> Optional[str]:
    """표가 지원하는 주기 목록(supported)과 claim이 원하는 주기(desired)를 맞춰서 실제로
    쓸 prd_se 하나를 고른다.

    - supported가 아직 마이그레이션 전(문자열)이면 그 값 그대로 반환(하위 호환)
    - desired가 목록에 있으면 그대로 사용 (claim이 원하는 걸 표가 지원하는 가장 좋은 경우)
    - 없으면 "Y"를 우선(옛날 기본값과 동일한 안전한 폴백), 그것도 없으면 목록의 첫 번째
    - 표가 뭘 지원하는지 아예 모르면(등록 안 된 표 등) None — fill_slots가 기존처럼 연
      단위로만 동작하게 된다."""
    if isinstance(supported, str):
        return supported
    if not supported:
        return None
    if desired and desired in supported:
        return desired
    return "Y" if "Y" in supported else supported[0]


def run_stage_4(
    claim_sentence: str,
    clarify_reply: Optional[str],
    article_date: date,
    *,
    table_id: Optional[str] = None,
    table_params: Optional[dict] = None,
) -> Optional[dict]:
    """4단계: fill_slots + clarify. 한 번에 안 채워지면 clarify_reply로 한 번 더 시도.
    그래도 부족하면 None (되묻기 미해결 → 5단계로 못 감)을 반환한다.

    table_id/table_params: 3단계가 이미 매칭한 표의 지원 주기 목록에서, claim 문장이
    실제로 원하는 주기(_infer_desired_granularity)와 맞춰 prd_se 하나를 고른다(2026-08-13,
    다중 주기 지원 작업 — 표 주기를 몰라 무조건 연도만 채우던 문제(2026-08-12)에 이어,
    표가 여러 주기를 지원해도 claim이 원하는 걸 못 고르던 문제까지 해결). 안 넘기면 기존과
    동일하게 연 단위로만 동작한다(하위 호환).

    반환하는 slots에 "prd_se"를 실제로 채워 넣는다 — build_kosis_slots가 이 값을
    KosisApiClient까지 그대로 전달해서, 표의 기본값이 아니라 여기서 고른 주기로 조회하게
    한다."""
    prd_se = None
    if table_id and table_params and table_id in table_params:
        supported = table_params[table_id].get("prdSe")
        desired = _infer_desired_granularity(claim_sentence)
        prd_se = _select_prd_se(supported, desired)

    slots = fill_slots(claim_sentence, {}, article_date, prd_se=prd_se)
    if prd_se:
        slots["prd_se"] = prd_se
    question = clarify(slots)
    print(f"[4단계 slot_filler] 1차 슬롯: {slots}")

    if question and clarify_reply:
        print(f"[4단계 clarify] 되묻기: \"{question}\" → (준비된 답변) \"{clarify_reply}\"")
        slots = fill_slots(clarify_reply, slots, article_date, prd_se=prd_se)
        if prd_se:
            slots["prd_se"] = prd_se
        question = clarify(slots)
        print(f"[4단계 slot_filler] 2차 슬롯: {slots}")

    if question:
        print(f"[4단계 clarify] 여전히 부족 → 되묻기: \"{question}\" (여기서 중단)")
        return None

    print("[4단계 clarify] 필수 슬롯 모두 채워짐 → 5단계 진행")
    return slots


def _prior_year_same_period(period: str) -> str:
    """"전년동월/동분기" 기준값(base)의 시점을 계산한다 — target 시점에서 연도만 1 빼고
    나머지(월/분기 코드)는 그대로 유지한다.

    2026-08-13 실측 발견: 예전엔 무조건 `str(int(period) - 1)`로 계산했는데, 이건 4자리
    연도("2024"->"2023")에만 맞는 계산이다. 6자리 월/분기 period가 도입된 뒤에도 이 로직을
    그대로 써서 "202501"(2025년 1분기) - 1 = "202500"이라는 존재하지 않는 시점이 나오고
    있었다 — 실제로는 "202401"(전년 동분기)이 나와야 하는데, 정수로 통째로 1을 빼는 바람에
    분기/월 부분이 아니라 연도 끝자리가 깎여서 인접한 다른 시점(예: 2024년 5~6월)과
    비교되는 사례로 재현됨("코스피 3000..." 기사 판정 오류 원인 추적 중 발견)."""
    if len(period) == 6:
        year, suffix = period[:4], period[4:]
        return f"{int(year) - 1:04d}{suffix}"
    return str(int(period) - 1)


# "전달/전월/전분기" 계열(직전 주기 비교, MoM/QoQ)인지 "작년/전년/지난해" 계열(전년동시점
# 비교, YoY)인지 판단하는 키워드. "작년" 같은 키워드는 그 자체로 공백이 없는 한 덩어리라
# comparison_target에 "작년대비"든 "작년 대비"든 상관없이 `in` 부분 문자열 검사로 그대로
# 걸린다 — 별도 공백 제거 전처리가 필요 없다.
_PRIOR_IMMEDIATE_KEYWORDS = ("전달", "전월", "전분기", "직전")
_PRIOR_YEAR_KEYWORDS = ("작년", "전년", "지난해")

# 2026-08-13 실측 발견: claim_extractor(LLM)가 기사 전체 맥락을 보고 "전달"을 이미 "3월"처럼
# 구체적인 절대 월/분기로 바꿔서 comparison_target에 채우는 경우가 실제로 더 흔했다("트리플
# 감소" 기사 8개 claim 중 다수가 "3월"로 나옴, "전달"이라는 원문 그대로는 하나도 안 나옴).
# 이런 "숫자+월/분기"만 있고 "작년/전년" 같은 연도 지시어가 없는 표현은, 기사가 다루는 시점
# 바로 직전 주기를 가리키는 게 거의 확실하므로(같은 해 안에서의 인접 비교) MoM/QoQ로 처리한다.
_BARE_MONTH_OR_QUARTER_RE = re.compile(r"^\d{1,2}(월|분기)$")


def _wants_prior_immediate_period(comparison_target: Optional[str]) -> bool:
    """claim.comparison_target 원문 표현이 "직전 주기 대비"(MoM/QoQ)를 가리키면 True.

    "작년"/"전년"/"지난해"가 포함돼 있으면(예: "작년 3월") 그 앞에 다른 신호가 있어도
    명시적 YoY로 우선 처리한다 — 절대 월 표현("3월")과 섞여 나올 수 있어(예: "지난해 11월")
    안전하게 먼저 걸러낸다."""
    if not comparison_target:
        return False
    stripped = comparison_target.strip()
    if any(kw in stripped for kw in _PRIOR_YEAR_KEYWORDS):
        return False
    if any(kw in stripped for kw in _PRIOR_IMMEDIATE_KEYWORDS):
        return True
    return bool(_BARE_MONTH_OR_QUARTER_RE.match(stripped))


def _prior_immediate_period(period: str, prd_se: Optional[str]) -> str:
    """"전달/전분기 대비"(MoM/QoQ) 기준값(base)의 시점을 계산한다 — target 시점에서
    월/분기를 1만큼 줄이고, 연초(1월/1분기)를 넘어가면 연도를 1 줄인다.

    2026-08-13 실측 발견: run_stage_5_6이 calc_type이 "증감"/"증감률"이기만 하면 comparison_target
    을 전혀 안 보고 무조건 _prior_year_same_period(전년동시점, YoY)로 base를 계산했다 — 그래서
    기사가 "전달 대비"(MoM)라고 명시해도 항상 "작년 같은 달"과 비교돼버려서, period 자체는
    정확히 뽑혔는데도(예: 202504) base가 202404(1년 전)로 계산되는 바람에 판정이 계속
    틀렸다("트리플 감소" 기사 재검증 중 발견 — claim 8개 중 다수가 "전달 대비"인데도 전부
    YoY로 계산되고 있었음).

    prd_se가 M/Q가 아니거나(예: Y) period가 6자리가 아니면 "직전 주기"라는 개념 자체가
    전년동시점과 구분이 안 되므로(연간 표에서 "전달"은 의미가 없음) 안전하게
    _prior_year_same_period로 폴백한다."""
    if len(period) == 6 and prd_se in ("M", "Q"):
        year, suffix = int(period[:4]), int(period[4:])
        max_suffix = 12 if prd_se == "M" else 4
        if suffix <= 1:
            return f"{year - 1:04d}{max_suffix:02d}"
        return f"{year:04d}{suffix - 1:02d}"
    return _prior_year_same_period(period)


def run_stage_5_6(
    table_id: str,
    generic_slots: dict,
    table_params: dict,
    client: KosisApiClient,
    calculator: KosisCalculator,
    *,
    comparison_target: Optional[str] = None,
    claim_sentence: Optional[str] = None,
    article_year: Optional[int] = None,
) -> Optional[ComputedResult]:
    """5·6단계. 7·8단계로 넘길 수 있도록 ComputedResult를 반환한다 (실패/스킵 시 None).

    comparison_target: claim_extractor가 뽑아둔 원문 비교 기준 표현(예: "전달", "작년",
    "전년동월"). "전달/전월/전분기" 계열이면 직전 주기(MoM/QoQ)를, 그 외(명시 없음 포함)는
    기존처럼 전년동시점(YoY)을 base로 계산한다(2026-08-13, _wants_prior_immediate_period
    참고) — 지정 안 하면(기본값 None) 항상 YoY라 하위 호환된다.

    claim_sentence/article_year: calc_type이 "최댓값검증"/"최솟값검증"일 때 극값 시작
    연도를 원문에서 계산하기 위해 필요(route_calc_type이 이 calc_type을 정해서
    generic_slots["calc_type"]에 덮어쓴 뒤 호출부가 넘겨준다) — 팀원(D)이 다른 브랜치에서
    만든 극값검증 기능을 이 브랜치의 오늘 자 수정사항(다중주기/MoM-YoY 구분)과 함께
    합친 것(2026-08-14)."""
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
            if _wants_prior_immediate_period(comparison_target):
                base_period = _prior_immediate_period(kosis_slots["period"], generic_slots.get("prd_se"))
            else:
                base_period = _prior_year_same_period(kosis_slots["period"])
            base_slots = dict(kosis_slots, period=base_period)
            base_resp = client(table_id, base_slots)
            target_resp = client(table_id, kosis_slots)
            print(f"[5단계 api_client] base   = {base_resp}")
            print(f"[5단계 api_client] target = {target_resp}")

            calc_fn = calculator.compute_change_rate if calc_type == "증감률" else calculator.compute_change
            result = calc_fn(base_resp, target_resp)
            print(f"[6단계 calculator] {result}")
            return result
        elif calc_type in ("최댓값검증", "최솟값검증") and kosis_slots.get("period"):
            start_year = resolve_since_event_start_year(claim_sentence or "")
            if start_year is None:
                start_year = resolve_n_years_since_start_year(claim_sentence or "", article_year)
            if start_year is None:
                if not ALL_TIME_RE.search(claim_sentence or ""):
                    print(
                        f"[5단계 api_client] calc_type={calc_type!r}인데 시작 연도 패턴("
                        "코로나 이후/N년 만에/역대)을 문장에서 못 찾음 → 스킵"
                    )
                    return None
                # "역대"(ALL_TIME_RE)는 기준 시점 자체가 없는 표현이라 문장만으론 시작
                # 연도를 못 구한다. agent_chat.py의 resolve_max_all_time_responses()가
                # 이미 팀 합의로 채택한 관행(2026-08-06)을 그대로 따른다: 넉넉히 이른
                # 연도(1960)부터 요청해도 KOSIS가 실제 데이터 있는 시점부터만 돌려주므로
                # (실측: DT_1DA7102S에 1999년부터 요청해도 2000년부터 응답) 표별 정확한
                # 최소 연도를 몰라도 안전하다 — table_params.json에 아직 표별 최소 연도가
                # 없어서 그쪽 기준은 못 쓴다.
                start_year = 1960

            historical = client.fetch_series(table_id, kosis_slots, str(start_year), str(article_year))
            current_resp = client(table_id, kosis_slots)
            print(f"[5단계 api_client] current    = {current_resp}")
            print(f"[5단계 api_client] historical({start_year}~{article_year}) = {len(historical)}건")

            check_fn = calculator.compute_max_check if calc_type == "최댓값검증" else calculator.compute_min_check
            result = check_fn(current_resp, historical)
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


def run_stage_7_8(
    claim, top, computed: ComputedResult, *, prd_se: Optional[str] = None
) -> Optional[tuple[Verdict, Optional[Explanation]]]:
    """7단계 judge + 8단계 explain. judge가 실패하면 이 주장 전체를 스킵(None)하지만,
    explain만 실패하는 경우는 Verdict는 살리고 Explanation만 None으로 반환한다 —
    DB 저장 레이어가 evidence 필드는 비어도 verification_result는 기록할 수 있게 하기 위함.

    prd_se: 4단계(run_stage_4)가 이미 표의 지원 주기 목록 중에서 골라둔 값을 그대로
    받는다(2026-08-13) — computed.period("202402" 등)가 월인지 분기인지 판정/설명
    프롬프트가 헷갈리지 않게 하기 위함(2026-08-12에 처음 도입). table_params에서 표 하나당
    prdSe 하나였던 옛날엔 여기서 직접 조회해도 됐지만, 이제 prdSe가 목록이라 "실제로 어떤
    주기로 조회했는지"는 4단계의 선택 결과를 그대로 받아야만 정확하다 — table_params를
    다시 조회하면 목록 전체(예: ['Y','Q','M'])를 받아 어느 것도 특정 못 한다. 안 넘기면
    기존과 동일하게 원본 period 문자열 그대로 노출."""
    try:
        verdict = judge(claim, computed, prd_se=prd_se)
        print(f"[7단계 judge] {verdict}")
    except JudgeError as e:
        print(f"[7단계 judge] 실패 ({type(e).__name__}: {e}) → 설명 생성 스킵")
        return None
    except Exception as e:
        print(f"[7단계 judge] 실패 ({type(e).__name__}: {e}) → 설명 생성 스킵")
        return None

    explanation: Optional[Explanation] = None
    try:
        explanation = explain(claim, top, computed, verdict, prd_se=prd_se)
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
        "calculation_required": calc_type in ("증감", "증감률", "최댓값검증", "최솟값검증"),
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
        claims = recover_missed_claims(article["article_text"], claims)
        print(f"[2단계 claim_extractor] {len(claims)}개 주장 추출")
    except Exception as e:
        print(f"[2단계 claim_extractor] 실패 ({type(e).__name__}: {e}) → 이 기사 스킵")
        return results

    # source_filter(2단계 이후 출처 검증 필터) — 실제로 KOSIS 국가승인통계를 생산하는
    # 기관이 출처인 claim만 3단계(표매칭)로 넘긴다. 이 필터는 이미 만들어져 있었는데
    # batch_runner.py에 연결이 안 돼 있어서, 상하이모터쇼 참가 기업 수·유튜버 수퍼챗 수입처럼
    # 출처가 아예 없거나(source_org=None) 개인/기업발 정보인 claim이 그대로 3~8단계를
    # 다 태우고 있었다(2026-08-13 실측 확인). 필터링 전/후 건수를 로그로 남긴다.
    #
    # classifier_reason(2번째 인자)을 다시 넘긴다 — claim_extractor(LLM)가 source_org를
    # 아예 못 채우는 경우가 실측으로 훨씬 흔했다(2026-08-13, --csv 15건 재실행에서 25개
    # claim 전부가 이 이유로 걸러짐 — 과기정통부가 기사 첫 문장 주어인 SKT 유심 기사조차
    # source_org=None으로 뽑힘). 한때 "국세청 관계자는 ~라고 말했다"처럼 기관명이 일반
    # 인용문에 스쳐 지나가는 오탐(유튜버 수퍼챗 수입 기사) 때문에 이 폴백을 통째로 껐었는데,
    # 대신 infer_org_from_reason 자체에 "발표/집계/공표/통계/조사"류 키워드가 reason에
    # 같이 있어야만 발동하는 안전장치를 달아서 재활성화했다(source_filter.py 참고) —
    # 통계 발표 reason은 복구하고, 일반 인용문 오탐은 계속 차단.
    claims = resolve_claim_sources(claims, cls_result.reason)
    before_filter = len(claims)
    claims = filter_verifiable_claims(claims)
    if before_filter != len(claims):
        print(f"[2단계 출처 필터] {before_filter}개 중 {before_filter - len(claims)}개 제외 (KOSIS 미검증 출처)")

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

        if _mentions_foreign_country(claim.population, claim.region, claim.comparison_target):
            # KOSIS는 대한민국 통계청 산하 포털이라 해외 국가 자체 통계(중국 GDP, 일본
            # 1인당 국민소득 등)를 원천적으로 제공하지 않는다. 근데 3단계 매칭이 "GDP"/
            # "물가" 같은 일반 키워드만 보고 국내 표로 잘못 매칭시켜서, 존재하지도 않는
            # 비교를 억지로 계산하다 잘못된 판정이 나오는 사례가 실측 확인됐다(2026-08-12
            # — 한국 vs 일본 1인당 국민소득 비교, 중국 1분기 GDP 등). 전망 claim과 같은
            # 원칙으로 3~8단계를 아예 건너뛰고 정직하게 판단불가 처리한다.
            print("[분류] population/region/comparison_target에 해외 국가 포함 → KOSIS 검증 불가, 즉시 판단불가 처리")
            verdict = Verdict(verdict="판단불가", gap_type=None, reason="해외 국가/지역 통계는 KOSIS(국내 통계)로 검증 불가")
            insert_verification(
                _build_verification_record(
                    article=article, claim=claim, top=None, generic_slots=None,
                    table_params=table_params, computed=None, verdict=verdict, explanation=None,
                    cls_result=cls_result, catalog_by_id=catalog_by_id,
                    verification_possible="불가", ambiguity_reason="해외 국가/지역 데이터는 KOSIS 검증 대상 아님",
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
        result = _finish_claim_with_top_candidate(
            article, claim, cls_result, top, table_params, client, calculator, catalog_by_id
        )
        if result is not None:
            results.append(result)

    return results


def _finish_claim_with_top_candidate(
    article: dict,
    claim,
    cls_result,
    top,
    table_params: dict,
    client: KosisApiClient,
    calculator: KosisCalculator,
    catalog_by_id: dict,
) -> Optional[dict]:
    """3단계(표매칭+리랭킹)에서 최종 top 후보가 이미 정해진 뒤, 4~8단계를 마저 실행하고
    DB에 저장한 뒤 결과 레코드(dict) 하나를 반환한다 (저장할 게 없으면 None).

    2026-08-13: run_article의 리랭킹 이후 로직을 그대로 뽑아온 것 — 로컬 RAM으로 리랭커
    모델(bge-reranker-v2-m3, 568M)을 못 돌려서 코랩에서 리랭킹만 대신 실행하는 흐름
    (export_for_rerank.py → 코랩 노트북 → resume_after_rerank.py)이 필요해졌는데, 리랭킹
    "이후" 단계(4~8단계+DB저장)는 정상 경로(run_article)와 코랩 경로(resume_after_rerank.py)
    둘 다 완전히 똑같이 해야 하므로 공유 함수로 뽑았다. run_article 자체의 동작은 이 리팩토링
    전후로 동일하다(로직 이동만, 변경 없음)."""
    print(f"[3단계 매핑] 최상위 후보: {top.table_name} ({top.table_id}) score={top.score:.3f}")

    # 안전장치: keyword_search가 못 찾아서 embedding_search만으로 나온 후보는
    # source_meta에 "unverified"로 표시된다(reranker.py의 _merge_candidates 참고).
    # 임베딩 코사인 유사도만으로는(리랭커 없이는) 노이즈와 진짜 신호를 구분하기 어렵다는 게
    # 실측으로 재확인됐다(2026-08-13, 완전 무관한 문장도 진짜 매칭과 비슷한 점수대가 나옴) —
    # 검증 안 된 매칭을 억지로 쓰지 않고 "매칭 없음"으로 처리한다.
    if top.source_meta and "unverified" in top.source_meta:
        print(f"[3단계 매핑] 최상위 후보가 검증 안 된 임베딩 전용 매칭(신뢰도 낮음) → 매칭 없음으로 처리")
        try:
            insert_verification(
                _build_verification_record(
                    article=article, claim=claim, top=top, generic_slots=None,
                    table_params=table_params, computed=None, verdict=None, explanation=None,
                    cls_result=cls_result, catalog_by_id=catalog_by_id,
                    verification_possible="애매",
                    ambiguity_reason="표 매칭 신뢰도가 낮아 검증 안 된 임베딩 전용 매칭임",
                )
            )
        except Exception as e:
            print(f"[DB 저장] 실패 ({type(e).__name__}: {e}) → 저장만 스킵, 배치는 계속")
        return {
            "article": article["label"],
            "claim_sentence": claim.sentence,
            "table_name": top.table_name,
            "verdict": "표매칭_불충분",
            "gap_type": None,
            "classifier_score": cls_result.score,
        }

    try:
        slots = run_stage_4(
            claim.sentence,
            article.get("clarify_reply"),
            article["published_date"],
            table_id=top.table_id,
            table_params=table_params,
        )
    except Exception as e:
        print(f"[4단계 slot_filler] 실패 ({type(e).__name__}: {e}) → 이 주장 스킵")
        return None
    if slots is None:
        return None

    # calc_type_router.route_calc_type()이 claim_type + claim.sentence 규칙만으로
    # calc_type을 결정한다 — 4단계 LLM이 slot_filler에서 추측한 slots["calc_type"]보다
    # 우선한다(팀원 D 실측: "청년실업률, 코로나 이후 최고" claim이 LLM 추측으로는 "증감률"로만
    # 채워져서 "작년 대비 증감률"로 잘못 계산되고, "7% 중반 대 0.0%"라는 무의미한 비교로
    # 오판정되는 사례가 확인됨). None이면 claim_type="전망"이거나 해외 국가 포함(이미 위에서
    # 걸렀어야 하는 케이스의 안전망) 등 규칙 라우팅 불가라는 뜻이라, 5~8단계를 건너뛰고
    # 즉시 판단불가 처리한다(2026-08-14, 팀원 D의 브랜치 작업을 오늘 자 수정사항과 병합).
    routed_calc_type = route_calc_type(claim)
    if routed_calc_type is None:
        print(
            f"[calc_type 라우팅] route_calc_type()이 None → claim_type={claim.claim_type!r} "
            "규칙 기반 라우팅 불가 → 5~8단계 건너뛰고 즉시 판단불가 처리"
        )
        verdict = Verdict(
            verdict="판단불가", gap_type=None,
            reason="calc_type 규칙 기반 라우팅 불가(claim_type=전망 또는 해외 국가 포함 등)",
        )
        try:
            insert_verification(
                _build_verification_record(
                    article=article, claim=claim, top=top, generic_slots=slots,
                    table_params=table_params, computed=None, verdict=verdict, explanation=None,
                    cls_result=cls_result, catalog_by_id=catalog_by_id,
                    verification_possible="불가",
                    ambiguity_reason="calc_type_router.route_calc_type()이 None을 반환(claim_type=전망 등)",
                )
            )
        except Exception as e:
            print(f"[DB 저장] 실패 ({type(e).__name__}: {e}) → 저장만 스킵, 배치는 계속")
        return {
            "article": article["label"],
            "claim_sentence": claim.sentence,
            "table_name": top.table_name,
            "verdict": "판단불가",
            "gap_type": None,
            "classifier_score": cls_result.score,
        }

    print(f"[calc_type 라우팅] LLM 추정값({slots.get('calc_type')!r}) 대신 규칙 기반 결과로 덮어씀 → {routed_calc_type!r}")
    slots["calc_type"] = routed_calc_type

    computed = run_stage_5_6(
        top.table_id, slots, table_params, client, calculator,
        comparison_target=claim.comparison_target,
        claim_sentence=claim.sentence, article_year=article["published_date"].year,
    )
    if computed is None:
        return None

    outcome = run_stage_7_8(claim, top, computed, prd_se=slots.get("prd_se"))
    if outcome is not None:
        verdict, explanation = outcome
        try:
            insert_verification(
                _build_verification_record(
                    article=article, claim=claim, top=top, generic_slots=slots,
                    table_params=table_params, computed=computed, verdict=verdict,
                    explanation=explanation, cls_result=cls_result, catalog_by_id=catalog_by_id,
                    verification_possible="가능", ambiguity_reason=None,
                )
            )
        except Exception as e:
            print(f"[DB 저장] 실패 ({type(e).__name__}: {e}) → 저장만 스킵, 배치는 계속")
        return {
            "article": article["label"],
            "claim_sentence": claim.sentence,
            "table_name": top.table_name,
            "verdict": verdict.verdict,
            "gap_type": verdict.gap_type,
            "classifier_score": cls_result.score,
        }
    return None


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


def main(use_csv_sample: bool = False, csv_n: int = 15, csv_seed: int = 42) -> None:
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

    articles = load_articles_from_csv(n=csv_n, seed=csv_seed) if use_csv_sample else ARTICLES

    all_results: list[dict] = []
    for article in articles:
        all_results.extend(
            run_article(article, client, calculator, table_params, embedding_cache, catalog_by_id)
        )

    print_review_summary(all_results)
    print(f"\n[DB] {len(all_results)}건을 data/verifications.db에 저장했습니다.")


def _parse_int_flag(argv: list[str], flag: str, default: int) -> int:
    """"{flag} 30" / "{flag}=30" 두 형식 다 지원하는 정수 CLI 인자 파서."""
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith(f"{flag}="):
            return int(arg.split("=", 1)[1])
    return default


def _parse_csv_n(argv: list[str], default: int = 15) -> int:
    """--csv-n 30 / --csv-n=30 형식으로 CSV 샘플 건수를 지정할 수 있게 한다.
    (프론트엔드 데모용 데이터 export를 위해 15건 고정값을 CLI에서 조절 가능하게 함 —
    1~8단계 전체는 judge/explain의 LLM 호출까지 있어 pipeline_1_4.py보다 건당 오래
    걸리고 병렬화/재시작 기능도 아직 없으니, 너무 큰 값은 권장하지 않는다.)"""
    return _parse_int_flag(argv, "--csv-n", default)


def _parse_csv_seed(argv: list[str], default: int = 42) -> int:
    """--csv-seed 7 / --csv-seed=7 형식으로 random.Random 시드를 지정한다.
    load_articles_from_csv가 시드를 안 바꾸면 항상 같은 42로 고정돼서, --csv-n을
    반복 실행해도 매번 같은 기사만 뽑힌다 — 다른 기사 조합을 보고 싶으면 시드를
    바꿔야 한다 (같은 시드+더 큰 csv_n은 기존 샘플을 그대로 포함해서 확장하지만,
    시드를 바꾸면 완전히 다른 무작위 샘플이 뽑힌다)."""
    return _parse_int_flag(argv, "--csv-seed", default)


if __name__ == "__main__":
    # Windows 콘솔 기본 인코딩(cp949)은 기사 원문에 섞인 일부 유니코드 문자(예: 특수 구두점)를
    # 인코딩 못 해 print()에서 UnicodeEncodeError로 배치 전체가 죽는다 — 진행 로그 출력용
    # stdout/stderr만 UTF-8로 바꿔서 막는다.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main(
        use_csv_sample="--csv" in sys.argv,
        csv_n=_parse_csv_n(sys.argv),
        csv_seed=_parse_csv_seed(sys.argv),
    )
