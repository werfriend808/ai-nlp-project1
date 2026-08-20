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
from agent.mapping.keyword_search import SYNONYMS, keyword_search, _kiwi
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import search_and_rerank, is_rrf_trusted
from agent.orchestrator.slot_filler import fill_slots, call_hcx, extract_json_fallback, is_region_grounded
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
from agent.interfaces import ComputedResult, Explanation, Verdict, KosisApiResponse
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


def _table_required_slots(table_id: Optional[str], catalog_by_id: dict) -> list[str]:
    """table_catalog.json의 표별 required_slots/optional_slots(한글 라벨)를 보고, 4단계
    되묻기가 실제로 다뤄야 할 슬롯 목록(clarify_rules.py의 내부 이름 체계)을 정한다.

    2026-08-17: "period"/"calc_type"은 표 구분 없이 항상 필요(계산 자체가 불가능하거나,
    calc_type은 뒤에서 route_calc_type()이 항상 덮어쓰므로 물어봐도 해는 없음)하지만,
    "region"은 표에 지역 축 자체가 없으면 되물을 이유가 없다 — table_catalog.json에
    "지역"이 required_slots든 optional_slots든 아예 안 걸린 표(예: DT_1DA7102S)면 region을
    뺀다. table_catalog.json에 없는 표(아직 B가 안 채웠거나 조회 실패)는 안전하게 기존
    동작대로 region을 그대로 요구한다(하위 호환 폴백).

    "업종"/"품목"처럼 그 외 표별 세부 축은 여기서 다루지 않는다 — 그건 clarify()의
    한 번짜리 고정 clarify_reply로는 답할 수 없는 질문이고(배치 파이프라인은 실제 대화가
    아니라 미리 준비된 문구로 재시도하는 구조), 실제로는 select_dimension_values()가
    claim 내용을 보고 이미 채워주고 있다(2026-08-16 작업) — 여기서 또 요구하면 답 못 할
    질문 때문에 claim이 불필요하게 스킵될 위험만 커진다."""
    entry = catalog_by_id.get(table_id) if table_id else None
    if entry is None:
        return ["period", "region", "calc_type"]
    has_region = "지역" in entry.get("required_slots", []) or "지역" in entry.get("optional_slots", [])
    return ["period", "region", "calc_type"] if has_region else ["period", "calc_type"]


def _normalize_statistic_name(expression: Optional[str]) -> Optional[str]:
    """3단계 keyword_search.py의 기존 SYNONYMS 사전을 재사용해서 기사 표현을 정규화된
    통계 용어로 바꾼다 (새 LLM 호출 없음). 매칭 안 되면 원래 표현 그대로 둔다."""
    if not expression:
        return None
    for raw_term, mapped_terms in SYNONYMS.items():
        if raw_term in expression:
            return mapped_terms[0]
    return expression



# 제목 뒤, 실제 본문 시작 전에 끼어드는 기자명/타임스탬프/댓글수 배지 블록.
# 실측 확인(2026-08-17): "최아리 기자 입력 2025.02.20. 06:00 업데이트 2025.02.20. 06:07 0
# 지난달 생산자물가가..." 식으로, 마지막 숫자(댓글수 등 UI 배지, 기사마다 0/1/4/23처럼
# 다른 값)까지 본문에 섞여 스크랩된다 — 프론트엔드 "기사 원문"에 이 숫자가 그대로
# 노출되는 문제로 발견됨. 업데이트 타임스탬프는 없는 기사도 있어 그 부분은 선택적으로 둔다.
_BYLINE_RE = re.compile(
    r"[가-힣]+\s*기자\s*입력\s*\d{4}\.\d{2}\.\d{2}\.\s*\d{2}:\d{2}"
    r"(?:\s*업데이트\s*\d{4}\.\d{2}\.\d{2}\.\s*\d{2}:\d{2})?"
    r"\s*\d+\s*"
)

# 2026-08-21 추가: 기사 본문이 max_len(3000자)보다 짧으면, 남는 자리에 스크랩된 페이지의
# "관련기사"/"많이 본 뉴스"/추천 위젯 텍스트가 그대로 딸려 들어온다 — 실측 확인(골든셋
# 라벨링 중 발견): '소상공인 365' 서비스 소개 기사에서 완전히 무관한 '포스코 인도 제철소
# 합작' claim이 뽑힌 사례, 재현해보니 그 기사 본문 뒤에 다른 기사 헤드라인 조각들이
# 이어붙어 있었다. claim_extractor는 이걸 진짜 본문인 줄 알고 성실하게 엉뚱한 claim을
# 추출해버린다. 확실한(오탐 위험이 낮은) 마커만 좁게 잡는다 — 마커를 넓히면 진짜 본문이
# 우연히 비슷한 단어를 포함할 때(예: AI 추천 시스템을 다루는 기사) 잘못 잘려나갈 위험이
# 있어 일부러 보수적으로 유지한다.
_JUNK_SECTION_RE = re.compile(r"By\s*Taboola|많이\s*본\s*뉴스|오늘의\s*멤버십|AI\s*추천")


def _clean_scraped_article_text(title: str, raw_text: str, max_len: int = 3000) -> str:
    """실제 스크랩 기사(CSV의 '기사 본문 전체')는 신문사 내비게이션 메뉴가 본문 앞에
    반복적으로 붙어있어서(광고/관련기사 텍스트까지 합치면 2만자 넘는 경우도 있음),
    그대로 HCX에 넘기면 "40003 Context length exceeded"로 거부당한다 (실제 재현됨).

    기사제목의 앞부분을 raw_text 안에서 찾아 그 위치부터 잘라내는 방식으로 내비게이션
    잡음을 건너뛰고, 이후 max_len자만 남겨서 컨텍스트 길이를 안전하게 유지한다.
    제목을 못 찾으면(예외 케이스) 그냥 앞에서부터 max_len자를 쓴다.

    제목 뒤에도 기자명/입력·업데이트 시각/댓글수 배지가 실제 본문 시작 전에 끼어드므로
    (_BYLINE_RE 참고), 찾아지면 그 블록까지 건너뛰고 진짜 첫 문장부터 시작한다.

    그리고 (_JUNK_SECTION_RE 참고) 본문 뒤에 "관련기사"류 잡음 섹션이 시작되는 지점을
    찾으면 그 지점 이후는 잘라낸다 — 짧은 기사에서 남는 자리가 무관한 다른 기사 내용으로
    채워지는 걸 막기 위함.
    """
    anchor = title[:12].strip()
    idx = raw_text.find(anchor) if anchor else -1
    start = idx if idx >= 0 else 0
    trimmed = raw_text[start : start + max_len]

    m = _BYLINE_RE.search(trimmed[: len(title) + 100])
    if m:
        trimmed = trimmed[: m.start()] + trimmed[m.end() :]

    junk_match = _JUNK_SECTION_RE.search(trimmed)
    if junk_match:
        trimmed = trimmed[: junk_match.start()]

    return trimmed


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


def _match_by_name(name_to_code: dict, texts: tuple) -> Optional[str]:
    """(공용 매칭 로직) name_to_code(예: itmId->이름, 또는 objL 코드->이름을 뒤집은 것)에서
    texts(claim.population, claim.statistic_expression 등)를 순서대로 하나씩 검사해,
    처음으로 매칭이 나온 필드의 결과를 확정해서 반환한다. 매칭 하나도 없으면 None.

    필드 우선순위: texts는 호출부가 이미 "더 구체적인 신호부터" 순서를 정해서 넘긴다고
    가정한다(예: population을 statistic_expression보다 먼저) — 한 필드 안에서 매칭이
    나오면 다른 필드는 아예 안 본다. 그 이유는 select_itm_id()가 원래 두 필드를 한꺼번에
    비교하다가 겪은 실측 버그 때문이다: "인구수"(statistic_expression)가 "총인구수"(길이4)에
    부분 매칭되는 게 "남자"(population, 길이2)보다 길다는 이유로 더 구체적인 population
    매칭을 밀어내버렸다 — 필드를 섞지 않고 먼저 매칭되는 필드에서 확정해야 이 오판을 막는다.

    한 필드 안에서는: (1) 완전 일치를 최우선으로 쓰고, (2) 없으면 부분 문자열 중 이름이
    가장 긴(=가장 구체적인) 쪽을 쓴다. 완전 일치를 먼저 보는 이유는 "농가"/"비농가"처럼
    부정 접두사로 만들어진 이름은 짧은 쪽이 긴 쪽의 부분 문자열이 돼서(population="농가"가
    name="비농가"의 부분 문자열), 길이 기준만 쓰면 "더 긴 쪽=더 구체적"이라는 가정이
    뒤집혀 정반대 항목이 뽑히는 실측 버그가 있었기 때문이다."""
    for text in texts:
        if not text:
            continue
        exact = next((code for name, code in name_to_code.items() if name == text), None)
        if exact is not None:
            return exact
        best_code, best_len = None, 0
        for name, code in name_to_code.items():
            if name in text or text in name:
                if len(name) > best_len:
                    best_code, best_len = code, len(name)
        if best_code is not None:
            return best_code
    return None


def select_itm_id(table_id: str, claim, table_params: dict) -> Optional[str]:
    """claim.statistic_expression/population(2단계가 이미 뽑아둔 필드)을 표의 items 목록
    (itmId -> 항목명)과 대조해서 가장 맞는 itmId를 고른다. 표에 items가 등록 안 됐거나
    매칭되는 항목이 없으면 None(호출부가 기존 itmId_fixed로 폴백)을 반환한다.

    배경(2026-08-16 실측): "성별 경제활동인구총괄"(DT_1DA7001S)처럼 한 표 안에 여러
    항목(취업자/실업자/실업률/고용률 등)이 같이 들어있는 표가, 지금까지는 claim 내용과
    무관하게 항상 같은 itmId(예: 실업률)로 고정 조회되고 있었다 — "취업자 수가
    13만5000명 늘었다"는 claim에 실업률 3.7%를 돌려주는 식으로, 표는 맞게 찾았는데
    표 안의 항목이 틀려서 판정 자체가 무의미해지는 사례가 실측 확인됐다(table_params.json
    등록 당시 이미 알려진 갭으로 문서화돼 있었음). 여기서 그 갭을 채운다.

    간단한 부분 문자열 매칭만 쓴다(임베딩 없음) — items 이름이 소수(표 하나당 대개
    10개 이하)이고 "취업자"/"실업률"처럼 뜻이 뚜렷이 갈리는 한국어 통계 용어라, 3단계
    표매칭(keyword_search)과 같은 원리로 규칙 기반이면 충분하다고 판단. 매칭 로직 자체는
    _match_by_name() 참고 — population을 statistic_expression보다 먼저 본다(성별/가구유형
    같은 "누구/어떤 대상" 신호는 대개 population에 실리는데, statistic_expression 쪽의
    범용 표현이 그걸 밀어내는 오판을 막기 위함).

    2026-08-20 버그 수정: table_params.json의 items는 itmId -> 항목명(예: {"T30": "취업자"})으로
    저장돼 있는데, _match_by_name()은 반대 방향(이름 -> 코드)을 기대한다(dimensions의
    code_map과 동일한 계약). 뒤집지 않고 그대로 넘기면 claim 문장을 "T30" 같은 itmId
    문자열과 대조하게 돼서 절대 매칭이 안 되고 항상 None(표의 itmId_fixed 기본값 폴백)만
    반환한다 — 이 함수가 원래 고치려던 "표 안 여러 항목 중 잘못된 게 고정 조회되는" 문제가
    이 함수 자체의 버그로 인해 여전히 재현됨(실측: "취업자 수" claim이 DT_1DA7001S의
    기본값인 실업률(T80)로 조회됨). 여기서 이름 -> 코드로 뒤집어서 넘긴다."""
    items = table_params.get(table_id, {}).get("items")
    if not items:
        return None
    name_to_code = {name: code for code, name in items.items()}
    return _match_by_name(name_to_code, (claim.population, claim.statistic_expression))


def select_dimension_values(table_id: str, claim, table_params: dict, existing_slots: dict) -> dict:
    """itmId 말고 objL1/objL2 같은 "dimension" 축도 고정값 문제가 있어서(예: 유형별
    매매가격지수 표가 항상 "종합"으로만 조회되고 "아파트"/"단독주택" 주장은 구분 못 함),
    select_itm_id()와 같은 원리를 표의 dimensions 전체로 일반화한 버전.

    배경(2026-08-16): itmId 문제를 고치고 나서 살펴보니, 같은 모양의 문제가 itmId 아닌
    dimensions에도 여럿 있었다(주택유형/대출종류/계정항목/사망원인 등 — 표마다 축 이름은
    달라도 전부 "code_map은 있는데 claim 내용과 무관하게 항상 default_value로만 고정
    조회된다"는 동일 패턴). dimensions[dim_name].code_map은 C가 표 조사할 때 이미 실제
    API로 검증해서 다 채워둔 데이터라 새로 조사할 필요 없이 그대로 재사용 가능하다.

    D(4단계 slot_filler)가 이미 값을 채운 축(가장 흔한 예: region)은 절대 덮어쓰지 않는다
    — existing_slots에 이미 값이 있으면 그 dim_name은 건드리지 않고 건너뛴다. 이 함수가
    실제로 값을 채우는 건 D가 애초에 모르는(제너릭 슬롯에 없는) 표별 전용 축들뿐이다."""
    dims = table_params.get(table_id, {}).get("dimensions", {})
    resolved: dict = {}
    for dim_name, dim in dims.items():
        if existing_slots.get(dim_name):
            continue
        code_map = dim.get("code_map")
        if not code_map:
            continue
        match = _match_by_name(code_map, (claim.population, claim.statistic_expression))
        if match is not None:
            resolved[dim_name] = match
    return resolved


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
    if generic_slots.get("itm_id"):
        # 2026-08-16: select_itm_id()가 claim에 맞는 항목을 찾았으면 그대로 실어서
        # KosisApiClient까지 전달한다 — 안 넘기면 api_client.py가 표의 itmId_fixed
        # 기본값으로 폴백한다(하위 호환, prd_se와 같은 원칙).
        kosis_slots["itm_id"] = generic_slots["itm_id"]

    for dim_name, dim in base.get("dimensions", {}).items():
        # 이 표에 정의된 축(dim_name)만 채운다. generic_slots에 값이 있으면 쓰고,
        # 없으면 표의 default_value로 채운다 (예: region이 없는 표는 gender/age만 봄).
        value = generic_slots.get(dim_name)
        kosis_slots[dim_name] = value if value is not None else dim.get("default_value")

    return kosis_slots


# "달"/"월"이 시점이 아니라 다른 단어의 일부로 등장하는 경우가 실측 확인됐다(2026-08-18,
# "344억달러" claim에서 "달러"의 "달"을 "지난달"의 "달"로 오인해 prd_se가 엉뚱하게 "M"으로
# 골라짐 — "조달"/"발달"/"도달"/"배달"/"달성"/"월드컵"/"세월"/"월급"/"월요일"도 같은 위험).
# 부분 문자열 검사 대신 형태소 분석기로 "달"/"월"이 독립된 토큰으로 떨어지는지 확인하면
# (kiwipiepy, keyword_search.py의 _kiwi와 동일 인스턴스 재사용) 이런 오탐을 글자 하나하나
# 블랙리스트로 등록할 필요 없이 한 번에 걸러낼 수 있다. 단, "지난달"/"이달"/"전달"/"전월"/
# "동월"/"매달"처럼 형태소 분석기가 통째로 한 단어로 묶어버리는 소수의 사전 등재 시점
# 표현만 예외적으로 화이트리스트에 남겨둔다(실측: kiwi.tokenize로 직접 확인).
_MONTH_FUSED_COMPOUNDS = ("지난달", "이달", "전달", "전월", "동월", "매달")


def _infer_desired_granularity(text: Optional[str]) -> Optional[str]:
    """claim 문장에서 이 주장이 월/분기/연 중 어느 주기를 가리키는지 best-effort로 추정한다.

    3단계(prd_se 선택 로직)의 핵심 — 2026-08-13, 64개 표 중 41개(64%)가 한 표에서 여러
    주기를 동시에 지원한다는 게 실측 확인돼서, "표가 지원하는 것 중 claim이 실제로 원하는
    주기"를 골라야 한다. "분기"가 명시되면 최우선(월/연보다 구체적인 표현), 그 다음
    월/달 표현, 나머지는 판단 보류(None) — 상위 호출부가 표 주기 목록에서 기본값을 쓴다."""
    if not text:
        return None

    if _kiwi is not None:
        tokens = {tok.form for tok in _kiwi.tokenize(text)}
        if "분기" in tokens:
            return "Q"
        if "월" in tokens or "달" in tokens or any(kw in text for kw in _MONTH_FUSED_COMPOUNDS):
            return "M"
        return None

    # kiwipiepy 미설치 시에만 예전 방식(단순 부분 문자열)으로 폴백 — "억달러" 등 오탐 위험 있음.
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
    catalog_by_id: Optional[dict] = None,
    claim_region: Optional[str] = None,
) -> Optional[dict]:
    """4단계: fill_slots + clarify. 한 번에 안 채워지면 clarify_reply로 한 번 더 시도.
    그래도 부족하면 None (되묻기 미해결 → 5단계로 못 감)을 반환한다.

    claim_region(2026-08-17 추가): 2단계(claim_extractor)가 이미 뽑아둔 Claim.region을
    4단계 시작 슬롯으로 시드한다. 예전엔 fill_slots()를 항상 빈 슬롯({})으로 불러서, 2단계가
    이미 맞게 뽑아둔 region을 4단계가 같은 문장 놓고 LLM으로 처음부터 다시 뽑았다 — 낭비인
    데다, DB에 저장되는 표시 필드(region은 claim.region을 그대로 씀)와 실제 KOSIS 조회에
    쓰이는 슬롯(4단계 결과)이 서로 다른 값을 가리킬 수 있는 불일치 위험이 있었다.
    fill_slots()는 이미 "이미 값 있는 슬롯은 안 덮어씀" 로직이 있어서, 여기서 시드만
    넣어주면 2단계가 채운 건 그대로 쓰고 못 채운 것만 4단계 LLM이 보강한다(2단계가 아예 못
    채웠으면 지금까지와 동일하게 동작 — 회귀 없음). 4단계 자체 로직에 있던
    is_region_grounded 방어도 여기서 그대로 적용해서(대조 대상은 claim_sentence), 문장에
    실제로 없는 지명이면(2단계가 기사 다른 부분 보고 잘못 지어냈을 가능성) 시드하지 않는다.

    claim.period는 일부러 시드하지 않는다 — 2단계가 뽑는 period는 "작년"/"지난달"처럼
    원문 표현 그대로라 4단계가 기대하는 정규화된 형식("2024"/"202404")이 아니다. 정규화는
    LLM 호출(normalize_time_expressions) 안에서 같이 일어나는 과정이라 건너뛸 수 없어서,
    이 최적화는 region에만 안전하게 적용된다.

    table_id/table_params: 3단계가 이미 매칭한 표의 지원 주기 목록에서, claim 문장이
    실제로 원하는 주기(_infer_desired_granularity)와 맞춰 prd_se 하나를 고른다(2026-08-13,
    다중 주기 지원 작업 — 표 주기를 몰라 무조건 연도만 채우던 문제(2026-08-12)에 이어,
    표가 여러 주기를 지원해도 claim이 원하는 걸 못 고르던 문제까지 해결). 안 넘기면 기존과
    동일하게 연 단위로만 동작한다(하위 호환).

    catalog_by_id: table_catalog.json 인덱스 — 표별로 실제 필요한 슬롯이 다르다는 걸
    반영한다(2026-08-17, _table_required_slots 참고). 안 넘기면 기존처럼 모든 표에 대해
    region까지 무조건 요구한다(하위 호환).

    반환하는 slots에 "prd_se"를 실제로 채워 넣는다 — build_kosis_slots가 이 값을
    KosisApiClient까지 그대로 전달해서, 표의 기본값이 아니라 여기서 고른 주기로 조회하게
    한다."""
    seed_slots: dict = {}
    if claim_region and is_region_grounded(claim_region, claim_sentence):
        seed_slots["region"] = claim_region

    prd_se = None
    if table_id and table_params and table_id in table_params:
        supported = table_params[table_id].get("prdSe")
        desired = _infer_desired_granularity(claim_sentence)
        prd_se = _select_prd_se(supported, desired)

    required_slots = _table_required_slots(table_id, catalog_by_id) if catalog_by_id is not None else None

    slots = fill_slots(claim_sentence, seed_slots, article_date, prd_se=prd_se)
    if prd_se:
        slots["prd_se"] = prd_se
    question = clarify(slots, required_slots)
    print(f"[4단계 slot_filler] 1차 슬롯: {slots}")

    if question and clarify_reply:
        print(f"[4단계 clarify] 되묻기: \"{question}\" → (준비된 답변) \"{clarify_reply}\"")
        slots = fill_slots(clarify_reply, slots, article_date, prd_se=prd_se)
        if prd_se:
            slots["prd_se"] = prd_se
        question = clarify(slots, required_slots)
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


def _normalize_whitespace(text: str) -> str:
    """공백 제거 비교용 정규화. keyword_search.py의 _normalize()와 동일한 이유 —
    기사 문장은 "원화 기준"/"국가 채무"처럼 code_map 라벨("원화기준"/"국가채무")과
    띄어쓰기가 다른 경우가 흔해서, 공백을 무시하고 비교한다."""
    return re.sub(r"\s+", "", text)


# ---------------------------------------------------------------------------
# "20대"/"70대 이상"·"청년"/"노인" 같은 나이 표현을 KOSIS 5세 단위 code_map과 대조해서
# 필요한 코드 목록으로 바꾸는 순수 판별 로직.
#
# 2026-08-17: 원래 agent_chat.py(실전2: 대화형 챗봇 에이전트)용으로 만들어져 있었는데,
# 정작 그쪽에서도 실제로 호출하는 곳이 없어서(grep 확인) 이 배치 파이프라인은 물론
# agent_chat.py 자신도 못 쓰고 있던 함수였다. 한 번은 agent/shared/age_group_patterns.py로
# 옮겼었지만, 챗봇(agent_chat.py) 자체를 정리할 계획이라 실제로 이 로직을 쓰는 곳은 이
# 배치 파이프라인 하나뿐이다 — "공용 모듈"이라는 틀이 필요 없어서, 아예 여기로 직접
# 들여왔다(fetch_by_codes+compute_sum 배선은 아래 _resolve_decade_age/_fetch_value 참고).
# ---------------------------------------------------------------------------
_AGE_BAND_RANGE_RE = re.compile(r"^(\d+)~(\d+)세$")
_AGE_100_PLUS_RE = re.compile(r"^100세이상$")
# (?<!\d): 매칭되는 한 자리 숫자 앞에 다른 숫자가 있으면(예: "1480대"의 "80대") 매칭하지
# 않는다 — 실제 기사 실측(2026-08-04, full_coverage_result.jsonl 3181건 분석)에서
# "1082만1480대"(자동차 판매 대수)처럼 큰 숫자의 끝자리가 나이대로 오탐되는 사례를 발견해서
# 추가한 정규식 레벨 방어. "20대"/"10·20대"처럼 진짜 나이대는 앞에 다른 숫자가 안 붙는다.
_DECADE_EXPR_RE = re.compile(r"(?<!\d)(\d)0대(\s*이상)?")


def _parse_age_band_label(label: str) -> Optional[tuple[int, Optional[int]]]:
    """code_map의 5세 단위 라벨을 (시작나이, 끝나이) 튜플로 변환한다. "100세이상"처럼
    끝이 없으면 끝나이는 None(무한대 취급). 매칭 안 되면 None."""
    m = _AGE_BAND_RANGE_RE.match(label)
    if m:
        return int(m.group(1)), int(m.group(2))
    if _AGE_100_PLUS_RE.match(label):
        return 100, None
    return None


# "숫자0대"가 진짜 나이대인지 문맥 확인 (1차 규칙 + 2차 LLM 더블체크). 실측(2026-08-04,
# 실제 기사 1005건/claim 3181건 분석)에서 같은 정규식이 나이대가 아닌 경우에도 오탐되는 걸
# 확인함: "상위 10대 기업"(순위), "K2 전차 총 180대"(대수/단위). 정규식 하나로는 이 둘을
# 못 걸러서, 확실한 신호가 있는 대다수는 규칙(공짜)으로 끝내고, 신호가 없는 애매한 소수만
# LLM에 묻는다.
_DECADE_AGE_FOLLOW_WHITELIST = (
    "후반", "초반", "중반", "이상", "미만", "인구", "실업률", "취업률", "취업자",
    "근로자", "직장인", "남성", "여성", "남자", "여자", "가구주", "세대주",
)
_DECADE_AGE_FOLLOW_BLACKLIST = ("기업", "은행", "국가", "기관", "대학", "도시", "종목", "그룹", "브랜드")
_DECADE_AGE_PRECEDE_BLACKLIST = (
    "전차", "차량", "트럭", "헬기", "전투기", "버스", "컴퓨터", "노트북", "항공기", "탱크", "장비",
)
# 조사로 자연스럽게 끝나는 경우("20대는", "20대가")도 나이대로 확정 — 순위/대수 표현은
# 보통 뒤에 바로 다른 명사(기업/전차 등)가 오지 조사만 딱 붙어 끝나지 않는다.
_DECADE_AGE_PARTICLE_CHARS = "는가이을를의도만에과와랑"


def _check_decade_age_context(normalized_sentence: str, match: re.Match) -> Optional[bool]:
    """"숫자0대" 매칭 앞뒤 문맥만 보고 확실히 판단되면 True/False, 확실하지 않으면(애매) None.

    None을 반환하면 호출하는 쪽이 LLM(2차)에게 물어봐야 한다 — 여기서 애매한 걸 억지로
    True/False로 단정하지 않는다."""
    follow = normalized_sentence[match.end() : match.end() + 6]
    precede = normalized_sentence[max(0, match.start() - 6) : match.start()]

    if any(w in follow for w in _DECADE_AGE_FOLLOW_BLACKLIST):
        return False
    if any(w in precede for w in _DECADE_AGE_PRECEDE_BLACKLIST):
        return False
    if any(w in follow for w in _DECADE_AGE_FOLLOW_WHITELIST):
        return True
    if not follow or follow[0] in _DECADE_AGE_PARTICLE_CHARS:
        return True
    return None


def _confirm_decade_age_with_llm(sentence: str, matched_expr: str) -> bool:
    """규칙(1차)으로 확실히 판단 안 되는 애매한 경우만 호출하는 2차 확인. 실패/근거없음/
    근거가 원문에 없으면(환각) 전부 안전하게 False(나이대 아님)로 처리한다 — "확신 없으면
    억지로 추측하지 않는다" 원칙."""
    prompt = f"""다음 문장에서 "{matched_expr}"라는 표현이 나이대(연령대)를 뜻하는지 판단하세요.

규칙:
- 나이대를 뜻하면(예: "20대 취업자 수", "40대 남성") value를 true로 답하세요.
- 개수를 세는 단위(예: "전차 180대")나 순위(예: "상위 10대 기업")처럼 나이가 아니면
  value를 false로 답하세요.
- 판단 근거가 되는, 문장에 실제로 나온 표현을 quote에 그대로 복사하세요. 지어내지 마세요.
- 설명이나 다른 텍스트는 절대 포함하지 마세요.

문장: "{sentence}"

응답 형식 (JSON만, 다른 텍스트 없이): {{"value": true 또는 false, "quote": "근거 문구"}}
"""
    try:
        raw = call_hcx(prompt)
    except Exception:
        return False

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = extract_json_fallback(raw)

    value = extracted.get("value")
    quote = extracted.get("quote")
    if value is not True or not quote:
        return False
    if _normalize_whitespace(str(quote)) not in _normalize_whitespace(sentence):
        return False
    return True


def _resolve_decade_age_codes(sentence: str, age_code_map: dict) -> Optional[tuple[str, list[str]]]:
    """문장에서 "20대"/"70대 이상" 같은 표현을 찾아, age_code_map(5세 단위 라벨들) 중 그
    구간에 해당하는 코드 목록을 반환한다. 매칭 없으면 None.

    반환값: (매칭된 표현 원문, 해당 구간 코드 리스트) — 코드가 2개 이상이면 호출하는 쪽이
    api_client를 코드별로 여러 번 호출한 뒤 calculator.compute_sum()으로 합산해야 한다."""
    normalized_sentence = _normalize_whitespace(sentence)
    m = _DECADE_EXPR_RE.search(normalized_sentence)
    if not m:
        return None

    context_check = _check_decade_age_context(normalized_sentence, m)
    if context_check is False:
        return None
    if context_check is None and not _confirm_decade_age_with_llm(sentence, m.group(0)):
        return None

    decade_start = int(m.group(1)) * 10
    is_or_above = bool(m.group(2))

    matched_codes = []
    for label, code in age_code_map.items():
        band = _parse_age_band_label(label)
        if band is None:
            continue
        band_start, band_end = band
        if is_or_above:
            if band_start >= decade_start:
                matched_codes.append(code)
        elif band_start >= decade_start and (band_end is not None and band_end < decade_start + 10):
            matched_codes.append(code)

    if not matched_codes:
        return None
    return m.group(0), matched_codes


# "청년"/"노인"/"청소년"/"중장년"/"아동"/"영유아" 같은 생애주기 명칭 처리. "20대"류(정확히
# 10세 구간)와 달리 이런 명칭은 법령마다 정의된 나이 구간이 있다(예: 청년은 청년기본법=
# 19~34세, 청년고용촉진법=15~29세로 법령마다 다름 — 2026-08-17 실측 확인, 표 하나가 실제로
# "청년(15~29세)"로 등록돼 있었음). 그래서 표에 이미 그 명칭을 포함한 라벨이 있으면(그 표가
# 실제로 채택한 정의) 그걸 최우선으로 쓰고, 없을 때만 아래 기본 범위로 5세 단위 코드를
# 합산한다 — 우리가 임의로 범위를 정해서 표의 실제 정의를 덮어쓰지 않기 위함.
#
# 기본 범위는 정부 통계에서 흔히 쓰이는 기준을 그대로 채택했다(영유아=영유아보육법,
# 아동=아동복지법, 청소년=청소년기본법(9~24세, 흔히 생각하는 "10대"보다 훨씬 넓음),
# 청년=청년기본법과 청년고용촉진법 중 더 자주 쓰이는 후자, 중장년=중장년내일센터 등 정책
# 기준, 노인/고령자/고령층=노인복지법). 경계(9/17/64 등)가 5세 단위 구간과 안 맞아떨어지는
# 경우, 구간에 완전히 포함되는 5세 단위 코드만 합산 대상으로 삼는다(부분 겹침은 제외 —
# 과대 포함보다 과소 포함이 안전하다고 판단).
_NAMED_AGE_GROUPS: dict[str, tuple[int, Optional[int]]] = {
    "영유아": (0, 6),
    "아동": (0, 17),
    "청소년": (9, 24),
    "청년": (15, 29),
    "중장년": (40, 64),
    "고령층": (65, None),
    "고령자": (65, None),
    "노인": (65, None),
}


def _resolve_named_age_group_codes(sentence: str, age_code_map: dict) -> Optional[tuple[str, list[str]]]:
    """"청년"/"노인" 같은 생애주기 명칭을 감지해서, 표에 그 명칭이 포함된 단일 라벨이
    있으면 그 코드 하나를, 없으면 _NAMED_AGE_GROUPS의 표준 범위에 해당하는 5세 단위 코드
    들을 합산 대상으로 반환한다. 감지 안 되거나 표에서 대응할 코드를 못 찾으면 None."""
    normalized_sentence = _normalize_whitespace(sentence)
    matched_group = next((g for g in _NAMED_AGE_GROUPS if g in normalized_sentence), None)
    if matched_group is None:
        return None

    explicit = next((code for label, code in age_code_map.items() if matched_group in label), None)
    if explicit is not None:
        return matched_group, [explicit]

    start, end = _NAMED_AGE_GROUPS[matched_group]
    matched_codes = []
    for label, code in age_code_map.items():
        band = _parse_age_band_label(label)
        if band is None:
            continue
        band_start, band_end = band
        if band_start < start:
            continue
        if end is not None and (band_end is None or band_end > end):
            continue
        matched_codes.append(code)

    if not matched_codes:
        return None
    return matched_group, matched_codes


def _resolve_decade_age(table_id: str, table_params: dict, claim_sentence: Optional[str]):
    """claim_sentence에서 "20대"/"70대 이상"류 10세단위 나이 표현이나 "청년"/"노인" 같은
    생애주기 명칭이 있고 이 표에 age dimension(5세단위 code_map)이 있으면 (매칭된 표현,
    필요한 코드 리스트)를 반환한다. 감지 안 되면 None. 숫자 기반 "20대"류를 먼저 보고,
    없으면 명칭 기반("청년" 등, 2026-08-17 추가 — agent_chat._resolve_named_age_group_codes
    참고)을 본다 — 한 문장에 둘 다 나오는 경우는 실질적으로 없다고 봐서 우선순위만 두고
    별도 병합 로직은 두지 않는다.

    반드시 한 claim(한 번의 run_stage_5_6 호출)당 한 번만 호출해서 재사용해야 한다 —
    내부에서 정규식으로 애매하면 LLM(_confirm_decade_age_with_llm)에 물어보는데, 증감
    claim처럼 같은 문장으로 base/target 두 번 조회하는 경우 매번 새로 호출하면 LLM을
    쓸데없이 두 번 부르고, temperature 때문에 base/target이 서로 다른 답(하나는 나이대로
    인정, 하나는 아니라고)을 받는 내부 불일치 위험도 생긴다."""
    age_code_map = table_params.get(table_id, {}).get("dimensions", {}).get("age", {}).get("code_map")
    if not age_code_map:
        return None
    sentence = claim_sentence or ""
    return _resolve_decade_age_codes(sentence, age_code_map) or _resolve_named_age_group_codes(
        sentence, age_code_map
    )


def _fetch_value(
    table_id: str,
    slots: dict,
    client: KosisApiClient,
    calculator: KosisCalculator,
    decade: Optional[tuple],
) -> KosisApiResponse:
    """5단계 조회 한 번. decade가 (라벨, 코드리스트)면(_resolve_decade_age 참고) 그 코드들을
    fetch_by_codes()로 한 번에 가져와 compute_sum()으로 합산한 뒤 KosisApiResponse 모양으로
    돌려준다(단위/시점은 합산 결과, org_id/itm_id 등은 응답 중 첫 번째 것 재사용 — 같은
    표·같은 조회라 전부 동일함). decade가 None이면 기존처럼 client() 단일 호출 그대로."""
    if decade is not None:
        label, codes = decade
        responses = client.fetch_by_codes(table_id, slots, "age", codes)
        summed = calculator.compute_sum(responses)
        print(f"[나이대 합산] '{label}' → {len(responses)}개 5세단위 코드 합산 = {summed.raw_value} {summed.unit}")
        first = responses[0]
        return KosisApiResponse(
            raw_value=summed.raw_value, unit=summed.unit, period=summed.period,
            org_id=first.org_id, itm_id=first.itm_id,
            obj_l1=first.obj_l1, obj_l2=first.obj_l2, prd_se=first.prd_se,
        )
    return client(table_id, slots)


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
    # 2026-08-16: "20대"/"70대 이상" 같은 10세단위 나이 표현은 이 claim 안에서 base/target
    # 등 여러 번 조회하더라도 단 한 번만 판별해서 재사용한다(_resolve_decade_age 참고 —
    # 중복 LLM 호출/내부 불일치 방지).
    decade = _resolve_decade_age(table_id, table_params, claim_sentence)
    try:
        if calc_type in ("증감", "증감률") and kosis_slots.get("period"):
            if _wants_prior_immediate_period(comparison_target):
                base_period = _prior_immediate_period(kosis_slots["period"], generic_slots.get("prd_se"))
            else:
                base_period = _prior_year_same_period(kosis_slots["period"])
            base_slots = dict(kosis_slots, period=base_period)
            base_resp = _fetch_value(table_id, base_slots, client, calculator, decade)
            target_resp = _fetch_value(table_id, kosis_slots, client, calculator, decade)
            print(f"[5단계 api_client] base   = {base_resp}")
            print(f"[5단계 api_client] target = {target_resp}")

            calc_fn = calculator.compute_change_rate if calc_type == "증감률" else calculator.compute_change
            # 2026-08-17: base_resp.unit이 이미 "%"인 통계(실업률/고용률/물가상승률처럼 값
            # 자체가 비율)는, 기사가 "5.1%로 전년 대비 0.2%포인트 상승"처럼 그 변화도 %로
            # 적어도 실제로는 두 %값의 단순 차이(포인트차)를 말하는 것이지, compute_change_rate가
            # 계산하는 상대적 증감률((target-base)/base*100, "%의 %")이 아니다. 이걸 구분 안
            # 하면 5.1%->5.6% 변화가 "5.7%(원래 값 대비 상대 증가율)"로 계산돼 기사의 "0.2%p"와
            # 전혀 다른 숫자가 나온다(실측 재현: 청년 실업률 claim, 5.1→5.6인데 증감률로
            # 계산하면 5.7%가 나와 기사 수치와 아예 딴 세상 값이 됨). 값 자체가 이미 %인
            # 경우엔 그냥 차이(compute_change)로 계산해야 "%포인트" 표현과 맞는다. (드물게
            # "영업이익률이 50% 증가했다"처럼 비율 자체의 상대적 증가를 말하는 경우엔 이 휴리스틱이
            # 틀릴 수 있지만, 한국 뉴스에서 실업률/고용률/물가상승률류 %지표는 거의 항상
            # %포인트로 비교되므로 더 흔한 케이스를 맞춘다.)
            if calc_type == "증감률" and base_resp.unit == "%":
                calc_fn = calculator.compute_change
            result = calc_fn(base_resp, target_resp)
            print(f"[6단계 calculator] {result}")
            return result
        elif calc_type in ("최댓값검증", "최솟값검증") and kosis_slots.get("period"):
            # 2026-08-16: 이 분기는 decade(나이대 합산)를 아직 지원하지 않는다 — historical
            # 전체 구간의 매 시점마다 5세단위 코드 여러 개를 합산해야 해서 스코프가 훨씬
            # 크다(단순조회/증감처럼 시점 1~2개가 아니라 수십 개 시점 전부). "20대 취업자
            # 역대 최고" 같은 claim은 지금은 decade 미적용 상태로 그대로 진행된다(알려진 갭).
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

            # 2026-08-20 버그 수정: 여기서 항상 4자리 연도("1960"/"2024")만 만들어 넘기고
            # 있었는데, fetch_series -> _validate_period_format은 표의 prd_se가 M/Q면 6자리
            # (YYYYMM 또는 YYYY+분기 2자리)를 요구한다 — 표가 월/분기 단위 전용(예: 부동산
            # 매매가격지수, GDP)이면 이 분기가 100% KosisApiError로 실패했다(2026-08-04
            # 통합 테스트 로그에 "fetch_series 월단위 gap"으로 기록된 채 미해결로 남아있던
            # 문제). kosis_slots["period"]는 이미 run_stage_4가 표의 실제 주기에 맞춰
            # 정규화해둔 값(예: "202506")이라 그대로 종료 시점으로 재사용하고, 시작 시점은
            # 해당 연도의 첫 주기("YYYY01" — 월의 1월이든 분기의 1분기든 같은 "01" 표기)로
            # 만든다. Y(연간) 표는 기존처럼 4자리 연도 그대로 둔다(하위 호환, 회귀 없음).
            prd_se = kosis_slots.get("prd_se")
            target_period = kosis_slots.get("period")
            if prd_se in ("M", "Q") and target_period and len(str(target_period)) == 6:
                start_period = f"{start_year}01"
                end_period = str(target_period)
            else:
                start_period = str(start_year)
                end_period = str(article_year)

            historical = client.fetch_series(table_id, kosis_slots, start_period, end_period)
            current_resp = client(table_id, kosis_slots)
            print(f"[5단계 api_client] current    = {current_resp}")
            print(f"[5단계 api_client] historical({start_period}~{end_period}) = {len(historical)}건")

            check_fn = calculator.compute_max_check if calc_type == "최댓값검증" else calculator.compute_min_check
            result = check_fn(current_resp, historical)
            print(f"[6단계 calculator] {result}")
            return result
        else:
            resp = _fetch_value(table_id, kosis_slots, client, calculator, decade)
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
    claim, top, computed: ComputedResult, *, prd_se: Optional[str] = None, article_date=None
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
    기존과 동일하게 원본 period 문자열 그대로 노출.

    article_date(2026-08-17 추가): claim.period가 "1월"처럼 연도 없는 상대 표현일 때 judge()가
    기사 발행일을 앵커로 시점을 확정할 수 있게 그대로 넘긴다(agent/verdict/judge.py 참고)."""
    article_date_str = article_date.isoformat() if hasattr(article_date, "isoformat") else article_date
    try:
        verdict = judge(claim, computed, prd_se=prd_se, article_date=article_date_str)
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

    # 안전장치: RRF(Reciprocal Rank Fusion, reranker.py의 _rrf_fuse 참고)로 1등이 된
    # 후보라도, keyword_search가 못 찾았고 리랭커도 독자적으로 1위로 보지 않았다면
    # (embedding/VDB 단독 저순위 근거뿐이라면) 신뢰하지 않는다(is_rrf_trusted() 참고).
    # 2026-08-18: 기존 "unverified면 무조건 매칭없음" 이분법 게이트를 RRF 기반으로 교체
    # — 임베딩 코사인 유사도 하나만으로는(리랭커 없이는) 노이즈와 진짜 신호를 구분하기
    # 어렵다는 게 실측으로 재확인됐지만(2026-08-13), 반대로 VDB 단독 후보가 keyword가
    # 못 찾은 진짜 정답인 사례도 실측 확인돼서(울릉군 기사, "고용률(시/군/구)") 소스
    # 종류만으로 무조건 버리진 않는다.
    if top.source_meta and not is_rrf_trusted(top.source_meta):
        print(f"[3단계 매핑] 최상위 후보가 RRF 기준으로도 신뢰도 낮음(keyword 미발견+리랭커 1위 아님) → 매칭 없음으로 처리")
        try:
            insert_verification(
                _build_verification_record(
                    article=article, claim=claim, top=top, generic_slots=None,
                    table_params=table_params, computed=None, verdict=None, explanation=None,
                    cls_result=cls_result, catalog_by_id=catalog_by_id,
                    verification_possible="애매",
                    ambiguity_reason="표 매칭 신뢰도가 낮음 (keyword 미발견 + 리랭커도 1위로 안 봄)",
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
            catalog_by_id=catalog_by_id,
            claim_region=claim.region,
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

    # 2026-08-16: 표 안에 항목(itmId)이 여러 개인 표(예: "성별 경제활동인구총괄"이
    # 취업자/실업자/실업률/고용률을 다 갖고 있는 경우)는 claim이 실제로 뭘 묻는지 보고
    # 골라야 한다 — 안 그러면 표는 맞게 찾고도 엉뚱한 항목(예: 취업자 수 claim에 실업률
    # 값)을 비교해서 판정 자체가 무의미해진다(오늘 실측 발견, table_params.json에 이미
    # "알려진 갭"으로 문서화돼 있던 문제). 못 찾으면(items 미등록/매칭 실패) None이라
    # slots에 안 실리고, build_kosis_slots가 표의 itmId_fixed 기본값으로 그대로 폴백한다.
    selected_itm = select_itm_id(top.table_id, claim, table_params)
    if selected_itm:
        print(f"[itmId 선택] claim.statistic_expression={claim.statistic_expression!r} → itmId={selected_itm!r}")
        slots["itm_id"] = selected_itm

    # 2026-08-16: itmId 말고 objL1/objL2 같은 dimension 축(주택유형/대출종류/계정항목 등)도
    # 같은 종류의 고정값 문제가 있어서(예: 유형별 매매가격지수가 "아파트" 주장에도 항상
    # "종합"으로만 조회됨) 같은 원리로 동적 선택한다. D가 이미 채운 축(region 등)은
    # select_dimension_values() 내부에서 건드리지 않고 건너뛴다.
    dim_values = select_dimension_values(top.table_id, claim, table_params, slots)
    if dim_values:
        print(f"[dimension 선택] {dim_values}")
        slots.update(dim_values)

    computed = run_stage_5_6(
        top.table_id, slots, table_params, client, calculator,
        comparison_target=claim.comparison_target,
        claim_sentence=claim.sentence, article_year=article["published_date"].year,
    )
    if computed is None:
        return None

    outcome = run_stage_7_8(
        claim, top, computed, prd_se=slots.get("prd_se"), article_date=article.get("published_date")
    )
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

    # 2026-08-17: run_stage_7_8이 None을 반환하는 건 judge()가 실패한 경우뿐이다(JudgeError —
    # LLM이 JSON 대신 대화체로 응답하는 등, 이 세션에서 실제로 여러 번 재현됨). 예전엔 여기서
    # 그냥 None을 반환해서 이 claim이 DB에 아무 기록도 안 남고 조용히 사라졌다 — 콘솔 로그에만
    # 흔적이 남고, 프론트엔드 원문 뷰어에는 그 문장이 하이라이트도 안 된 평문으로 남아 마치
    # "2단계 추출 누락"처럼 보였다(실제로는 추출은 됐고 판정 단계에서 버려진 것, 실제 사례로
    # 재현 확인). 판정 실패도 하나의 결과로 남겨서(판단불가) claim 자체는 항상 눈에 보이게 한다.
    fallback_verdict = Verdict(
        verdict="판단불가", gap_type=None,
        reason="7단계 판정(judge) 응답 처리 중 오류가 발생해 자동 판정에 실패했습니다.",
    )
    try:
        insert_verification(
            _build_verification_record(
                article=article, claim=claim, top=top, generic_slots=slots,
                table_params=table_params, computed=computed, verdict=fallback_verdict,
                explanation=None, cls_result=cls_result, catalog_by_id=catalog_by_id,
                verification_possible="불가", ambiguity_reason="judge() 실패(JudgeError 등) — 콘솔 로그 참고",
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
