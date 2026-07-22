"""
interfaces.py
=================
파이프라인 1~8단계의 입력/출력 데이터 타입 정의.

Day1 킥오프에서 팀 전체가 합의한 "계약(contract)" 파일입니다.
각자 담당 모듈을 만들 때 이 파일을 import해서 타입을 맞춰 쓰세요.

    from interfaces import ClassificationResult, Claim, TableCandidate, Verdict

절대 이 타입들을 각자 임의로 바꾸지 마세요.
바꿔야 할 일이 생기면 팀 전체에 공유 후 여기서 함께 수정합니다.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


# ---------------------------------------------------------------------------
# 1단계 — classifier.py (기사 관련도 분류)
# 담당: A  |  모델: HCX-DASH-002
# 입력: 기사 본문(str)
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    label: bool          # True = 관련 기사(국가 공식 통계/수치 기반 주장 포함), False = 무관
    score: float          # 0.0 ~ 1.0, 확신도. 0.4~0.6은 애매 구간 → 사람 리뷰 큐로
    reason: str            # 판단 근거 한 문장


# ---------------------------------------------------------------------------
# 2단계 — claim_extractor.py (수치 주장 문장 추출)
# 담당: A  |  모델: HCX-003
# 입력: 기사 본문(str)  (※ 1단계 결과 자체가 아니라 원본 기사 본문을 다시 받음)
# 출력: Claim의 리스트 (문장 하나하나 따로 호출 X, 기사 전체 넣고 리스트로 한 번에)
# ---------------------------------------------------------------------------

ClaimType = Literal["규모", "증감률", "비교", "전망"]

@dataclass
class Claim:
    sentence: str
    claim_type: ClaimType
    period: Optional[str] = None       # 예: "2024년" / 시점 불명확하면 None
    unit: Optional[str] = None          # 예: "%", "가구" / 없으면 None
    population: Optional[str] = None    # 예: "국내 과수 농가" / 없으면 None


# ---------------------------------------------------------------------------
# 3단계 — 통계표 매핑 (keyword_search + embedding_search + reranker)
# 담당: B  |  모델: 제공 임베딩 v1·v2 + 제공 리랭커 (LLM 호출 아님, 벡터 연산)
# 입력: Claim 1건
# 출력: TableCandidate의 리스트 (top-k)
# ---------------------------------------------------------------------------

@dataclass
class TableCandidate:
    table_id: str
    table_name: str
    score: float                          # 코사인 유사도 or 리랭커 재정렬 점수
    required_slots: list[str] = field(default_factory=list)  # 이 표를 조회하려면 필요한 슬롯들
    source_meta: Optional[str] = None      # 표 설명/출처 메타 (8단계 설명 생성에 재사용)


# ---------------------------------------------------------------------------
# 4단계 — slot_filler.py / clarify.py (슬롯 채우기, 되묻기)
# 담당: D  |  모델: HCX-DASH-002 기본, 애매한 케이스만 HCX-003
# 입력: 자연어(사용자 발화 또는 Claim) + TableCandidate
# 출력: slots (dict) + 부족하면 되묻기 질문(str)
# ---------------------------------------------------------------------------

Slots = dict[str, str]   # 예: {"region": "서울", "period": "2024", "calc_type": "증감률"}

@dataclass
class ClarifyResult:
    slots: Slots
    missing_slots: list[str] = field(default_factory=list)  # 아직 못 채운 슬롯
    clarify_question: Optional[str] = None  # missing_slots가 있을 때만 채움 (템플릿 우선)


# ---------------------------------------------------------------------------
# 5단계 — kosis/api_client.py (API 호출)
# 담당: C  |  모델: 불필요 (순수 HTTP 요청 + 파라미터 매핑)
# 입력: slots (dict) → orgId/itmId/objL1/prdSe 등으로 매핑해서 호출
# 출력: KosisApiResponse
# ---------------------------------------------------------------------------

@dataclass
class KosisApiResponse:
    raw_value: float
    unit: str
    period: str
    org_id: str
    itm_id: str
    obj_l1: Optional[str] = None
    prd_se: Optional[str] = None


# ---------------------------------------------------------------------------
# 6단계 — kosis/calculator.py (표 연산)
# 담당: C  |  모델: 불필요 (반드시 파이썬 연산, LLM은 결과 "설명"에만 사용)
# 입력: KosisApiResponse (복수 가능 — 합계/비율/증감 계산 시)
# 출력: ComputedResult
# ---------------------------------------------------------------------------

CalcType = Literal["합계", "비율", "증감", "증감률"]

@dataclass
class ComputedResult:
    calc_type: CalcType
    raw_value: float
    unit: str
    period: str


# ---------------------------------------------------------------------------
# 7단계 — 비교·판정 (일치/불일치/판단불가)
# 담당: D  |  모델: 1차 필터는 코드 규칙, 애매 경계만 HCX-003/007
# 입력: Claim(기사 수치) + ComputedResult(KOSIS 계산값)
# 출력: Verdict
# ---------------------------------------------------------------------------

VerdictType = Literal["일치", "불일치", "판단불가"]
GapType = Optional[Literal["수치", "기간", "모집단", "과장표현"]]

@dataclass
class Verdict:
    verdict: VerdictType
    gap_type: GapType = None
    reason: str = ""


# ---------------------------------------------------------------------------
# 8단계 — 검증 결과 설명 생성 (LLM 기반 최종 설명)
# 담당: D + A  |  모델: HCX-007 / RAG Reasoning 모델
# 입력: Claim + TableCandidate + ComputedResult + Verdict
# 출력: Explanation (사람이 읽을 수 있는 설명, 근거+한계 포함 강제)
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    claim_sentence: str
    table_name: str
    calc_summary: str      # 계산 과정 요약 (예: "2023 대비 2024 증감률 -3.2%")
    verdict: VerdictType
    explanation_text: str  # 반드시 (1)근거통계 (2)계산방식 (3)판정이유 (4)한계 4가지 포함
    limitation: Optional[str] = None  # 판단불가일 때 특히 명시 (통계 부재/정의 불명확 등)


# ---------------------------------------------------------------------------
# 파이프라인 전체 흐름 참고 (주석용, 실제 실행 코드 아님)
# ---------------------------------------------------------------------------
#
# 기사 본문(str)
#   -> [1] classifier          -> ClassificationResult
#   -> [2] claim_extractor     -> list[Claim]
#   -> [3] table matching      -> list[TableCandidate]  (Claim별)
#   -> [4] slot_filler/clarify -> ClarifyResult (slots 완성될 때까지 반복)
#   -> [5] api_client          -> KosisApiResponse
#   -> [6] calculator          -> ComputedResult
#   -> [7] 비교·판정            -> Verdict
#   -> [8] 설명 생성             -> Explanation  (최종 산출물)