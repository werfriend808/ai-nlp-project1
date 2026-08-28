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
# 담당: A  |  모델: HCX-007 (2026-08-22 HCX-DASH-002에서 교체 — 골든셋 정답률 66.0% -> 90.0%)
# 입력: 기사 본문(str)
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    label: bool          # True = 관련 기사(국가 공식 통계/수치 기반 주장 포함), False = 무관
    score: float          # 0.0 ~ 1.0, 확신도. 0.4~0.6은 애매 구간 → 사람 리뷰 큐로
    reason: str            # 판단 근거 한 문장


# ---------------------------------------------------------------------------
# 2단계 — claim_extractor.py (수치 주장 문장 추출)
# 담당: A  |  모델: HCX-DASH-002 (2026-07-24 HCX-003에서 교체 — 긴 기사 컨텍스트 초과
#   문제 대응, agent/preprocessing/eval_claim_extractor_model.py 실측 근거)
# 입력: 기사 본문(str)  (※ 1단계 결과 자체가 아니라 원본 기사 본문을 다시 받음)
# 출력: Claim의 리스트 (문장 하나하나 따로 호출 X, 기사 전체 넣고 리스트로 한 번에)
# ---------------------------------------------------------------------------

ClaimType = Literal["규모", "증감률", "비교", "전망"]
ComparisonOperator = Literal["증가", "감소", "동일", "초과", "미만"]
# 2026-08-16 추가(C, 팀 공유 예정) — claim_type="규모"에 "수준값"(특정 시점의 총량/절대값,
# 예: "취업자 수는 2857만6000명으로 집계됐다")과 "증감폭"(그 자체가 변화량, 예: "취업자 수가
# 13만5000명 늘어난 것으로 나타났다")이 섞여 있는데 구분할 방법이 없어서, calc_type_router가
# 후자도 전부 "단순조회"(값 하나만 조회)로 보내는 바람에 증감폭 주장을 검증할 방법 자체가
# 없던 버그(실제 배치 재현: 취업자 증감폭 claim 다수가 판단불가로 빠짐)를 고치기 위해 도입.
ValueType = Literal["수준값", "증감폭"]

@dataclass
class Claim:
    sentence: str
    claim_type: ClaimType
    period: Optional[str] = None       # 예: "2024년" / 시점 불명확하면 None
    unit: Optional[str] = None          # 예: "%", "가구" / 없으면 None
    population: Optional[str] = None    # 예: "국내 과수 농가" / 없으면 None
    # --- 신규 (검증 DB 스키마 대응, 팀 상의 후 추가) ---
    # 기존 필드 순서/의미는 그대로 두고 기본값 있는 필드만 뒤에 추가함 —
    # Claim(sentence=..., claim_type=...)처럼 키워드 인자로 생성하는 기존 코드는 안 깨짐.
    statistic_expression: Optional[str] = None  # 기사가 실제로 쓴 지표 표현 (예: "청년 고용")
    value: Optional[float] = None               # 문장의 핵심 수치 (부호 없음 — 방향은 comparison_operator가 담당)
    value_type: Optional[ValueType] = None      # value가 수준값인지 증감폭인지 (claim_type="규모"일 때만 의미 있음)
    comparison_operator: Optional[ComparisonOperator] = None  # 수치의 비교/변화 방향
    comparison_target: Optional[str] = None     # 비교 시점·대상 원문 표현 (예: "전년동월", "정부 전망치")
    comparison_value: Optional[float] = None    # 기사에 명시된 비교 대상 수치 (없으면 None)
    region: Optional[str] = None                # 수치가 적용되는 지역 (예: "전국", "수도권")
    source_org: Optional[str] = None            # 기사가 인용한 기관 (예: "통계청")
    source_report: Optional[str] = None         # 기사가 인용한 조사·보고서명
    # --- 2026-08-21 추가(검색 정확도 개선용, optional dimension) ---
    # age/gender는 population과 달리 "문장에 명시적으로 언급됐을 때만" 채우는 정규화된
    # 구조화 필드다 — 없으면 반드시 None(하드 필터/감점의 근거로 안 쓰이고, 소프트 부스팅에서
    # "값이 있을 때만" 가산점을 주는 용도라 없다고 해서 불이익을 주면 안 됨).
    age: Optional[str] = None                   # 정규화된 연령 구간 (예: "65세 이상", "20~29세"). 명시 안 되면 None.
    gender: Optional[str] = None                # "여성" | "남성". 명시 안 되면 None.
    # search_query: KOSIS 표 검색용 짧은 문구(HyDE 스타일) — "{정규화 지표명} {있는 dimension만} {정규화 기관명}".
    # 별도 LLM 호출 아님(claim_extractor 프롬프트가 이 필드도 같이 생성) — 구체적 수치/연도는
    # 절대 포함하지 않는다(VDB 문서의 "(연월)"이 claim의 대상 시점이 아니라 크롤링 시점이라
    # 넣으면 오히려 노이즈가 됨, 2026-08-21 실측).
    search_query: Optional[str] = None


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
    # 2026-08-21 추가(PHASE 6): table_params.json(64개 수동 카탈로그)에 없는 VDB 전용 표를
    # detail_cache.get_table_detail()로 조회하려면 orgId가 필요하다. kosis_vdb_tables에는
    # 이미 있던 컬럼인데 query_vdb.py가 안 읽어와서 지금까지 후보에 안 실려 있었다.
    org_id: Optional[str] = None


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
    obj_l2: Optional[str] = None   # ← 추가
    prd_se: Optional[str] = None


# ---------------------------------------------------------------------------
# 6단계 — kosis/calculator.py (표 연산)
# 담당: C  |  모델: 불필요 (반드시 파이썬 연산, LLM은 결과 "설명"에만 사용)
# 입력: KosisApiResponse (복수 가능 — 합계/비율/증감 계산 시)
# 출력: ComputedResult
# ---------------------------------------------------------------------------

CalcType = Literal["단순조회", "합계", "비율", "증감", "증감률", "최댓값검증", "최솟값검증"]

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
#   -> [2.5] calc_type_router (agent/orchestrator/calc_type_router.py) -> CalcType | None
#            claim_type/sentence/population/region만 보고 즉시 판단 가능한 것만 먼저
#            거른다(3~4단계를 안 기다림). 해외 국가 언급·전망(claim_type="전망")·스키마 밖
#            값이면 여기서 바로 None -> 호출부가 3~8단계를 건너뛰고 즉시 판단불가 처리.
#            공식 8단계 다이어그램엔 없지만 실제 파이프라인엔 존재하는 라우팅 단계라 여기
#            명시해둠(2026-08-21 추가, 문서 누락 발견).
#   -> [3] table matching      -> list[TableCandidate]  (calc_type_router를 통과한 Claim만)
#   -> [4] slot_filler/clarify -> ClarifyResult (slots 완성될 때까지 반복)
#   -> [5] api_client          -> KosisApiResponse
#   -> [6] calculator          -> ComputedResult
#   -> [7] 비교·판정            -> Verdict
#   -> [8] 설명 생성             -> Explanation  (최종 산출물)
