"""agent/kosis/dynamic_slot_mapping.py — PHASE 7: table_params.json(64개 수동 카탈로그)
밖에서(VDB) 찾은 표에 대해, claim의 슬롯(region/age/gender)을 실제 KOSIS 축 코드로
매핑한다.

수동 카탈로그 표는 사람이 미리 code_map을 검증해뒀지만, VDB 표는 agent.kosis.detail_cache가
그때그때 가져온 code_maps(라벨→코드)만 있다 — 이름 매칭으로 코드를 찾아야 하므로 완벽하지
않을 수 있다. 그래서 SUCCESS/AMBIGUOUS/NOT_FOUND/INCOMPATIBLE 네 상태를 명시적으로 구분하고,
하나라도 SUCCESS가 아니면 전체를 not_verifiable로 처리한다 — 애매한 cell을 억지로 골라
TRUE/FALSE를 만들어내는 것보다 NOT_VERIFIABLE이 안전하다는 원칙(2026-08-21 확정)에 따른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

SlotStatus = Literal["success", "ambiguous", "not_found", "incompatible", "not_needed"]


@dataclass
class SlotMappingResult:
    overall_status: Literal["success", "not_verifiable"]
    resolved: dict[str, str] = field(default_factory=dict)  # axis_name(표 기준) -> code
    details: dict[str, SlotStatus] = field(default_factory=dict)  # claim 필드명 -> 상태
    reason: Optional[str] = None


# claim 필드 -> 그 값이 매칭될 만한 축 이름 후보(부분 문자열 포함 매칭용). KOSIS 축 이름은
# 표마다 "시도별"/"행정구역별"/"지역별" 등으로 제각각이라 정확히 하나로 고정 못 한다.
_FIELD_TO_AXIS_HINTS = {
    "region": ["지역", "시도", "행정구역", "시군구"],
    "age": ["연령"],
    "gender": ["성별"],
}

# 2026-08-23 실측 발견(건설업 취업자 claim, gold=DT_1DA7E26S): region="전국"인데 표에
# 지역 축 자체가 없는 표(예: DT_1DA7E06S_NEW "산업별 취업자")를 "incompatible"로 걸러서
# 값 조회 자체가 안 됐다. 근데 "전국"은 "지역을 좁혀야 한다"는 뜻이 아니라 정반대로 "지역
# 구분 없이 전체"라는 뜻이라, 애초에 지역 축이 없는 표(=이미 전국 집계뿐인 표)와 논리적으로
# 모순이 없다 — "서울만" 요청했는데 지역 축이 없는 경우(진짜 incompatible)와는 달라야 한다.
_NATIONWIDE_REGION_VALUES = {"전국", "전체", "전국 평균", "전국평균"}


def _normalize_whitespace(text: str) -> str:
    """공백 유무 차이(claim="65세 이상" vs 코드맵 라벨="65세이상")를 흡수한다 —
    source_filter.py의 동일 목적 함수와 같은 원칙(2026-08-05 도입 패턴 재사용)."""
    return "".join(text.split())


def _find_axis_for_field(field_name: str, code_maps: dict) -> Optional[str]:
    hints = _FIELD_TO_AXIS_HINTS.get(field_name, [])
    for axis_name in code_maps:
        if any(h in axis_name for h in hints):
            return axis_name
    return None


def _resolve_value(value: str, code_map: dict) -> tuple[SlotStatus, Optional[str]]:
    if value in code_map:
        return "success", code_map[value]

    normalized_value = _normalize_whitespace(value)
    # 양방향 부분 문자열 매칭(claim="서울" vs 코드맵 라벨="서울특별시", "65세 이상" vs
    # "65세이상" 같은 공백/표기 차이를 흡수).
    matches = [
        (label, code)
        for label, code in code_map.items()
        if normalized_value in _normalize_whitespace(label) or _normalize_whitespace(label) in normalized_value
    ]
    if len(matches) == 1:
        return "success", matches[0][1]
    if len(matches) > 1:
        return "ambiguous", None
    return "not_found", None


def map_claim_slots(claim, code_maps: dict) -> SlotMappingResult:
    """claim.region/age/gender 중 값이 있는 것만 code_maps에서 코드로 변환한다.
    값이 없는 필드(optional dimension)는 검사 대상이 아니다 — "not_needed"로 표시하고
    전체 판정에 영향 안 준다(null이라고 감점/실패 처리하지 않음)."""
    resolved: dict[str, str] = {}
    details: dict[str, SlotStatus] = {}

    for field_name in ("region", "age", "gender"):
        value = getattr(claim, field_name, None)
        if not value:
            details[field_name] = "not_needed"
            continue

        axis_name = _find_axis_for_field(field_name, code_maps)
        if axis_name is None:
            if field_name == "region" and _normalize_whitespace(value) in {
                _normalize_whitespace(v) for v in _NATIONWIDE_REGION_VALUES
            }:
                # "전국"은 지역을 좁히라는 게 아니라 안 좁혀도 된다는 뜻 — 지역 축이 아예
                # 없는 표(=이미 전국 단위)와 모순되지 않는다.
                details[field_name] = "not_needed"
                continue
            details[field_name] = "incompatible"  # 이 표엔 해당 축 자체가 없음
            continue

        status, code = _resolve_value(value, code_maps[axis_name])
        details[field_name] = status
        if status == "success":
            resolved[axis_name] = code

    if any(s in ("ambiguous", "not_found", "incompatible") for s in details.values()):
        bad = [f"{k}={v}" for k, v in details.items() if v not in ("success", "not_needed")]
        return SlotMappingResult(
            overall_status="not_verifiable", resolved=resolved, details=details,
            reason=f"슬롯 매핑 실패: {', '.join(bad)}",
        )
    return SlotMappingResult(overall_status="success", resolved=resolved, details=details)
