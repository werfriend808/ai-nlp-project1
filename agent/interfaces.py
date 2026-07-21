"""공통 타입 전체 (1~8단계 입출력 계약)"""
from dataclasses import dataclass, field


@dataclass
class ClassificationResult:
    label: str
    score: float
    reason: str


@dataclass
class Claim:
    sentence: str
    claim_type: str
    period: str
    unit: str
    population: str


@dataclass
class TableCandidate:
    table_id: str
    table_name: str
    score: float
    required_slots: list[str] = field(default_factory=list)


Slots = dict[str, str]


@dataclass
class ComputedResult:
    value: float
    unit: str


@dataclass
class Verdict:
    verdict: str
    gap_type: str
    reason: str
