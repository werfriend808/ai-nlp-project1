// db/store.py의 verifications 테이블 스키마(33개 컬럼)를 그대로 옮긴 타입.
// db/export_json.py로 내보낸 JSON 배열의 원소 하나가 이 형태다.

export type ClaimType = "규모" | "증감률" | "비교" | "전망";
export type ComparisonOperator = "증가" | "감소" | "동일" | "초과" | "미만";
export type VerdictType = "일치" | "불일치" | "판단불가";
export type GapType = "수치" | "기간" | "모집단" | "과장표현";
export type VerificationPossible = "가능" | "애매" | "불가";

export interface VerificationRecord {
  id: number;
  result_id: string;
  article_title: string;
  article_url: string | null;
  claim_sentence: string;
  claim_type: ClaimType;
  statistic_expression: string | null;
  normalized_statistic_name: string | null;
  statistic_category: string | null;
  value: number | null;
  unit: string | null;
  comparison_operator: ComparisonOperator | null;
  comparison_target: string | null;
  comparison_value: number | null;
  time_expression: string | null;
  reference_time: string | null;
  population: string | null;
  region: string | null;
  source_org: string | null;
  source_report: string | null;
  kosis_table_id: string | null;
  kosis_table: string | null;
  kosis_item: string | null;
  kosis_dimension: Record<string, string | null> | null;
  calculation_required: 0 | 1 | null;
  calculation_type: string | null;
  verification_possible: VerificationPossible;
  ambiguity_reason: string | null;
  verification_result: VerdictType | null;
  mismatch_reason: GapType | null;
  evidence: string | null;
  classifier_score: number | null;
  reviewer_agrees: 0 | 1 | null;
  reviewer_corrected_verdict: string | null;
  created_at: string;
}
