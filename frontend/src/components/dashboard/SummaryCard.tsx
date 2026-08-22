import type { ReactNode } from "react";
import type { VerificationRecord } from "../../types/verification";
import { ScoreGauge } from "./ScoreGauge";
import { ConfidenceDots } from "./ConfidenceDots";
import { VerdictIcon } from "./VerdictIcon";
import { VERDICT_COUNT_BOX_CLASS, verdictCountLabel } from "../../lib/verdictColors";

interface SummaryCardProps {
  records: VerificationRecord[];
  reviewFilterActive: boolean;
  onToggleReviewFilter: () => void;
}

interface ItemRow {
  icon: ReactNode;
  label: string;
  description: string;
  count: number;
  colorClass: string;
}

export function SummaryCard({ records, reviewFilterActive, onToggleReviewFilter }: SummaryCardProps) {
  const total = records.length;
  // verdictCountLabel: 일치/불일치는 그대로, 그 외(null=표매칭 신뢰도 낮음, "판단불가"=검증
  // 대상 자체가 아님/미래예측 등)는 전부 "애매" 하나로 묶는다 — 기사 목록 배지(VerdictSummaryBadges)
  // 랑 똑같은 기준을 써야 "판단불가" 건이 어느 항목에도 안 잡히고 사라지는 불일치가 안 생긴다.
  const 일치 = records.filter((r) => verdictCountLabel(r.verification_result) === "일치").length;
  const 불일치 = records.filter((r) => verdictCountLabel(r.verification_result) === "불일치").length;
  const 검토필요 = records.filter((r) => verdictCountLabel(r.verification_result) === "애매").length;
  const matched = 일치 + 불일치;

  // "종합 평가" 점수: 애초에 검증 자체가 불가능한 주제(해외 통계·기업 실적 등)까지 억지로
  // 다 맞히려는 지표가 아니라, "일치/불일치처럼 확정 판정까지 도달한 비율"을 정직하게 보여준다.
  const matchRate = total > 0 ? (matched / total) * 100 : 0;
  const scoreStatus =
    matchRate >= 60 ? "양호" : matchRate >= 30 ? "보통" : "낮음";
  const scoreColorClass =
    matchRate >= 60
      ? "stroke-emerald-500"
      : matchRate >= 30
        ? "stroke-amber-500"
        : "stroke-red-500";

  const avgScore =
    total > 0
      ? records.reduce((sum, r) => sum + (r.classifier_score ?? 0), 0) / total
      : 0;
  const confidenceFilled = Math.round(avgScore * 4);
  const confidenceLabel = avgScore >= 0.8 ? "높음" : avgScore >= 0.6 ? "보통" : "낮음";

  // 2026-08-22 추가: "지금 처리된 기사가 몇 건인지 한눈에 안 보인다"는 피드백 — records는
  // claim(주장) 단위라 그대로 세면 기사 수가 아니라 주장 수가 나온다. article_title 기준
  // 고유 기사 수를 따로 센다.
  const articleCount = new Set(records.map((r) => r.article_title)).size;

  const rows: ItemRow[] = [
    {
      icon: <VerdictIcon verdict="일치" />,
      label: "일치",
      description: "기사 수치가 KOSIS 공식 통계와 일치함",
      count: 일치,
      colorClass: VERDICT_COUNT_BOX_CLASS["일치"],
    },
    {
      icon: <VerdictIcon verdict="불일치" />,
      label: "불일치",
      description: "기사 수치가 KOSIS 공식 통계와 차이가 있음",
      count: 불일치,
      colorClass: VERDICT_COUNT_BOX_CLASS["불일치"],
    },
    {
      icon: <VerdictIcon verdict="애매" />,
      label: "검토 필요",
      description: "표 매칭 신뢰도가 낮거나 KOSIS로 검증 불가능한 주제",
      count: 검토필요,
      colorClass: VERDICT_COUNT_BOX_CLASS["애매"],
    },
  ];

  return (
    <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm transition-shadow duration-200 hover:shadow-md hover:shadow-indigo-500/5 dark:border-gray-700 dark:bg-gray-900">
      <div className="flex items-center gap-2 border-b border-gray-100 px-6 py-4 dark:border-gray-800">
        <span className="h-2 w-2 rounded-full bg-gradient-to-br from-indigo-500 to-fuchsia-500" />
        <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
          품질 검증 대시보드
        </span>
      </div>

      <div className="flex flex-col gap-6 p-6">
        <div className="flex flex-wrap items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <ScoreGauge value={matchRate} colorClass={scoreColorClass} />
            <div>
              <p className="text-xs font-medium text-gray-500 dark:text-gray-400">종합 평가</p>
              <p className="text-lg font-bold text-gray-900 dark:text-gray-50">{scoreStatus}</p>
              <p className="text-xs text-gray-400">표 매칭까지 도달한 비율 기준</p>
            </div>
          </div>
          <div>
            <p className="text-xs font-medium text-gray-500 dark:text-gray-400">누적 기사 처리량</p>
            <p className="text-lg font-bold text-gray-900 dark:text-gray-50">
              {articleCount.toLocaleString()} 기사 처리
            </p>
          </div>
          <ConfidenceDots filled={confidenceFilled} label={confidenceLabel} />
        </div>

        <div className="flex flex-col gap-1">
          <p className="mb-1 text-xs font-medium text-gray-500 dark:text-gray-400">평가 항목</p>
          {rows.map((row) => (
            <div
              key={row.label}
              className="flex items-center justify-between gap-3 border-t border-gray-100 py-2.5 first:border-t-0 dark:border-gray-800"
            >
              <div className="flex items-center gap-3">
                {row.icon}
                <div>
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{row.label}</p>
                  <p className="text-xs text-gray-400">{row.description}</p>
                </div>
              </div>
              <span
                className={`shrink-0 rounded-full px-2.5 py-1 text-xs font-semibold ${row.colorClass}`}
              >
                {row.count}건
              </span>
            </div>
          ))}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-gray-50 p-4 dark:bg-gray-800">
          <div>
            <p className="text-xs font-medium text-gray-500 dark:text-gray-400">검토 코멘트</p>
            <p className="mt-1 text-sm text-gray-700 dark:text-gray-300">
              {matchRate >= 60
                ? "표 매칭 성공률이 양호합니다."
                : "표 매칭 성공률이 낮습니다 — KOSIS 카탈로그 확장이 필요합니다."}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-right">
              <p className="text-xs font-medium text-gray-500 dark:text-gray-400">권장 조치</p>
              <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">{검토필요}건</p>
            </div>
            <button
              type="button"
              onClick={onToggleReviewFilter}
              className={`rounded-lg px-3 py-2 text-sm font-medium transition ${
                reviewFilterActive
                  ? "bg-indigo-600 text-white hover:bg-indigo-700"
                  : "bg-white text-indigo-600 ring-1 ring-indigo-200 hover:bg-indigo-50 dark:bg-gray-900 dark:ring-indigo-800"
              }`}
            >
              {reviewFilterActive ? "전체 보기" : "상세 보기 →"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
