import type { VerificationRecord } from "../../types/verification";
import { VERDICT_COUNT_BOX_CLASS, verdictCountLabel } from "../../lib/verdictColors";

interface VerdictSummaryBadgesProps {
  records: VerificationRecord[];
}

const ORDER = ["일치", "불일치", "애매"] as const;

// 기사 하나에 딸린 여러 claim 레코드를 판정별로 묶어, 색깔 박스 안에 개수만 보여주는
// 미니멀한 요약으로 표시한다 (일치=초록, 불일치=빨강, 그 외(판단불가/표매칭 신뢰도 낮음)는
// 전부 "애매" 하나로 묶어서 주황 — 색이 어차피 같아서 따로 세면 "1, 1"처럼 같은 색 박스가
// 중복으로 보여 혼란스럽다).
export function VerdictSummaryBadges({ records }: VerdictSummaryBadgesProps) {
  const counts = new Map<(typeof ORDER)[number], number>();
  for (const r of records) {
    const label = verdictCountLabel(r.verification_result);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {ORDER.filter((label) => counts.has(label)).map((label) => (
        <span
          key={label}
          title={label}
          className={`inline-flex h-6 min-w-6 items-center justify-center rounded px-1.5 text-xs font-semibold ${VERDICT_COUNT_BOX_CLASS[label]}`}
        >
          {counts.get(label)}
        </span>
      ))}
    </div>
  );
}
