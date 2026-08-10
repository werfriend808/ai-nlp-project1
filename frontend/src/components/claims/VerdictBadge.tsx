import type { VerdictType } from "../../types/verification";
import { VERDICT_BADGE_CLASS, verdictLabel } from "../../lib/verdictColors";

interface VerdictBadgeProps {
  verdict: VerdictType | null;
}

export function VerdictBadge({ verdict }: VerdictBadgeProps) {
  const label = verdictLabel(verdict);

  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium ${VERDICT_BADGE_CLASS[label]}`}
    >
      {label}
    </span>
  );
}
