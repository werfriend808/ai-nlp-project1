import type { ArticleGroup } from "../../lib/articles";
import { verdictCountLabel } from "../../lib/verdictColors";
import { kosisTableUrl } from "../../lib/kosis";
import { ScoreGauge } from "../dashboard/ScoreGauge";
import { ConfidenceDots } from "../dashboard/ConfidenceDots";

interface InsightsPanelProps {
  group: ArticleGroup;
  tableOrgIds: Record<string, string>;
}

// 이 기사에서 검증에 쓰인 KOSIS 표들을 중복 없이 뽑는다.
function uniqueKosisTables(group: ArticleGroup): { id: string; name: string }[] {
  const seen = new Map<string, string>();
  for (const r of group.records) {
    if (r.kosis_table_id && r.kosis_table && !seen.has(r.kosis_table_id)) {
      seen.set(r.kosis_table_id, r.kosis_table);
    }
  }
  return Array.from(seen, ([id, name]) => ({ id, name }));
}

function summaryText(total: number, 일치: number, 불일치: number): string {
  if (불일치 > 0) {
    return `⚠️ 이 기사의 수치 주장 ${total}건 중 ${불일치}건이 KOSIS 공식 통계와 차이가 있습니다.`;
  }
  if (일치 > 0) {
    return `✅ 검증 가능했던 주장은 KOSIS 공식 통계와 일치합니다 (${일치}/${total}건).`;
  }
  return `🔍 이 기사의 수치 주장 ${total}건은 표 매칭 신뢰도가 낮거나 KOSIS로 검증하기 어려운 주제입니다.`;
}

export function InsightsPanel({ group, tableOrgIds }: InsightsPanelProps) {
  const total = group.records.length;
  const 일치 = group.records.filter((r) => verdictCountLabel(r.verification_result) === "일치").length;
  const 불일치 = group.records.filter((r) => verdictCountLabel(r.verification_result) === "불일치").length;
  const matchRate = total > 0 ? ((일치 + 불일치) / total) * 100 : 0;
  const scoreColorClass =
    matchRate >= 60 ? "stroke-emerald-500" : matchRate >= 30 ? "stroke-amber-500" : "stroke-red-500";

  const avgScore =
    total > 0 ? group.records.reduce((sum, r) => sum + (r.classifier_score ?? 0), 0) / total : 0;
  const confidenceFilled = Math.round(avgScore * 4);
  const confidenceLabel = avgScore >= 0.8 ? "높음" : avgScore >= 0.6 ? "보통" : "낮음";

  const kosisTables = uniqueKosisTables(group);

  return (
    <div className="sticky top-24 flex h-fit flex-col gap-5 overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm transition-shadow duration-200 hover:shadow-md hover:shadow-indigo-500/5 dark:border-gray-700 dark:bg-gray-900">
      <div className="flex items-center gap-2 border-b border-gray-100 px-5 py-4 dark:border-gray-800">
        <span className="h-2 w-2 rounded-full bg-gradient-to-br from-indigo-500 to-fuchsia-500" />
        <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">Fact-Check Insights</span>
      </div>

      <div className="flex flex-col gap-5 px-5 pb-5">
        <div>
          <p className="mb-2 text-xs font-medium text-gray-500 dark:text-gray-400">AI 분석 요약</p>
          <p className="text-sm text-gray-700 dark:text-gray-300">
            {summaryText(total, 일치, 불일치)}
          </p>
        </div>

        <div>
          <p className="mb-3 text-xs font-medium text-gray-500 dark:text-gray-400">핵심 지표</p>
          <div className="flex flex-col gap-4">
            <div className="flex justify-center">
              <ScoreGauge value={matchRate} colorClass={scoreColorClass} />
            </div>
            <div className="flex justify-start">
              <ConfidenceDots filled={confidenceFilled} label={confidenceLabel} />
            </div>
          </div>
        </div>

        {kosisTables.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-medium text-gray-500 dark:text-gray-400">
              관련 KOSIS 데이터
            </p>
            <ul className="flex flex-col gap-1.5">
              {kosisTables.map((t) => (
                <li key={t.id}>
                  <a
                    href={kosisTableUrl(t.id, tableOrgIds[t.id], t.name)}
                    target="_blank"
                    rel="noreferrer"
                    className="text-xs text-indigo-600 hover:underline dark:text-indigo-400"
                  >
                    🔗 {t.name}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
