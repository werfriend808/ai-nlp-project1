import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { VerificationRecord } from "../../types/verification";

interface FunnelChartProps {
  records: VerificationRecord[];
}

const COLORS: Record<string, string> = {
  일치: "#10b981",
  불일치: "#ef4444",
  판단불가: "#9ca3af",
  애매: "#f59e0b",
};

// 주의: 이건 "최종 판정 결과 분포"이지, 1~8단계 각각에서 몇 건이 걸러졌는지 보여주는
// 진짜 단계별 퍼널이 아니다. batch_runner.py가 지금은 최종 결과 리스트만 반환하고
// 1단계 필터링/2단계 추출실패/3단계 매칭실패/4단계 되묻기 미해결 같은 중간 단계
// 탈락 건수는 print()로만 찍고 구조화된 값으로 반환하지 않는다 — 그 계측이 추가되면
// (팀원 확인 후 feat/c-funnel-tracking에서 진행 예정) 이 컴포넌트를 진짜 단계별
// 퍼널로 확장할 수 있다.
function buildVerdictDistribution(records: VerificationRecord[]) {
  const counts: Record<string, number> = { 일치: 0, 불일치: 0, 판단불가: 0, 애매: 0 };
  for (const r of records) {
    const label = r.verification_result ?? "애매";
    counts[label] = (counts[label] ?? 0) + 1;
  }
  return Object.entries(counts).map(([verdict, count]) => ({ verdict, count }));
}

export function FunnelChart({ records }: FunnelChartProps) {
  const data = buildVerdictDistribution(records);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-700 dark:bg-gray-900">
      <h2 className="mb-1 text-sm font-semibold text-gray-700 dark:text-gray-300">
        최종 판정 분포
      </h2>
      <p className="mb-4 text-xs text-gray-400">
        (단계별 탈락 건수 퍼널은 아직 계측 전 — 최종 판정 결과 기준 분포입니다)
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
          <XAxis dataKey="verdict" tick={{ fontSize: 12 }} />
          <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
          <Tooltip />
          <Bar dataKey="count" radius={[4, 4, 0, 0]}>
            {data.map((entry) => (
              <Cell key={entry.verdict} fill={COLORS[entry.verdict]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
