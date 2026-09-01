interface ScoreGaugeProps {
  value: number; // 0-100
  colorClass: string; // Tailwind stroke-* class for the progress arc
}

export function ScoreGauge({ value, colorClass }: ScoreGaugeProps) {
  // 96px였을 때 "100%" 텍스트가 원 안쪽 여백 없이 꽉 차 보인다는 지적(2026-09-01) —
  // 텍스트 크기는 그대로 두고 원 자체를 키워서 텍스트 주변에 숨 쉴 공간을 만든다.
  const size = 128;
  const strokeWidth = 10;
  const center = size / 2;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - Math.min(100, Math.max(0, value)) / 100);

  return (
    <div className="relative h-32 w-32 shrink-0">
      {/* 예전엔 두 줄(100%, 매칭률)을 각각 <text>로 따로 그리고 y좌표를 손으로 어림
          잡아 배치했는데(center-8 / center+14), 그건 "두 줄을 합친 덩어리가 정중앙에
          오게"가 아니라 "각 줄을 대충 그럴듯한 위치에" 놓는 거라 값이 바뀔 때마다
          미묘하게 안 맞아 보였다(실측 지적, 2026-09-01 — "100%랑 매칭률까지 합쳐서
          정중앙"을 원한다는 요청). foreignObject로 HTML을 SVG 안에 넣고
          flex items-center justify-center를 쓰면, 두 줄을 실제로 렌더링한 뒤 그
          "덩어리 전체"의 실측 높이를 기준으로 중앙 정렬해준다 — 픽셀을 손으로
          맞출 필요 없이 항상 정확하다. */}
      <svg width={size} height={size}>
        <g transform={`rotate(-90 ${center} ${center})`}>
          <circle
            cx={center}
            cy={center}
            r={radius}
            strokeWidth={strokeWidth}
            className="fill-none stroke-gray-100 dark:stroke-gray-800"
          />
          <circle
            cx={center}
            cy={center}
            r={radius}
            strokeWidth={strokeWidth}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className={`fill-none transition-all duration-700 ease-out ${colorClass}`}
          />
        </g>
        <foreignObject x={0} y={0} width={size} height={size}>
          <div className="flex h-full w-full flex-col items-center justify-center gap-0.5">
            <span className="text-2xl leading-none font-bold text-gray-900 dark:text-gray-50">
              {Math.round(value)}%
            </span>
            <span className="text-[12px] leading-none text-gray-400">매칭률</span>
          </div>
        </foreignObject>
      </svg>
    </div>
  );
}
