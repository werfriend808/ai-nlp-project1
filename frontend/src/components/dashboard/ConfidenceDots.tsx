interface ConfidenceDotsProps {
  filled: number; // 0~4
  total?: number;
  label: string;
}

export function ConfidenceDots({ filled, total = 4, label }: ConfidenceDotsProps) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium text-stone-500 dark:text-stone-400">통계 기사 확실성</span>
      <div className="flex items-center gap-2">
        <div className="flex gap-1.5">
          {Array.from({ length: total }).map((_, i) => (
            <span
              key={i}
              className={`h-3.5 w-3.5 rounded-full ${
                i < filled ? "bg-stone-600" : "bg-stone-200 dark:bg-stone-700"
              }`}
            />
          ))}
        </div>
        <span className="text-sm font-semibold text-stone-700 dark:text-stone-300">{label}</span>
      </div>
    </div>
  );
}
