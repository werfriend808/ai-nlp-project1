interface ConfidenceDotsProps {
  filled: number; // 0~4
  total?: number;
  label: string;
}

export function ConfidenceDots({ filled, total = 4, label }: ConfidenceDotsProps) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium text-gray-500 dark:text-gray-400">신뢰도 (Confidence)</span>
      <div className="flex items-center gap-2">
        <div className="flex gap-1.5">
          {Array.from({ length: total }).map((_, i) => (
            <span
              key={i}
              className={`h-3.5 w-3.5 rounded-full ${
                i < filled ? "bg-indigo-500" : "bg-gray-200 dark:bg-gray-700"
              }`}
            />
          ))}
        </div>
        <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">{label}</span>
      </div>
    </div>
  );
}
