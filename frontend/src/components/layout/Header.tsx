import type { ReactNode } from "react";

interface HeaderProps {
  children?: ReactNode;
  wide?: boolean;
}

export function Header({ children, wide = false }: HeaderProps) {
  return (
    <header className="sticky top-0 z-10 relative bg-white/80 px-6 py-4 backdrop-blur-md dark:bg-gray-950/80">
      <div
        className={`mx-auto flex flex-wrap items-center justify-between gap-3 ${wide ? "max-w-7xl" : "max-w-6xl"}`}
      >
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className="grid h-8 w-8 flex-none place-items-center rounded-[10px] bg-gradient-to-br from-indigo-500 to-fuchsia-500 shadow-sm"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4 text-white" fill="none">
              <path
                d="M5 12.5l4.5 4.5L19 7"
                stroke="currentColor"
                strokeWidth={2.5}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <div>
            <h1 className="bg-gradient-to-r from-indigo-600 to-fuchsia-600 bg-clip-text text-lg font-bold tracking-tight text-transparent dark:from-indigo-400 dark:to-fuchsia-400">
              AI 뉴스 사실 검증 대시보드
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              기사 속 수치 주장을 KOSIS 공식 통계와 대조한 결과
            </p>
          </div>
        </div>
        {children}
      </div>
      <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-indigo-500 via-fuchsia-500 to-transparent opacity-60" />
    </header>
  );
}
