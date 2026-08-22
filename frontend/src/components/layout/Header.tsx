import type { ReactNode } from "react";

interface HeaderProps {
  children?: ReactNode;
  wide?: boolean;
}

export function Header({ children, wide = false }: HeaderProps) {
  return (
    <header className="sticky top-0 z-10 relative overflow-hidden bg-white/70 px-6 py-4 shadow-[0_1px_0_0_rgba(0,0,0,0.04)] backdrop-blur-xl dark:bg-gray-950/70">
      <div
        className={`relative mx-auto flex flex-wrap items-center justify-between gap-3 ${wide ? "max-w-7xl" : "max-w-6xl"}`}
      >
        <div className="flex items-center gap-3">
          <span
            aria-hidden
            className="grid h-10 w-10 flex-none place-items-center rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-fuchsia-500 shadow-lg shadow-indigo-500/30"
          >
            <svg viewBox="0 0 24 24" className="h-5 w-5 text-white" fill="none">
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
            <h1 className="bg-gradient-to-r from-indigo-600 via-purple-600 to-fuchsia-600 bg-clip-text text-xl font-extrabold tracking-tight text-transparent dark:from-indigo-400 dark:via-purple-400 dark:to-fuchsia-400">
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
