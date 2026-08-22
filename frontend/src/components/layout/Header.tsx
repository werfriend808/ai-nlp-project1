import type { ReactNode } from "react";

interface HeaderProps {
  children?: ReactNode;
  wide?: boolean;
}

export function Header({ children, wide = false }: HeaderProps) {
  return (
    <header className="sticky top-0 z-10 relative overflow-hidden bg-white/70 px-6 py-4 shadow-[0_1px_0_0_rgba(0,0,0,0.04)] backdrop-blur-xl dark:bg-gray-950/70">
      {/* 2026-08-22: 헤더가 "고전적"이라는 피드백 — 은은한 그라디언트 블롭(glow)을 로고
          뒤에 깔아서 요즘 SaaS 대시보드에서 흔한 "떠 있는 유리 패널" 느낌을 준다. 텍스트
          가독성에 영향 없게 blur를 크게 주고 opacity를 낮게 유지한다. */}
      <div
        aria-hidden
        className="pointer-events-none absolute -left-10 -top-16 h-40 w-40 rounded-full bg-gradient-to-br from-indigo-400 via-fuchsia-400 to-transparent opacity-20 blur-3xl dark:opacity-30"
      />
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
