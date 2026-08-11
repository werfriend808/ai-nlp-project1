import type { ReactNode } from "react";

interface HeaderProps {
  children?: ReactNode;
  wide?: boolean;
}

export function Header({ children, wide = false }: HeaderProps) {
  return (
    <header className="sticky top-0 z-10 border-b border-gray-100 bg-white/80 px-6 py-4 backdrop-blur-md dark:border-gray-800 dark:bg-gray-950/80">
      <div
        className={`mx-auto flex flex-wrap items-center justify-between gap-3 ${wide ? "max-w-6xl" : "max-w-4xl"}`}
      >
        <div>
          <h1 className="text-base font-bold tracking-tight text-gray-900 dark:text-gray-50">
            AI 뉴스 사실 검증 대시보드
          </h1>
          <p className="text-xs text-gray-500 dark:text-gray-400">
            기사 속 수치 주장을 KOSIS 공식 통계와 대조한 결과
          </p>
        </div>
        {children}
      </div>
    </header>
  );
}
