interface HeaderProps {
  wide?: boolean;
  // 2026-08-26(5): 목록 화면이 없어지면서, 결과 화면에 있을 때 "처음(URL 입력) 화면으로
  // 돌아가는" 길이 필요해졌다 — 로고/제목을 누르면 홈으로 돌아가게 한다(흔한 관례).
  onLogoClick?: () => void;
}

export function Header({ wide = false, onLogoClick }: HeaderProps) {
  return (
    <header className="sticky top-0 z-10 relative overflow-hidden bg-white/70 px-6 py-4 shadow-[0_1px_0_0_rgba(0,0,0,0.04)] backdrop-blur-xl dark:bg-stone-950/70">
      <div
        className={`relative mx-auto flex flex-wrap items-center justify-between gap-3 ${wide ? "max-w-7xl" : "max-w-6xl"}`}
      >
        <button
          type="button"
          onClick={onLogoClick}
          className="flex items-center gap-3 rounded-lg text-left outline-none focus-visible:ring-2 focus-visible:ring-stone-400"
        >
          <span
            aria-hidden
            className="grid h-10 w-10 flex-none place-items-center rounded-xl bg-stone-600 shadow-sm"
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
            <h1 className="text-xl font-extrabold tracking-tight text-stone-900 dark:text-stone-50">
              AI 뉴스 사실 검증
            </h1>
            <p className="text-xs text-stone-500 dark:text-stone-400">
              기사 속 수치 주장을 KOSIS 공식 통계와 대조한 결과
            </p>
          </div>
        </button>
      </div>
      <div className="absolute inset-x-0 bottom-0 h-px bg-stone-200 dark:bg-stone-800" />
    </header>
  );
}
