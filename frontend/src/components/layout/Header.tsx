import type { ReactNode } from "react";

interface HeaderProps {
  children?: ReactNode;
  wide?: boolean;
}

// 2026-08-22: "고전적"이라는 피드백을 받고 처음엔 그라디언트/글로우를 더 화려하게
// 넣어봤는데, 그다음 피드백은 "아예 다른 느낌"이었다 — 색을 더 쓰는 대신 반대 방향으로
// 가서, Notion/Linear 라이트 모드 같은 미니멀 모노톤으로 갈아엎는다. 그라디언트/아이콘
// 배지를 다 빼고 크고 두꺼운 흑백 타이포그래피로 무게감을 준다 — 색은 이 헤더가 아니라
// 데이터(심각도 배지, 카드 점) 쪽에만 남겨서 그쪽이 더 도드라지게 한다.
export function Header({ children, wide = false }: HeaderProps) {
  return (
    <header className="sticky top-0 z-10 border-b border-gray-900 bg-white px-6 py-5 dark:border-gray-100 dark:bg-gray-950">
      <div
        className={`mx-auto flex flex-wrap items-center justify-between gap-4 ${wide ? "max-w-7xl" : "max-w-6xl"}`}
      >
        <div className="flex items-baseline gap-3">
          <h1 className="text-2xl font-black tracking-tight text-gray-900 dark:text-gray-50">
            AI 뉴스 사실 검증
          </h1>
          <p className="hidden text-sm text-gray-400 dark:text-gray-500 sm:block">
            기사 속 수치 주장을 KOSIS 공식 통계와 대조한 결과
          </p>
        </div>
        {children}
      </div>
    </header>
  );
}
