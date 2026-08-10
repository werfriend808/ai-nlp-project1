export function Header() {
  return (
    <header className="border-b border-gray-200 bg-white px-6 py-4 dark:border-gray-700 dark:bg-gray-900">
      <h1 className="text-2xl font-bold text-gray-950 dark:text-gray-50">
        AI 뉴스 사실 검증 대시보드
      </h1>
      <p className="text-sm text-gray-500 dark:text-gray-400">
        기사 속 수치 주장을 KOSIS 공식 통계와 대조한 결과
      </p>
    </header>
  );
}
