// 2026-08-22: 이모지(✅❌🔍)가 "조잡해 보인다"는 피드백 — 헤더/검증 버튼에 이미 쓰고 있는
// 스트로크 기반 SVG 아이콘 스타일(viewBox 24x24, strokeWidth 2.5, 둥근 선끝)로 통일해서
// 색이 있는 원형 배지 안에 넣었다. verdictColors.ts의 VERDICT_ICON을 이 컴포넌트로
// 대체하면, ArticleListPanel/ArticleDetail 등 기존 소비처는 {VERDICT_ICON[verdict]}를
// 그대로 렌더링하는 자리라 별도 수정 없이 이 JSX가 그 자리에 그대로 들어간다.
//
// 2026-08-26(3): 진한 solid(→반투명 solid) 배지 + 흰색 글리프 조합이 두 차례 다
// "칙칙하다/똥색 같다"는 피드백을 받았다. 참고 이미지의 아이콘 배지들은 전부 옅은
// 파스텔 배경 + 그 안의 진한 색 아이콘 조합이라, 이 컴포넌트도 흰색 글리프를 버리고
// currentColor로 바꿔서 옅은 배경(-100) 위에 진한 색(-600) 아이콘이 올라가는 방식으로
// 바꿨다 — StatIcons.tsx가 이미 쓰던 것과 같은 패턴.
import type { ReactElement } from "react";

type VerdictKind = "일치" | "불일치" | "애매";

interface VerdictIconProps {
  verdict: VerdictKind;
  className?: string;
}

const BADGE_CLASS: Record<VerdictKind, string> = {
  일치: "bg-match-100 text-match-600 dark:bg-match-900/40 dark:text-match-300",
  불일치: "bg-mismatch-100 text-mismatch-600 dark:bg-mismatch-900/40 dark:text-mismatch-300",
  애매: "bg-caution-100 text-caution-600 dark:bg-caution-900/40 dark:text-caution-300",
};

function CheckGlyph() {
  return (
    <path
      d="M5 12.5l4.5 4.5L19 7"
      stroke="currentColor"
      strokeWidth={2.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
    />
  );
}

function CrossGlyph() {
  return (
    <path
      d="M7 7l10 10M17 7L7 17"
      stroke="currentColor"
      strokeWidth={2.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      fill="none"
    />
  );
}

function SearchGlyph() {
  return (
    <>
      <circle cx="10.5" cy="10.5" r="5.5" stroke="currentColor" strokeWidth={2.2} fill="none" />
      <path d="M18.5 18.5l-3.6-3.6" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" />
    </>
  );
}

const GLYPHS: Record<VerdictKind, () => ReactElement> = {
  일치: CheckGlyph,
  불일치: CrossGlyph,
  애매: SearchGlyph,
};

export function VerdictIcon({ verdict, className = "h-7 w-7" }: VerdictIconProps) {
  const Glyph = GLYPHS[verdict];
  return (
    <span
      className={`grid shrink-0 place-items-center rounded-full ${BADGE_CLASS[verdict]} ${className}`}
    >
      <svg viewBox="0 0 24 24" className="h-3.5 w-3.5">
        <Glyph />
      </svg>
    </span>
  );
}
