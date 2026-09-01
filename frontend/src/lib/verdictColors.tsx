import type { ReactNode } from "react";
import type { VerdictType } from "../types/verification";
import { VerdictIcon } from "../components/dashboard/VerdictIcon";

// verification_result가 null인 레코드(판정 자체가 안 된 애매/표매칭_불충분 상태)는
// "애매"로 취급 — 화면 전체(뱃지/하이라이트/카드)에서 일관되게 쓰는 색 매핑.
export type VerdictLabel = VerdictType | "애매";

export function verdictLabel(verdict: VerdictType | null): VerdictLabel {
  return verdict ?? "애매";
}

// 기사 목록 카드의 "요약 개수 박스"에서만 쓰는 라벨. 판단불가/애매(표매칭 신뢰도 낮음)는
// 색도 똑같이 주황이라 박스를 따로 두면 "1, 1"처럼 같은 색 박스 두 개가 나란히 나와
// 헷갈린다 — 둘 다 "확실하지 않음"이라는 같은 의미라 하나로 합쳐서 센다.
export function verdictCountLabel(verdict: VerdictType | null): "일치" | "불일치" | "애매" {
  if (verdict === "일치" || verdict === "불일치") return verdict;
  return "애매";
}

export const VERDICT_BADGE_CLASS: Record<VerdictLabel, string> = {
  일치: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
  불일치: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  판단불가: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
  애매: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
};

export const VERDICT_HIGHLIGHT_CLASS: Record<VerdictLabel, string> = {
  일치:
    "bg-emerald-50 font-semibold text-emerald-900 underline decoration-emerald-500 decoration-2 underline-offset-2 dark:bg-emerald-900/20 dark:text-emerald-200",
  불일치:
    "bg-red-50 font-semibold text-red-900 underline decoration-red-500 decoration-2 underline-offset-2 dark:bg-red-900/20 dark:text-red-200",
  판단불가:
    "bg-amber-50 font-semibold text-amber-900 underline decoration-amber-500 decoration-2 underline-offset-2 dark:bg-amber-900/20 dark:text-amber-200",
  애매:
    "bg-amber-50 font-semibold text-amber-900 underline decoration-amber-500 decoration-2 underline-offset-2 dark:bg-amber-900/20 dark:text-amber-200",
};

// 원문 하이라이트 위에 뜨는 번호 배지 색 — 하이라이트와 같은 계열 색을 써서 번호가 하이라이트와
// 분리된 별개 요소가 아니라 한 세트로 보이게 한다.
export const VERDICT_MARKER_CLASS: Record<VerdictLabel, string> = {
  일치: "bg-emerald-500 text-white",
  불일치: "bg-red-500 text-white",
  판단불가: "bg-amber-500 text-white",
  애매: "bg-amber-500 text-white",
};

export const VERDICT_COUNT_BOX_CLASS: Record<"일치" | "불일치" | "애매", string> = {
  일치: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
  불일치: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  애매: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
};

// ArticleDetail.tsx의 text-xs 요약 pill(px-2.5 py-1) 안에 들어가는 아이콘이라 h-6(24px)은
// 텍스트 크기에 비해 너무 커서 pill 세로 패딩이 아이콘에 맞춰 늘어나 보이는 문제가
// 있었다(실측 지적, 2026-09-01) — 내부 글리프 자체가 h-3.5(14px) 고정이므로 바깥 원을
// h-4(16px)로 줄여서 pill의 text-xs 리듬에 맞춘다.
export const VERDICT_ICON: Record<"일치" | "불일치" | "애매", ReactNode> = {
  일치: <VerdictIcon verdict="일치" className="h-4 w-4" />,
  불일치: <VerdictIcon verdict="불일치" className="h-4 w-4" />,
  애매: <VerdictIcon verdict="애매" className="h-4 w-4" />,
};

// 기사 목록 테이블 행의 왼쪽 강조 테두리 색.
export const VERDICT_ACCENT_BORDER_CLASS: Record<"일치" | "불일치" | "애매", string> = {
  일치: "border-l-emerald-500",
  불일치: "border-l-red-500",
  애매: "border-l-amber-500",
};
