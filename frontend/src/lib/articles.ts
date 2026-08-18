import type { VerificationRecord } from "../types/verification";
import { verdictCountLabel, type VerdictLabel } from "./verdictColors";

export interface ArticleGroup {
  articleTitle: string;
  articleUrl: string | null;
  records: VerificationRecord[];
}

const SEVERITY_ORDER: Exclude<VerdictLabel, "판단불가">[] = ["불일치", "애매", "일치"];

// 기사 하나(여러 claim)를 대표하는 "가장 심각한" 판정 하나로 요약한다 — 목록 정렬/필터/
// 강조색에 공통으로 쓰는 기준 (불일치 > 애매(검토 필요) > 일치 순).
export function dominantVerdict(group: ArticleGroup): "불일치" | "애매" | "일치" {
  const labels = new Set(group.records.map((r) => verdictCountLabel(r.verification_result)));
  return SEVERITY_ORDER.find((label) => labels.has(label)) ?? "일치";
}

// 대표 판정과 같은 카테고리에 속한 claim 개수 — "이 기사에서 문제되는 게 몇 건인지"를
// 한눈에 보여주기 위한 것 (예: 대표 판정이 불일치면 불일치 건수, 불일치가 없고 애매만
// 있으면 애매 건수).
export function dominantVerdictCount(group: ArticleGroup): number {
  const label = dominantVerdict(group);
  return group.records.filter((r) => verdictCountLabel(r.verification_result) === label).length;
}

export function latestCreatedAt(group: ArticleGroup): string {
  return group.records.reduce(
    (latest, r) => (r.created_at > latest ? r.created_at : latest),
    group.records[0]?.created_at ?? "",
  );
}

// 기사 작성일(articleDates, data_set.csv 기반)을 우선 쓰고, 없으면(export 이전 데이터 등)
// 검증 실행 시각으로 대체 — 목록/상세 화면에서 공통으로 쓰는 날짜 산출 기준.
export function articleDisplayDate(group: ArticleGroup, articleDates: Record<string, string>): string {
  return articleDates[group.articleTitle] ?? latestCreatedAt(group);
}

export function formatDate(iso: string): string {
  return iso ? iso.slice(0, 10).replaceAll("-", ".") : "—";
}

// 같은 기사에서 나온 여러 수치 주장(레코드)이 갤러리에서 뒤섞이지 않도록
// article_title 기준으로 묶는다. 원본 배열의 첫 등장 순서를 그대로 유지한다.
export function groupByArticle(records: VerificationRecord[]): ArticleGroup[] {
  const groups = new Map<string, ArticleGroup>();

  for (const record of records) {
    const key = record.article_title;
    let group = groups.get(key);
    if (!group) {
      group = { articleTitle: key, articleUrl: record.article_url, records: [] };
      groups.set(key, group);
    }
    group.records.push(record);
  }

  return Array.from(groups.values());
}
