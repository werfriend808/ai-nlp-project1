import type { VerificationRecord } from "../types/verification";

export interface ArticleGroup {
  articleTitle: string;
  articleUrl: string | null;
  records: VerificationRecord[];
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
