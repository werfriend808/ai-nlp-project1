import { useEffect, useMemo, useState } from "react";
import { Header } from "./components/layout/Header";
import { FunnelChart } from "./components/funnel/FunnelChart";
import { ArticleList } from "./components/articles/ArticleList";
import { ArticleDetail } from "./components/articles/ArticleDetail";
import { MOCK_VERIFICATIONS } from "./data/mockVerifications";
import { MOCK_ARTICLE_TEXTS } from "./data/mockArticleTexts";
import { groupByArticle } from "./lib/articles";
import type { VerificationRecord } from "./types/verification";

const EXPORT_JSON_PATH = "/data/verifications.json";
const ARTICLES_JSON_PATH = "/data/articles.json";

async function fetchJson<T>(path: string): Promise<T | null> {
  // Vite dev 서버는 없는 경로도 SPA 폴백으로 index.html(200 text/html)을 돌려주기 때문에
  // res.ok만으로는 "파일이 진짜 있는지" 못 가려서 content-type도 함께 확인한다.
  try {
    const res = await fetch(path);
    if (!res.ok || !res.headers.get("content-type")?.includes("json")) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

function App() {
  const [records, setRecords] = useState<VerificationRecord[]>(MOCK_VERIFICATIONS);
  const [articleTexts, setArticleTexts] = useState<Record<string, string>>(MOCK_ARTICLE_TEXTS);
  const [usingMockData, setUsingMockData] = useState(true);
  const [selectedArticle, setSelectedArticle] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    // db/export_json.py가 만든 실데이터가 frontend/public/data/에 있으면 그걸 쓰고,
    // 아직 export 전이거나 배치를 한 번도 안 돌린 상태라면 MOCK_*로 남아있는다 — 빈 화면
    // 대신 항상 뭔가는 보이게 하기 위함.
    Promise.all([
      fetchJson<VerificationRecord[]>(EXPORT_JSON_PATH),
      fetchJson<Record<string, string>>(ARTICLES_JSON_PATH),
    ]).then(([verifications, articles]) => {
      if (cancelled || !verifications || verifications.length === 0) return;
      setRecords(verifications);
      setArticleTexts(articles ?? {});
      setUsingMockData(false);
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const articleGroups = useMemo(() => groupByArticle(records), [records]);
  const selectedGroup = articleGroups.find((g) => g.articleTitle === selectedArticle);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <Header />
      <main className="mx-auto flex max-w-4xl flex-col gap-6 px-6 py-6">
        {usingMockData && (
          <p className="rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:bg-amber-900/30 dark:text-amber-300">
            {EXPORT_JSON_PATH}를 찾지 못해 목업 데이터로 표시 중입니다. 배치 실행 후
            `python -m db.export_json --out frontend/public/data/verifications.json`로
            export하면 자동으로 실데이터가 표시됩니다.
          </p>
        )}

        {selectedGroup ? (
          <ArticleDetail
            group={selectedGroup}
            articleText={articleTexts[selectedGroup.articleTitle]}
            onBack={() => setSelectedArticle(null)}
          />
        ) : (
          <>
            <FunnelChart records={records} />
            <ArticleList groups={articleGroups} onSelect={setSelectedArticle} />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
