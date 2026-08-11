import { useEffect, useMemo, useState } from "react";
import { Header } from "./components/layout/Header";
import { SummaryCard } from "./components/dashboard/SummaryCard";
import { ArticleListPanel } from "./components/articles/ArticleListPanel";
import { ArticleDetail } from "./components/articles/ArticleDetail";
import { MOCK_VERIFICATIONS } from "./data/mockVerifications";
import { MOCK_ARTICLE_TEXTS } from "./data/mockArticleTexts";
import { MOCK_ARTICLE_DATES } from "./data/mockArticleDates";
import { groupByArticle, latestCreatedAt } from "./lib/articles";
import { verdictCountLabel } from "./lib/verdictColors";
import type { VerificationRecord } from "./types/verification";

const DATE_RANGE_OPTIONS = ["전체", 7, 30, 60, 100] as const;
type DateRangeOption = (typeof DATE_RANGE_OPTIONS)[number];

const EXPORT_JSON_PATH = "/data/verifications.json";
const ARTICLES_JSON_PATH = "/data/articles.json";
const ARTICLE_DATES_JSON_PATH = "/data/articleDates.json";
const TABLE_ORG_IDS_JSON_PATH = "/data/tableOrgIds.json";

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
  const [articleDates, setArticleDates] = useState<Record<string, string>>(MOCK_ARTICLE_DATES);
  const [tableOrgIds, setTableOrgIds] = useState<Record<string, string>>({});
  const [usingMockData, setUsingMockData] = useState(true);
  const [selectedArticle, setSelectedArticle] = useState<string | null>(null);
  const [reviewFilterActive, setReviewFilterActive] = useState(false);
  // 데모 데이터가 전부 작년(2025년) 기사라 "최근 N일"을 기본값으로 두면 오늘 날짜 기준
  // 며칠이 지나든 다 걸러져 빈 화면이 뜬다 — 기본은 "전체"로 시작해서 항상 뭔가는 보이게
  // 하고, 필요할 때만 사용자가 직접 기간을 좁히게 한다.
  const [dateRange, setDateRange] = useState<DateRangeOption>("전체");

  useEffect(() => {
    let cancelled = false;

    // db/export_json.py가 만든 실데이터가 frontend/public/data/에 있으면 그걸 쓰고,
    // 아직 export 전이거나 배치를 한 번도 안 돌린 상태라면 MOCK_*로 남아있는다 — 빈 화면
    // 대신 항상 뭔가는 보이게 하기 위함.
    Promise.all([
      fetchJson<VerificationRecord[]>(EXPORT_JSON_PATH),
      fetchJson<Record<string, string>>(ARTICLES_JSON_PATH),
      fetchJson<Record<string, string>>(ARTICLE_DATES_JSON_PATH),
      fetchJson<Record<string, string>>(TABLE_ORG_IDS_JSON_PATH),
    ]).then(([verifications, articles, dates, orgIds]) => {
      if (cancelled || !verifications || verifications.length === 0) return;
      setRecords(verifications);
      setArticleTexts(articles ?? {});
      setArticleDates(dates ?? {});
      setTableOrgIds(orgIds ?? {});
      setUsingMockData(false);
    });

    return () => {
      cancelled = true;
    };
  }, []);

  // 목록↔상세 전환을 브라우저 히스토리(History API)와 연동한다. 이게 없으면 뒤로가기가
  // 앱 상태를 모르고 그냥 브라우저를 완전히 벗어나버린다 — 기사 클릭 시 pushState로
  // 히스토리 항목을 쌓고, popstate(뒤로/앞으로가기)가 발생하면 URL의 article 쿼리를
  // 읽어 그 상태로 복원한다. 새로고침/북마크로 상세 페이지에 바로 들어오는 것도 덤으로 된다.
  useEffect(() => {
    const syncFromUrl = () => {
      const params = new URLSearchParams(window.location.search);
      setSelectedArticle(params.get("article"));
    };
    syncFromUrl();
    window.addEventListener("popstate", syncFromUrl);
    return () => window.removeEventListener("popstate", syncFromUrl);
  }, []);

  const allArticleGroups = useMemo(() => groupByArticle(records), [records]);

  // 날짜 범위 필터: 기사 작성일(articleDates, data_set.csv 기반) 기준 "지금부터 N일 이내"인
  // 기사만 남긴다. 작성일을 못 구한 기사는 검증 실행 시각(created_at)으로 대신 판단한다 —
  // 그래야 작성일 export가 없던 예전 데이터(articleDates.json 미존재)에서도 필터가 안 죽는다.
  const visibleGroups = useMemo(() => {
    const cutoff = dateRange === "전체" ? null : Date.now() - dateRange * 24 * 60 * 60 * 1000;
    const inRange =
      cutoff === null
        ? allArticleGroups
        : allArticleGroups.filter((g) => {
            const dateStr = articleDates[g.articleTitle] ?? latestCreatedAt(g);
            const t = Date.parse(dateStr);
            return Number.isNaN(t) || t >= cutoff;
          });
    // SummaryCard의 "상세 보기" 버튼: 검토가 필요한 주장(verdictCountLabel 기준 "애매" —
    // 표매칭 신뢰도 낮음 뿐 아니라 "판단불가"도 포함)이 하나라도 있는 기사만 걸러서
    // 보여준다 — 그냥 장식용 버튼이 아니라 실제로 목록을 필터링한다.
    return reviewFilterActive
      ? inRange.filter((g) => g.records.some((r) => verdictCountLabel(r.verification_result) === "애매"))
      : inRange;
  }, [allArticleGroups, articleDates, dateRange, reviewFilterActive]);

  const recordsInRange = useMemo(() => visibleGroups.flatMap((g) => g.records), [visibleGroups]);

  const selectedGroup = allArticleGroups.find((g) => g.articleTitle === selectedArticle);

  const handleSelectArticle = (articleTitle: string) => {
    const url = `?article=${encodeURIComponent(articleTitle)}`;
    window.history.pushState({ articleTitle }, "", url);
    setSelectedArticle(articleTitle);
  };

  // 상세 화면의 "목록으로"는 상태만 지우는 게 아니라 실제로 history.back()을 호출해야
  // 한다 — 그래야 이 화면에서 브라우저 뒤로가기를 눌렀을 때도 같은 자리(목록)로 오게
  // 되어, 인앱 버튼과 브라우저 뒤로가기가 서로 어긋나지 않는다.
  const handleBack = () => window.history.back();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
      <Header wide={!!selectedGroup}>
        {!selectedGroup && (
          <div className="flex gap-1 rounded-lg border border-gray-200 bg-white p-1 dark:border-gray-700 dark:bg-gray-900">
            {DATE_RANGE_OPTIONS.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => setDateRange(option)}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition ${
                  dateRange === option
                    ? "bg-indigo-600 text-white"
                    : "text-gray-500 hover:text-gray-700 dark:text-gray-400"
                }`}
              >
                {option === "전체" ? "전체" : `최근 ${option}일`}
              </button>
            ))}
          </div>
        )}
      </Header>
      <main
        className={`mx-auto flex flex-col gap-6 px-6 py-6 ${selectedGroup ? "max-w-6xl" : "max-w-4xl"}`}
      >
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
            tableOrgIds={tableOrgIds}
            onBack={handleBack}
          />
        ) : (
          <>
            <SummaryCard
              records={recordsInRange}
              reviewFilterActive={reviewFilterActive}
              onToggleReviewFilter={() => setReviewFilterActive((v) => !v)}
            />
            <ArticleListPanel
              groups={visibleGroups}
              articleDates={articleDates}
              onSelect={handleSelectArticle}
            />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
