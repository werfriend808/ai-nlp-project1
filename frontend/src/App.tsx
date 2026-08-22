import { useCallback, useEffect, useMemo, useState } from "react";
import { Header } from "./components/layout/Header";
import { SummaryCard } from "./components/dashboard/SummaryCard";
import { ArticleListPanel } from "./components/articles/ArticleListPanel";
import { ArticleDetail } from "./components/articles/ArticleDetail";
import { VerifyNewArticleButton } from "./components/verify/VerifyNewArticleButton";
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

// 2026-08-21: 실시간 검증(VerifyNewArticleButton)이 끝난 직후에는 로컬 정적 파일이 아니라
// agent/api/server.py(AWS 서버)가 방금 갱신한 JSON을 직접 읽어야 한다 — 로컬 프론트와
// AWS 서버는 서로 다른 컴퓨터라 서버가 자기 디스크에 쓴 파일이 로컬로 자동으로 오지
// 않기 때문(server.py가 /data를 정적 서빙하도록 추가해둠).
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

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

  // db/export_json.py가 만든 실데이터가 frontend/public/data/에 있으면 그걸 쓰고, 아직
  // export 전이거나 배치를 한 번도 안 돌린 상태라면 MOCK_*로 남아있는다 — 빈 화면 대신
  // 항상 뭔가는 보이게 하기 위함. 2026-08-21: "새로운 뉴스 기사 검증하기"가 끝난 뒤에도
  // 같은 로직으로 다시 불러와야 해서 함수로 뽑았다(마운트 시 1회 + 검증 완료 시 재호출).
  //
  // fetch 시 매번 새 URL(캐시 버스팅 쿼리)을 붙인다 — 브라우저가 이전 GET 응답을 캐싱해서
  // 검증 완료 후 재조회해도 갱신 전 데이터가 그대로 보이는 문제를 막기 위함.
  //
  // baseUrl: 마운트 시 최초 로드는 로컬 정적 파일(""), 실시간 검증 완료 후 재호출은
  // API 서버(AWS)를 baseUrl로 넘겨서 방금 그 서버가 갱신한 최신 데이터를 직접 받는다.
  const loadData = useCallback((baseUrl = "") => {
    const bust = `?t=${Date.now()}`;
    return Promise.all([
      fetchJson<VerificationRecord[]>(baseUrl + EXPORT_JSON_PATH + bust),
      fetchJson<Record<string, string>>(baseUrl + ARTICLES_JSON_PATH + bust),
      fetchJson<Record<string, string>>(baseUrl + ARTICLE_DATES_JSON_PATH + bust),
      fetchJson<Record<string, string>>(baseUrl + TABLE_ORG_IDS_JSON_PATH + bust),
    ]).then(([verifications, articles, dates, orgIds]) => {
      if (!verifications || verifications.length === 0) return;
      setRecords(verifications);
      setArticleTexts(articles ?? {});
      setArticleDates(dates ?? {});
      setTableOrgIds(orgIds ?? {});
      setUsingMockData(false);
    });
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleVerificationDone = useCallback(() => loadData(API_BASE_URL), [loadData]);

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
                className={`rounded-md px-3 py-1.5 text-xs font-semibold transition ${
                  dateRange === option
                    ? "bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900"
                    : "text-gray-500 hover:bg-gray-50 hover:text-gray-700 dark:text-gray-400 dark:hover:bg-gray-800"
                }`}
              >
                {option === "전체" ? "전체" : `최근 ${option}일`}
              </button>
            ))}
          </div>
        )}
      </Header>
      <main
        // pb-28(하단 여백)은 화면 하단 중앙에 고정된 "새로운 뉴스 기사 검증하기" 버튼
        // (VerifyNewArticleButton, fixed bottom-6 + 버튼 자체 높이)이 스크롤 맨 아래에서
        // 검증 기사 목록 마지막 줄과 겹치는 문제 수정(2026-08-22 실측) — 버튼이 차지하는
        // 공간만큼 본문 하단에 여유를 둬서 항상 그 아래로 스크롤할 수 있게 한다.
        className={`mx-auto flex flex-col gap-6 px-6 pb-28 pt-6 ${selectedGroup ? "max-w-7xl" : "max-w-6xl"}`}
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
            articleDates={articleDates}
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
      <VerifyNewArticleButton onVerificationDone={handleVerificationDone} />
    </div>
  );
}

export default App;
