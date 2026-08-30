import { useCallback, useEffect, useMemo, useState } from "react";
import { Header } from "./components/layout/Header";
import { ArticleDetail } from "./components/articles/ArticleDetail";
import { RecentArticlesList } from "./components/articles/RecentArticlesList";
import { RecentArticlesSummaryCard } from "./components/articles/RecentArticlesSummaryCard";
import { VerifyArticleForm } from "./components/verify/VerifyArticleForm";
import { VerifyAnotherArticleButton } from "./components/verify/VerifyAnotherArticleButton";
import { useVerifyJobs } from "./lib/useVerifyJobs";
import { MOCK_VERIFICATIONS } from "./data/mockVerifications";
import { MOCK_ARTICLE_TEXTS } from "./data/mockArticleTexts";
import { MOCK_ARTICLE_DATES } from "./data/mockArticleDates";
import { groupByArticle } from "./lib/articles";
import type { VerificationRecord } from "./types/verification";

// 2026-08-26(5): 프론트 구조 개편 — 예전엔 이미 검증된 기사들을 목록/통계로 브라우징하는
// 화면이 메인이었는데, 이제는 "기사 URL을 입력하면 그 기사 결과를 보여주는" 단일 조회
// 도구로 바꾼다(사용자 요청). SummaryCard/ArticleListPanel/ArticleSearchBar와 그걸 위한
// 날짜 필터·검토필요 필터·검색 상태는 전부 삭제 — 더 이상 쓰이지 않는다.
const EXPORT_JSON_PATH = "/data/verifications.json";
const ARTICLES_JSON_PATH = "/data/articles.json";
const ARTICLE_DATES_JSON_PATH = "/data/articleDates.json";
const TABLE_ORG_IDS_JSON_PATH = "/data/tableOrgIds.json";

// 2026-08-21: 실시간 검증이 끝난 직후에는 로컬 정적 파일이 아니라 agent/api/server.py
// (AWS 서버)가 방금 갱신한 JSON을 직접 읽어야 한다 — 로컬 프론트와 AWS 서버는 서로 다른
// 컴퓨터라 서버가 자기 디스크에 쓴 파일이 로컬로 자동으로 오지 않기 때문(server.py가
// /data를 정적 서빙하도록 추가해둠).
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
  // 2026-08-28(2): "최근 검증한 기사" 전체 목록을 보여주는 전용 화면 — 입력 화면에는
  // 압축 요약 카드만 두고, 이 상태가 true일 때만 RecentArticlesList를 전체로 펼친다.
  const [showHistory, setShowHistory] = useState(false);

  // db/export_json.py가 만든 실데이터가 frontend/public/data/에 있으면 그걸 쓰고, 아직
  // export 전이거나 배치를 한 번도 안 돌린 상태라면 MOCK_*로 남아있는다 — 빈 화면 대신
  // 항상 뭔가는 보이게 하기 위함. 검증이 끝난 뒤에도 같은 로직으로 다시 불러와야 해서
  // 함수로 뽑았다(마운트 시 1회 + 검증 완료 시 재호출).
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

  // 목록↔상세 대신 이제 "입력 화면↔결과 화면" 전환을 브라우저 히스토리(History API)와
  // 연동한다. 기사 결과로 이동할 때 pushState로 히스토리 항목을 쌓고, popstate(뒤로/
  // 앞으로가기)가 발생하면 URL의 article 쿼리를 읽어 그 상태로 복원한다. 새로고침/북마크로
  // 결과 화면에 바로 들어오는 것도 덤으로 된다(검증 링크를 남한테 공유하기도 쉬움).
  useEffect(() => {
    const syncFromUrl = () => {
      const params = new URLSearchParams(window.location.search);
      setSelectedArticle(params.get("article"));
      setShowHistory(params.get("view") === "history");
    };
    syncFromUrl();
    window.addEventListener("popstate", syncFromUrl);
    return () => window.removeEventListener("popstate", syncFromUrl);
  }, []);

  const allArticleGroups = useMemo(() => groupByArticle(records), [records]);
  const selectedGroup = allArticleGroups.find((g) => g.articleTitle === selectedArticle);

  const handleSelectArticle = useCallback((articleTitle: string) => {
    const url = `?article=${encodeURIComponent(articleTitle)}`;
    window.history.pushState({ articleTitle }, "", url);
    setSelectedArticle(articleTitle);
  }, []);

  // URL 검증이 끝난 뒤 그 기사 결과로 이동 — 먼저 서버가 방금 갱신한 최신 데이터를
  // 다시 불러온 다음(그래야 방금 끝난 기사가 records 안에 들어있음), 그 기사로 이동한다.
  // 순서를 반대로 하면(이동 먼저) selectedGroup을 못 찾아서 빈 화면이 뜬다.
  const handleArticleVerified = useCallback(
    (articleTitle: string) => {
      loadData(API_BASE_URL).then(() => handleSelectArticle(articleTitle));
    },
    [loadData, handleSelectArticle],
  );

  // 2026-08-26(5): useVerifyJobs를 App 최상위에서 한 번만 들고 있는다 — 메인 화면(인라인
  // 입력 폼)과 결과 화면("다른 기사 검증하기" 모달)이 같은 job 상태를 공유해야, 예를 들어
  // 메인 화면에서 URL 3개를 한꺼번에 넣고 그중 하나 결과를 먼저 보러 이동해도 나머지
  // 2개가 백그라운드에서 계속 진행되다가, 다시 메인으로 돌아오거나 모달을 열었을 때
  // 그 진행 상황이 그대로 보인다.
  const verifyJobs = useVerifyJobs({
    onJobDone: (articleTitle, isSingle) => {
      if (isSingle && articleTitle) handleArticleVerified(articleTitle);
    },
  });

  // 상세 화면의 "목록으로"는 상태만 지우는 게 아니라 실제로 history.back()을 호출해야
  // 한다 — 그래야 이 화면에서 브라우저 뒤로가기를 눌렀을 때도 같은 자리(입력 화면)로
  // 오게 되어, 인앱 버튼과 브라우저 뒤로가기가 서로 어긋나지 않는다.
  const handleBack = () => window.history.back();

  const goHome = () => {
    if (selectedArticle || showHistory) window.history.pushState({}, "", window.location.pathname);
    setSelectedArticle(null);
    setShowHistory(false);
  };

  const openHistory = () => {
    window.history.pushState({ view: "history" }, "", "?view=history");
    setShowHistory(true);
  };

  return (
    <div className="min-h-screen bg-stone-200 dark:bg-stone-950">
      <Header wide={!!selectedGroup} onLogoClick={goHome} />
      <main
        // pb-28(하단 여백)은 결과 화면에서 화면 하단 중앙에 고정된 "다른 기사 검증하기"
        // 버튼이 스크롤 맨 아래에서 본문과 겹치는 걸 막기 위함 — 입력 화면은 그 버튼이
        // 없으니 필요 없다.
        className={`mx-auto flex flex-col gap-6 px-6 pt-6 ${
          selectedGroup ? "max-w-7xl pb-28" : showHistory ? "max-w-2xl pb-10" : "max-w-xl pb-10"
        }`}
      >
        {usingMockData && (
          <p className="rounded-md bg-caution-50 px-3 py-2 text-xs text-caution-800 dark:bg-caution-900/30 dark:text-caution-300">
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
        ) : showHistory ? (
          // 2026-08-28(2): "최근 검증한 기사" 전용 화면 — 압축 요약 카드를 눌러야 여기로
          // 들어온다(입력 화면엔 목록을 안 펼쳐서 지저분해 보이지 않게). handleSelectArticle을
          // 바로 부른다 — 이미 로드된 records 안에서 찾는 것뿐이라 재조회가 필요 없다.
          <>
            <button
              type="button"
              onClick={handleBack}
              className="inline-flex w-fit items-center gap-1 rounded-full border border-stone-200 bg-white px-3 py-1 text-xs font-medium text-stone-600 hover:border-stone-300 hover:bg-stone-100 hover:text-stone-700 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-300 dark:hover:bg-stone-800/50"
            >
              ← 처음으로
            </button>
            <RecentArticlesList
              groups={allArticleGroups}
              articleDates={articleDates}
              onSelect={handleSelectArticle}
            />
          </>
        ) : (
          <>
            <div className="rounded-2xl border border-stone-200 bg-white p-8 shadow-sm dark:border-stone-700 dark:bg-stone-900">
              <h2 className="text-lg font-bold text-stone-900 dark:text-stone-100">
                기사 URL을 입력하세요
              </h2>
              <p className="mt-1 text-sm text-stone-500 dark:text-stone-400">
                뉴스 기사 링크를 넣으면 기사 속 수치 주장을 KOSIS 공식 통계와 대조해서 보여드려요.
              </p>
              <div className="mt-5">
                <VerifyArticleForm
                  urlInputs={verifyJobs.urlInputs}
                  jobs={verifyJobs.jobs}
                  isSubmitting={verifyJobs.isSubmitting}
                  isAllTerminal={verifyJobs.isAllTerminal}
                  doneCount={verifyJobs.doneCount}
                  failedCount={verifyJobs.failedCount}
                  onAddInputs={verifyJobs.handleAddInputs}
                  onRemoveInput={verifyJobs.handleRemoveInput}
                  onInputChange={verifyJobs.handleInputChange}
                  onSubmit={verifyJobs.handleSubmit}
                  onReset={verifyJobs.reset}
                  onViewArticle={handleArticleVerified}
                  autoFocus
                />
              </div>
            </div>

            {/* 2026-08-28(2): 전체 목록 대신 압축 요약 카드만 — 누르면 위 showHistory
                화면으로 이동. 목록이 비어있으면(count=0) 카드 자체를 안 보여줌. */}
            <RecentArticlesSummaryCard count={allArticleGroups.length} onClick={openHistory} />
          </>
        )}
      </main>
      {selectedGroup && (
        <VerifyAnotherArticleButton
          urlInputs={verifyJobs.urlInputs}
          jobs={verifyJobs.jobs}
          isSubmitting={verifyJobs.isSubmitting}
          isAllTerminal={verifyJobs.isAllTerminal}
          doneCount={verifyJobs.doneCount}
          failedCount={verifyJobs.failedCount}
          hasAnyBusy={verifyJobs.hasAnyBusy}
          onAddInputs={verifyJobs.handleAddInputs}
          onRemoveInput={verifyJobs.handleRemoveInput}
          onInputChange={verifyJobs.handleInputChange}
          onSubmit={verifyJobs.handleSubmit}
          onReset={verifyJobs.reset}
          onArticleVerified={handleArticleVerified}
        />
      )}
    </div>
  );
}

export default App;
