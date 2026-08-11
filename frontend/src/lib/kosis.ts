// KOSIS 표 상세 페이지로 바로 연결되는 딥링크. orgId+tblId 둘 다 있어야 정확한 표
// 페이지로 가고(실제 KOSIS URL 형식 확인함), orgId를 모르면 통합검색으로 대신 연결한다
// (검색 결과 목록이라 표 하나로 바로 안 가지만, 완전히 틀린 링크보다는 낫다).
export function kosisTableUrl(tblId: string, orgId: string | undefined, tableName: string): string {
  if (!orgId) {
    return `https://kosis.kr/search/search.do?query=${encodeURIComponent(tableName)}`;
  }
  const params = new URLSearchParams({
    orgId,
    tblId,
    vw_cd: "MT_ZTITLE",
    list_id: "",
    seqNo: "",
    lang_mode: "ko",
    language: "kor",
    obj_var_id: "",
    itm_id: "",
    conn_path: "MT_ZTITLE",
  });
  return `https://kosis.kr/statHtml/statHtml.do?${params.toString()}`;
}
