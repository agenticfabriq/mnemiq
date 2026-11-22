import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { host, navigation } from "./test/core";
import { appRootFrom, dashboardRefOf, usePageContext } from "./page";

describe("dashboardRefOf", () => {
  it.each([
    ["/dashboard/12/", "", "12"],
    ["/dashboard/sales-kpis/", "", "sales-kpis"],
    ["/dashboard/12", "", "12"],
    ["/dashboard/list/", "", null],
    ["/dashboard/p/abc/", "", null],
    ["/superset/dashboard/12/", "", null],
    ["/analytics/dashboard/12/", "/analytics", "12"],
    ["/analytics/dashboard/12/", "", null],
    ["/analyticsx/dashboard/12/", "/analytics", null],
  ])("%s under root %j gives %j", (pathname, root, ref) => {
    expect(dashboardRefOf(pathname, root)).toBe(ref);
  });
});

describe("appRootFrom", () => {
  const bootstrap = (root: unknown) => JSON.stringify({ common: { application_root: root } });
  it("keeps a root of Superset's own shape and refuses anything else", () => {
    expect(appRootFrom(bootstrap("/analytics"))).toBe("/analytics");
    // As rc2 reads it (`normalizePathWithFallback`): one trailing slash dropped before the shape is checked.
    expect(appRootFrom(bootstrap("/analytics/"))).toBe("/analytics");
    expect(appRootFrom(bootstrap("/"))).toBe("");
    expect(appRootFrom(bootstrap("/app/.."))).toBe("");
    expect(appRootFrom(bootstrap("javascript:x"))).toBe("");
    expect(appRootFrom(bootstrap(""))).toBe("");
    expect(appRootFrom(bootstrap(7))).toBe("");
    expect(appRootFrom(null)).toBe("");
    expect(appRootFrom("{not json")).toBe("");
  });
});

/** Superset's dashboard API as the page context asks it: a ref answers its id and title, or a status. */
function superset(dashboards: Record<string, { id: number; title: string } | number>) {
  return vi.fn<typeof fetch>(async (input) => {
    const ref = decodeURIComponent(String(input).split("/api/v1/dashboard/")[1] ?? "");
    const found = dashboards[ref];
    if (found === undefined) return new Response("{}", { status: 404 });
    if (typeof found === "number") return new Response("{}", { status: found });
    return new Response(JSON.stringify({ id: found.id, result: { id: found.id, dashboard_title: found.title } }),
      { status: 200, headers: { "content-type": "application/json" } });
  });
}

describe("usePageContext", () => {
  beforeEach(() => {
    host.reset();
    history.replaceState(null, "", "/");
  });
  afterEach(() => vi.useRealTimers());

  it("moving between two dashboards updates the context", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const fetchFn = superset({ "1": { id: 1, title: "One" }, "2": { id: 2, title: "Two" } });
    host.navigate("dashboard");
    history.pushState(null, "", "/dashboard/1/");
    const { result } = renderHook(() => usePageContext({ appRoot: "", fetchFn }));
    await waitFor(() => expect(result.current.dashboard).toEqual({ ref: "1", id: 1, title: "One", readable: true, hasCharts: null }));
    const fired: string[] = [];
    navigation.onDidChangePage((page) => fired.push(page));
    host.navigate("dashboard"); // rc2 derives the same page: no event
    history.pushState(null, "", "/dashboard/2/");
    expect(fired).toEqual([]);
    await act(async () => { vi.advanceTimersByTime(1000); });
    await waitFor(() => expect(result.current.dashboard).toEqual({ ref: "2", id: 2, title: "Two", readable: true, hasCharts: null }));
    expect(result.current.page).toBe("dashboard");
  });

  it("a slug resolves to its id and title", async () => {
    const fetchFn = superset({ "sales-kpis": { id: 12, title: "Sales KPIs" } });
    host.navigate("dashboard");
    history.pushState(null, "", "/analytics/dashboard/sales-kpis/");
    const { result } = renderHook(() => usePageContext({ appRoot: "/analytics", fetchFn }));
    await waitFor(() => expect(result.current.dashboard).toEqual(
      { ref: "sales-kpis", id: 12, title: "Sales KPIs", readable: true, hasCharts: null }));
    const [url, init] = fetchFn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/analytics/api/v1/dashboard/sales-kpis");
    expect(init).toMatchObject({ credentials: "same-origin" });
  });

  it("a dashboard the person cannot read is named by nothing", async () => {
    const fetchFn = superset({ "9": 403, "10": 404 });
    host.navigate("dashboard");
    history.pushState(null, "", "/dashboard/9/");
    const { result, rerender } = renderHook(() => usePageContext({ appRoot: "", fetchFn }));
    await waitFor(() => expect(fetchFn).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(result.current.dashboard).toEqual({ ref: "9", id: null, title: null, readable: false }));
    history.pushState(null, "", "/dashboard/10/");
    window.dispatchEvent(new PopStateEvent("popstate"));
    rerender();
    await waitFor(() => expect(result.current.dashboard).toEqual({ ref: "10", id: null, title: null, readable: false }));
  });

  it("off a dashboard there is none, and Superset is not asked", async () => {
    const fetchFn = superset({});
    history.pushState(null, "", "/chart/list/");
    host.navigate("chart_list");
    const { result } = renderHook(() => usePageContext({ appRoot: "", fetchFn }));
    expect(result.current).toEqual({ page: "chart_list", dashboard: null });
    expect(fetchFn).not.toHaveBeenCalled();
  });
});
