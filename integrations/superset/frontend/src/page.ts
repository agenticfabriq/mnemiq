// Which page, and which dashboard, is on screen. rc2's `navigation.onDidChangePage` fires only when the derived page
// changes (`notifyLocationChanged` in superset-frontend/src/core/navigation), so moving from one dashboard to another
// fires nothing: the pathname is also read on `popstate` and once a second while the panel is mounted.
import { useEffect, useState } from "react";
import { navigation } from "@apache-superset/core";

export type Page = ReturnType<typeof navigation.getPage>;

export type DashboardContext = { ref: string; id: number | null; title: string | null; readable: boolean };

export type PageContext = { page: Page; dashboard: DashboardContext | null };

// rc2's own rule for an application root it will prefix paths with (`APP_ROOT_PATH_RE` in
// superset-frontend/src/utils/getBootstrapData.ts): slash-led segments, no `.` or `..` segment.
const APP_ROOT_PATH_RE = /^(?:\/[\w~%-]+(?:\.[\w~%-]+)*)*$/;

/** The application root Superset serves under, read from `#app`'s `data-bootstrap` as rc2 reads it, or "" when there
 * is none or it does not have the shape rc2 keeps. */
export function appRootFrom(bootstrap: string | null): string {
  if (!bootstrap) return "";
  try {
    const root = (JSON.parse(bootstrap) as { common?: { application_root?: unknown } })?.common?.application_root;
    return typeof root === "string" && APP_ROOT_PATH_RE.test(root) ? root : "";
  } catch {
    return "";
  }
}

/** The application root of this page. */
export const pageAppRoot = (): string => appRootFrom(document.getElementById("app")?.getAttribute("data-bootstrap") ?? null);

/** The dashboard a pathname shows, by id or slug: `<root>/dashboard/<ref>/`, the trailing slash optional. Never the
 * list, and never a permalink (`/dashboard/p/<key>/`, which Superset redirects to the id form). */
export function dashboardRefOf(pathname: string, appRoot: string): string | null {
  if (appRoot && !pathname.startsWith(`${appRoot}/`)) return null;
  const found = /^\/dashboard\/([^/]+)\/?$/.exec(pathname.slice(appRoot.length));
  if (!found || found[1] === "list" || found[1] === "p") return null;
  try {
    return decodeURIComponent(found[1]);
  } catch {
    return null;
  }
}

type Options = { appRoot?: string; fetchFn?: typeof fetch };

/** The page on screen and, on a dashboard, which one: its ref, then its id and title as Superset answers them to
 * this person (`GET <root>/api/v1/dashboard/<ref>`, same origin). A dashboard Superset will not show them (403, 404)
 * is not readable, and nothing of it is named. */
export function usePageContext({ appRoot, fetchFn }: Options = {}): PageContext {
  const root = appRoot ?? pageAppRoot();
  const [page, setPage] = useState<Page>(() => navigation.getPage());
  const [pathname, setPathname] = useState(() => window.location.pathname);
  const [resolved, setResolved] = useState<DashboardContext | null>(null);

  useEffect(() => {
    const read = () => setPathname(window.location.pathname);
    const changed = navigation.onDidChangePage((next) => {
      setPage(next);
      read();
    });
    window.addEventListener("popstate", read);
    const timer = window.setInterval(read, 1000);
    return () => {
      changed.dispose();
      window.removeEventListener("popstate", read);
      window.clearInterval(timer);
    };
  }, []);

  const ref = dashboardRefOf(pathname, root);

  useEffect(() => {
    if (ref === null) return;
    let current = true;
    const ask = fetchFn ?? ((...args: Parameters<typeof fetch>) => fetch(...args));
    ask(`${root}/api/v1/dashboard/${encodeURIComponent(ref)}`, { credentials: "same-origin", redirect: "manual",
      headers: { Accept: "application/json" } })
      .then(async (response) => {
        if (!response.ok) return { ref, id: null, title: null, readable: false };
        const result = ((await response.json()) as { result?: { id?: unknown; dashboard_title?: unknown } })?.result;
        return typeof result?.id === "number"
          ? { ref, id: result.id, title: typeof result.dashboard_title === "string" ? result.dashboard_title : null,
              readable: true }
          : { ref, id: null, title: null, readable: false };
      })
      .catch(() => ({ ref, id: null, title: null, readable: false }))
      .then((found) => current && setResolved(found));
    return () => {
      current = false;
    };
  }, [ref, root, fetchFn]);

  if (ref === null) return { page, dashboard: null };
  // Until Superset has answered for this ref, it is named by nothing and offers nothing.
  return { page, dashboard: resolved?.ref === ref ? resolved : { ref, id: null, title: null, readable: false } };
}
