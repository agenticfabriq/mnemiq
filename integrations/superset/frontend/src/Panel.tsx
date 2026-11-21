import React, { useEffect, useMemo, useState, useSyncExternalStore, type CSSProperties } from "react";
import { Button, Spin, theme } from "antd";
import { authentication, chat } from "@apache-superset/core";
import { PanelApi } from "./api";
import { askInPanel } from "./ask";
import { driver } from "./build";
import { changeInPanel } from "./change";
import { isRunning } from "./shared";
import { dashboardRefOf, pageAppRoot, resolveDashboard, usePageContext } from "./page";
import { probe, type Readiness } from "./readiness";
import { getSnapshot, subscribe } from "./store";
import { BuildCard } from "./views/BuildCard";
import { ChangeCard } from "./views/ChangeCard";
import { Composer } from "./views/Composer";
import { Conversation } from "./views/Conversation";
import { Header } from "./views/Header";
import { Measures } from "./views/Measures";
import { UnfinishedBuilds } from "./views/Unfinished";

// 640 px tall, at most 80% of the window: `min(640px, 80vh)`, spelt so jsdom, whose CSS parser drops `min()`, keeps it.
const FLOATING: CSSProperties = { width: 420, height: 640, maxHeight: "80vh" };
const DOCKED: CSSProperties = { width: "100%", height: "100%" };

/** The panel Superset mounts, floating over the page or docked beside it. Its styles are inline, under one root
 * element, with antd's colours, so it brings no stylesheet of its own. */
export function Panel() {
  const [mode, setMode] = useState(chat.getDisplayMode());
  useEffect(() => {
    const changed = chat.onDidChangeDisplayMode((next) => setMode(next));
    setMode(chat.getDisplayMode());
    return () => changed.dispose();
  }, []);
  const { token } = theme.useToken();
  const root: CSSProperties = {
    ...(mode === "floating" ? FLOATING : DOCKED),
    display: "flex",
    flexDirection: "column",
    boxSizing: "border-box",
    background: token.colorBgContainer,
    color: token.colorText,
    border: mode === "floating" ? `1px solid ${token.colorBorderSecondary}` : undefined,
    borderRadius: mode === "floating" ? token.borderRadiusLG : undefined,
    fontFamily: token.fontFamily,
    fontSize: token.fontSize,
    overflow: "hidden",
  };
  return (
    <section data-mnemiq-panel={mode} style={root} aria-label="mnemiq">
      <Header mode={mode} />
      <div style={{ flex: 1, overflow: "auto", padding: token.padding }}>
        <Body />
      </div>
    </section>
  );
}

/** The panel's API client for this page: the forwarder under the page's application root, with Superset's CSRF
 * token on a POST. */
function usePanelApi(): PanelApi {
  return useMemo(() => new PanelApi(pageAppRoot(), () => authentication.getCSRFToken()), []);
}

/** What the person may do here, asked on each opening; until this opening's answer says they are ready, nothing else
 * is shown or asked (an earlier opening's answer may be out of date: their roles can change while it is closed). */
function Body() {
  const api = usePanelApi();
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  useEffect(() => {
    const asking = new AbortController();
    probe(api, asking.signal)
      .then((found) => setReadiness(found))
      .catch(() => {});
    return () => asking.abort();
  }, [api]);
  if (readiness === null) return <Spin size="small" />;
  if (readiness.kind !== "ready") return <p role="alert">{readiness.sentence}</p>;
  return <Ready api={api} starters={readiness.starters} />;
}

/** The panel for a person mnemiq answers: the dashboard on screen, when they may read it; their unfinished builds;
 * the opening questions until the first is asked; the conversation and the build; and the composer. */
function Ready({ api, starters }: { api: PanelApi; starters: string[] }) {
  const { dashboard } = usePageContext();
  const { turns, runningBuild, planning, change: changing } = useSyncExternalStore(subscribe, getSnapshot);
  const { token } = theme.useToken();
  useEffect(() => driver.attach(api), [api]);
  const onScreen = dashboard?.readable && dashboard.id !== null ? dashboard : null;
  const asking = isRunning(turns);
  const ask = (question: string) => askInPanel(api, question);
  // The target is worked out when the goal is sent, from the path on screen then (Superset moves between dashboards
  // with no page event, so the page context may be a second behind): that dashboard, resolved as the person; a new one
  // when ticked; and, off a dashboard or on one they cannot read, neither, so mnemiq may offer one that covers it.
  const build = (goal: string, fresh: boolean) => {
    if (driver.busy()) return false;
    const root = pageAppRoot();
    void driver.start(goal, fresh ? { fresh: true } : async () => {
      const ref = dashboardRefOf(window.location.pathname, root);
      if (ref === null) return {};
      const found = await resolveDashboard(root, ref);
      return found.readable && found.id !== null ? { dashboardId: found.id } : {};
    });
    return true;
  };
  // A change goes to the dashboard on screen when it is sent, and only to one the person may read.
  const change = (request: string) => changeInPanel(api, request, async () => {
    const root = pageAppRoot();
    const ref = dashboardRefOf(window.location.pathname, root);
    if (ref === null) return null;
    const found = await resolveDashboard(root, ref);
    return found.readable ? found.id : null;
  });
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: token.padding }}>
      {onScreen ? (
        <section aria-label="This dashboard" style={{ color: token.colorTextSecondary }}>
          On <strong style={{ color: token.colorText }}>{onScreen.title ?? "this dashboard"}</strong>
        </section>
      ) : null}
      {onScreen?.id != null ? <Measures key={onScreen.id} api={api} dashboardId={onScreen.id} /> : null}
      <UnfinishedBuilds api={api} dashboardId={onScreen?.id ?? null} />
      {turns.length === 0 && starters.length > 0 ? (
        <section aria-label="Starters" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
          {starters.map((starter) => (
            <Button key={starter} disabled={asking} onClick={() => ask(starter)}
              style={{ textAlign: "left", whiteSpace: "normal", height: "auto" }}>{starter}</Button>
          ))}
        </section>
      ) : null}
      <Conversation turns={turns} />
      <BuildCard onScreenId={onScreen?.id ?? null} />
      <ChangeCard api={api} />
      <Composer busy={{ ask: asking, build: planning || runningBuild !== null, change: changing?.sending ?? false }}
        onDashboard={onScreen ? onScreen.title ?? "this dashboard" : null} onAsk={ask} onBuild={build} onChange={change} />
    </div>
  );
}
