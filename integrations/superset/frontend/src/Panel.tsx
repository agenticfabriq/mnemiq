import React, { useEffect, useMemo, useState, useSyncExternalStore, type CSSProperties } from "react";
import { Button, Spin, theme } from "antd";
import { authentication, chat } from "@apache-superset/core";
import { PanelApi } from "./api";
import { pageAppRoot, usePageContext } from "./page";
import { probe } from "./readiness";
import { getSnapshot, subscribe, update } from "./store";
import { Header } from "./views/Header";

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

/** What the person may do here, asked once per opening; until mnemiq says they are ready, nothing else is asked. */
function Body() {
  const api = usePanelApi();
  const { readiness } = useSyncExternalStore(subscribe, getSnapshot);
  useEffect(() => {
    const asking = new AbortController();
    probe(api, asking.signal)
      .then((found) => update((state) => ({ ...state, readiness: found })))
      .catch(() => {});
    return () => asking.abort();
  }, [api]);
  if (readiness === null) return <Spin size="small" />;
  if (readiness.kind !== "ready") return <p role="alert">{readiness.sentence}</p>;
  return <Ready starters={readiness.starters} />;
}

/** The panel for a person mnemiq answers: the dashboard on screen, when they may read it, and the opening questions. */
function Ready({ starters }: { starters: string[] }) {
  const { dashboard } = usePageContext();
  const { token } = theme.useToken();
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: token.padding }}>
      {dashboard?.readable && dashboard.id !== null ? (
        <section aria-label="This dashboard" style={{ color: token.colorTextSecondary }}>
          On <strong style={{ color: token.colorText }}>{dashboard.title ?? "this dashboard"}</strong>
        </section>
      ) : null}
      {starters.length > 0 ? (
        <section aria-label="Starters" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
          {starters.map((starter) => (
            <Button key={starter} style={{ textAlign: "left", whiteSpace: "normal", height: "auto" }}>{starter}</Button>
          ))}
        </section>
      ) : null}
    </div>
  );
}
