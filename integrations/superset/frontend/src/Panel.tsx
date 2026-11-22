import React, { useEffect, useMemo, useState, useSyncExternalStore, type CSSProperties } from "react";
import { Alert, Button, Spin, theme } from "antd";
import { authentication, chat } from "@apache-superset/core";
import { PanelApi } from "./api";
import { driver } from "./build";
import type { Choose, Page } from "./conversation";
import { dashboardRefOf, pageAppRoot, resolveDashboard, usePageContext } from "./page";
import { probe, type Readiness } from "./readiness";
import { getSnapshot, subscribe } from "./store";
import { sendInPanel } from "./talk";
import { BuildCard } from "./views/BuildCard";
import { Composer } from "./views/Composer";
import { Conversation } from "./views/Conversation";
import { ConversationList } from "./views/ConversationList";
import { Header } from "./views/Header";
import { Measures } from "./views/Measures";
import { Messages } from "./views/Messages";
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

/** Where a message is sent from, worked out when it is sent: the dashboard on the path on screen then, resolved as the
 * person (Superset moves between dashboards with no page event, so the page context may be a second behind), or no
 * dashboard off one or on one they cannot read. */
async function pageNow(): Promise<Page> {
  const root = pageAppRoot();
  const ref = dashboardRefOf(window.location.pathname, root);
  if (ref === null) return { dashboardId: null, dashboardTitle: null, dashboardHasCharts: null };
  const found = await resolveDashboard(root, ref);
  return found.readable && found.id !== null
    ? { dashboardId: found.id, dashboardTitle: found.title, dashboardHasCharts: found.hasCharts ?? null }
    : { dashboardId: null, dashboardTitle: null, dashboardHasCharts: null };
}

export const DESCRIBE = "Describe what this dashboard should show";
const ANYTHING = "Ask about your data, build a dashboard, or change this one";
const OFF_A_DASHBOARD = "Ask about your data, or describe a dashboard to build";

/** The panel for a person mnemiq answers: their conversations and New; the dashboard on screen, when they may read
 * it; their unfinished builds; the opening questions until the conversation has a message; the conversation, the
 * message in flight and the build running; the dashboard's measures; and one box to write in. */
function Ready({ api, starters }: { api: PanelApi; starters: string[] }) {
  const { dashboard } = usePageContext();
  const { conversationId, messages, turns, sending, notice } = useSyncExternalStore(subscribe, getSnapshot);
  const { token } = theme.useToken();
  useEffect(() => driver.attach(api), [api]);
  const onScreen = dashboard?.readable && dashboard.id !== null ? dashboard : null;
  const send = (text: string, choose?: Choose) => {
    if (sending) return false;
    void pageNow().then((page) => sendInPanel(api, text, page, choose));
    return true;
  };
  const placeholder = onScreen === null ? OFF_A_DASHBOARD : onScreen.hasCharts === false ? DESCRIBE : ANYTHING;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: token.padding }}>
      <ConversationList api={api} current={conversationId} busy={sending} />
      {onScreen ? (
        <section aria-label="This dashboard" style={{ color: token.colorTextSecondary }}>
          On <strong style={{ color: token.colorText }}>{onScreen.title ?? "this dashboard"}</strong>
        </section>
      ) : null}
      <UnfinishedBuilds api={api} dashboardId={onScreen?.id ?? null} />
      {messages.length === 0 && turns.length === 0 && starters.length > 0 ? (
        <section aria-label="Starters" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
          {starters.map((starter) => (
            <Button key={starter} disabled={sending} onClick={() => send(starter)}
              style={{ textAlign: "left", whiteSpace: "normal", height: "auto" }}>{starter}</Button>
          ))}
        </section>
      ) : null}
      <Messages api={api} messages={messages} conversationId={conversationId} onScreenId={onScreen?.id ?? null}
        onChoose={(text, choose) => send(text, choose)} busy={sending} />
      <Conversation turns={turns} />
      <BuildCard onScreenId={onScreen?.id ?? null} />
      {notice ? <Alert type="warning" showIcon message={notice} /> : null}
      {onScreen?.id != null ? <Measures key={onScreen.id} api={api} dashboardId={onScreen.id} /> : null}
      <Composer busy={sending} placeholder={placeholder} onSend={(text) => send(text)} />
    </div>
  );
}
