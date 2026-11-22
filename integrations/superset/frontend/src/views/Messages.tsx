import React, { useState, type CSSProperties } from "react";
import { Alert, Button, Checkbox, Table, Typography, theme } from "antd";
import type { PanelApi } from "../api";
import { ALREADY_BUILDING } from "../build";
import {
  isChangePreview,
  showResults,
  type BuildProposal,
  type ChangePreview,
  type Choose,
  type Message,
  type Page,
  type Results,
} from "../conversation";
import { pageAppRoot } from "../page";
import type { Cell, Turn } from "../shared";
import { confirmInPanel, declineInPanel } from "../talk";
import { AnswerCard } from "./AnswerCard";

const dashboardUrl = (id: number) => `${pageAppRoot()}/dashboard/${id}/`;

// What became of a proposal, once the person answered it.
const SETTLED: Record<string, string> = {
  done: "Done.",
  applying: "This may or may not have been done; check the dashboard.",
  replanned: "Planned again, below.",
  declined: "Not now.",
  chosen: "Answered, below.",
  stale: "Nothing was changed: the dashboard changed since this preview. Ask again.",
  partial: "Done in part; ask again for the rest.",
};
const DELETES = "Removing a chart deletes it: Superset moves it to its trash, or deletes it for good where it keeps none.";

type Props = {
  api: PanelApi;
  messages: Message[];
  conversationId: string | null;
  /** The dashboard on screen, as the page context last read it: what Build it and Apply are offered on. Each confirm
   * reads it again from the address when it is clicked (`pageNow`). */
  onScreenId: number | null;
  pageNow: () => Promise<Page>;
  /** Sends a message again where the person chose, after a question back. */
  onChoose: (text: string, choose: Choose) => void;
  /** A message, confirm or decline in flight. */
  busy: boolean;
  /** A build being planned or run on this page, which another Build it waits for. */
  building: boolean;
  /** The answer just given, shown as it streamed in place of its kept card. */
  fresh: { conversationId: string; index: number; turn: Turn } | null;
};

/** The conversation as mnemiq keeps it: each message, with the dashboard it was sent from when that changed, and
 * each proposal's answers while it waits. */
export function Messages(props: Props) {
  const { messages } = props;
  const { token } = theme.useToken();
  const muted: CSSProperties = { color: token.colorTextSecondary };
  let lastPlace: number | null | undefined;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: token.padding }}>
      {messages.map((message) => {
        let marker: string | null = null;
        if (message.role === "person") {
          const here = message.dashboard?.id ?? null;
          if (here !== lastPlace && !(lastPlace === undefined && here === null)) {
            marker = here === null ? "off a dashboard" : `on ${message.dashboard?.title ?? "a dashboard"}`;
          }
          lastPlace = here;
        }
        return (
          <div key={`${props.conversationId}:${message.index}`} data-message={message.index}
            style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
            {marker ? <span style={{ ...muted, fontSize: token.fontSizeSM }}>{marker}</span> : null}
            <Shown {...props} message={message} muted={muted} />
          </div>
        );
      })}
    </div>
  );
}

function Shown(props: Props & { message: Message; muted: CSSProperties }) {
  const { api, message, messages, conversationId, onChoose, busy, muted, fresh, pageNow, building } = props;
  const { token } = theme.useToken();
  if (message.role === "person") {
    return (
      <p style={{ margin: 0, alignSelf: "flex-end", maxWidth: "85%", padding: token.paddingXS,
        borderRadius: token.borderRadius, background: token.colorFillTertiary }}>{message.text}</p>
    );
  }
  if (message.kind === "answer" && message.answer) {
    if (fresh && fresh.conversationId === conversationId && fresh.index === message.index) {
      return <AnswerCard turn={fresh.turn} />;
    }
    return <KeptAnswer api={api} message={message} conversationId={conversationId} muted={muted} />;
  }
  if (message.kind === "clarify" && message.clarify) {
    // The message it replies to, by its index; before mnemiq named it, the one just before.
    const replied = typeof message.replies_to === "number"
      ? messages.find((m) => m.index === message.replies_to)
      : messages.find((m) => m.index === message.index - 1);
    const text = replied?.role === "person" ? replied.text : null;
    return (
      <section aria-label="Question back" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span>{message.clarify.question}</span>
        {text ? (
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {(["ask", "build", "change"] as const).map((choose) => (
              <Button key={choose} size="small" disabled={busy} onClick={() => onChoose(text, choose)}>
                {choose === "ask" ? "Answer it" : choose === "build" ? "Plan a build" : "Change the dashboard"}
              </Button>
            ))}
          </div>
        ) : null}
      </section>
    );
  }
  if (message.proposal && isChangePreview(message.proposal)) {
    return <Preview {...props} preview={message.proposal} />;
  }
  if (message.proposal) {
    return <Proposal {...props} proposal={message.proposal} />;
  }
  if (message.offer) {
    const waiting = message.outcome?.state === "offered";
    const choose = (choice: "add" | "new") => void confirmInPanel(api, message, { choice }, pageNow);
    return (
      <section aria-label="Offer" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span>{message.text}</span>
        {message.offer.reason ? <span style={muted}>{message.offer.reason}</span> : null}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <a href={dashboardUrl(message.offer.dashboard_id)}>Open it</a>
          {waiting ? (
            <>
              <Button size="small" disabled={busy || building} onClick={() => choose("add")}>Add these charts to it</Button>
              <Button size="small" disabled={busy || building} onClick={() => choose("new")}>Build a new one</Button>
            </>
          ) : <span style={muted}>{SETTLED[message.outcome?.state ?? ""] ?? ""}</span>}
        </div>
      </section>
    );
  }
  if (message.refused) {
    return (
      <div>
        <span>{message.refused.reason}</span>
        <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
          {message.refused.parts.map((part) => <li key={part.part}>{part.part}: {part.reason}</li>)}
        </ul>
      </div>
    );
  }
  if (message.shown) {
    return (
      <div>
        <span>{message.text}</span>
        <div style={{ marginTop: 4 }}><a href={dashboardUrl(message.shown.dashboard_id)}>Open it</a></div>
      </div>
    );
  }
  return <Typography.Paragraph style={{ margin: 0 }}>{message.text}</Typography.Paragraph>;
}

/** A kept answer: its sentence as written then, its SQL folded, and its table run again as the person on request.
 * The rows live in this card's state alone, so closing the panel, or moving to another conversation, drops them. */
function KeptAnswer({ api, message, conversationId, muted }:
  { api: PanelApi; message: Message; conversationId: string | null; muted: CSSProperties }) {
  const answer = message.answer!;
  const [results, setResults] = useState<Results | null>(null);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const show = async () => {
    if (conversationId === null) return;
    setLoading(true);
    setFailed(null);
    const found = await showResults(api, conversationId, message.index);
    setLoading(false);
    if (found.ok) setResults(found.body);
    else setFailed(found.refusal.sentence);
  };
  return (
    <article style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {answer.sentence ? <Typography.Paragraph style={{ margin: 0 }}>{answer.sentence}</Typography.Paragraph> : null}
      {message.at ? <span style={{ ...muted, fontSize: 12 }}>Answered {message.at.slice(0, 10)}</span> : null}
      {answer.sql ? (
        <details>
          <summary style={muted}>SQL</summary>
          <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}><code>{answer.sql}</code></pre>
        </details>
      ) : null}
      {answer.sql && results === null ? (
        <Button size="small" style={{ alignSelf: "flex-start" }} loading={loading} onClick={() => void show()}>
          Show results
        </Button>
      ) : null}
      {failed ? <Alert type="warning" showIcon message={failed} /> : null}
      {results && "refused" in results ? <Alert type="warning" showIcon message={results.refused} /> : null}
      {results && !("refused" in results) ? (
        <>
          <Table size="small" pagination={false} scroll={{ x: true }} rowKey="key"
            columns={results.columns.map((name, i) => ({
              title: name, dataIndex: String(i), key: String(i),
              render: (value: Cell) => (value === null ? <span style={muted} title="no value">—</span> : String(value)),
            }))}
            dataSource={results.rows.map((row, r) => ({ key: r, ...Object.fromEntries(row.map((v, i) => [String(i), v])) }))} />
          {results.truncated || results.capped ? (
            <span style={muted}>{results.rows.length} of {results.capped ? "at least " : ""}{results.row_count} rows shown</span>
          ) : null}
        </>
      ) : null}
    </article>
  );
}

/** A build waiting for "Build it": its plan, and Build it only on the dashboard it is for, and not while this page
 * plans or runs another build. A build left mid-confirm (`applying`) can be confirmed again: mnemiq reads the build's
 * own state. */
function Proposal({ api, message, proposal, onScreenId, muted, busy, building, pageNow }:
  Props & { message: Message; proposal: BuildProposal; muted: CSSProperties }) {
  const { token } = theme.useToken();
  const state = message.outcome?.state;
  const here = proposal.dashboard_id === null || proposal.dashboard_id === onScreenId;
  const waiting = state === "proposed" || state === "applying";
  return (
    <section aria-label="Proposal" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS,
      padding: token.paddingSM, border: `1px solid ${token.colorBorderSecondary}`, borderRadius: token.borderRadius }}>
      <span>{message.text}</span>
      {proposal.plan?.charts?.length ? (
        <section aria-label="Plan" style={muted}>
          {proposal.plan.charts.map((chart) => (
            <div key={chart.key}>{chart.title} <span>({chart.family}{chart.measures?.length ? `: ${chart.measures.join(", ")}` : ""})</span></div>
          ))}
        </section>
      ) : null}
      {proposal.left_out?.length ? (
        <div style={muted}>Already on the dashboard: {proposal.left_out.map((part) => part.part).join("; ")}</div>
      ) : null}
      {state === "applying" ? <span style={muted}>{SETTLED.applying}</span> : null}
      {waiting ? (
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <Button type="primary" size="small" disabled={!here || busy || building}
            onClick={() => void confirmInPanel(api, message, {}, pageNow)}>Build it</Button>
          <Button size="small" disabled={busy} onClick={() => void declineInPanel(api, message)}>Not now</Button>
          {!here && proposal.dashboard_id !== null ? (
            <a href={dashboardUrl(proposal.dashboard_id)}>Open the dashboard this is for</a>
          ) : null}
          {building ? <span style={muted}>{ALREADY_BUILDING}</span> : null}
        </div>
      ) : <span style={muted}>{SETTLED[state ?? ""] ?? ""}</span>}
    </section>
  );
}

/** A change waiting for "Apply": mnemiq's words for it (what will and will not change), each chart's edit as it
 * will be written, the charts it would remove (each confirmed by its own tick, and said to delete), and Apply only on
 * the dashboard it was previewed on. Once answered, what came of it, chart by chart. */
function Preview({ api, message, preview, onScreenId, muted, busy, pageNow }:
  Props & { message: Message; preview: ChangePreview; muted: CSSProperties }) {
  const { token } = theme.useToken();
  const [removing, setRemoving] = useState<number[]>([]);
  const state = message.outcome?.state;
  const outcome = message.outcome;
  const here = preview.dashboard_id === onScreenId;
  const titled = (id: number) => preview.edits.find((edit) => edit.chart_id === id)?.title
    ?? preview.would_remove.find((chart) => chart.chart_id === id)?.title ?? `chart ${id}`;
  const nothing = preview.edits.length === 0 && removing.length === 0;
  return (
    <section aria-label="Preview" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS,
      padding: token.paddingSM, border: `1px solid ${token.colorBorderSecondary}`, borderRadius: token.borderRadius }}>
      {message.text ? <span>{message.text}</span> : null}
      {preview.edits.length ? (
        <ul aria-label="Edits" style={{ margin: 0, paddingLeft: 18 }}>
          {preview.edits.map((edit) => <li key={edit.chart_id}>{edit.said}</li>)}
        </ul>
      ) : null}
      {preview.would_remove.length && state === "proposed" ? (
        <section aria-label="To remove" style={{ display: "flex", flexDirection: "column" }}>
          <span style={{ color: token.colorError }}>{DELETES}</span>
          {preview.would_remove.map((chart) => (
            <Checkbox key={chart.chart_id} checked={removing.includes(chart.chart_id)}
              onChange={(event) => setRemoving((now) => event.target.checked
                ? [...now, chart.chart_id] : now.filter((id) => id !== chart.chart_id))}>
              Delete {chart.title ?? `chart ${chart.chart_id}`}
            </Checkbox>
          ))}
        </section>
      ) : null}
      {state === "proposed" ? preview.not_editor.map((entry) => (
        <span key={`e${entry.chart_id}`} style={muted}>
          {titled(entry.chart_id ?? 0)} is not yours to edit{entry.editors.length ? `; its editors are ${entry.editors.join(", ")}` : ""}.
        </span>
      )) : null}
      {state === "proposed" ? (
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <Button type="primary" danger={removing.length > 0} size="small" disabled={!here || busy || nothing}
            onClick={() => void confirmInPanel(api, message,
              removing.length ? { remove_chart_ids: removing } : {}, pageNow)}>Apply</Button>
          <Button size="small" disabled={busy} onClick={() => void declineInPanel(api, message)}>Not now</Button>
          {!here ? <a href={dashboardUrl(preview.dashboard_id)}>Open the dashboard this is for</a> : null}
        </div>
      ) : (
        <section aria-label="What came of it" style={{ display: "flex", flexDirection: "column", gap: 2, ...muted }}>
          <span>{SETTLED[state ?? ""] ?? ""}</span>
          {outcome?.changed?.length ? <span>Changed: {outcome.changed.map(titled).join(", ")}.</span> : null}
          {outcome?.removed?.length ? <span>Deleted: {outcome.removed.map(titled).join(", ")}.</span> : null}
          {(outcome?.left ?? []).map((left) => <span key={`l${left.chart_id}`}>{titled(left.chart_id)} left as it is: {left.reason}.</span>)}
          {(state === "done" || state === "partial") && here ? (
            <Button size="small" style={{ alignSelf: "flex-start" }} onClick={() => window.location.reload()}>
              Reload to see the changes
            </Button>
          ) : null}
        </section>
      )}
    </section>
  );
}
