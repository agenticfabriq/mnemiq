import React, { useState, type CSSProperties } from "react";
import { Alert, Button, Checkbox, Table, Typography, theme } from "antd";
import type { PanelApi } from "../api";
import {
  isChangePreview,
  showResults,
  type BuildProposal,
  type ChangePreview,
  type Choose,
  type Message,
  type Results,
} from "../conversation";
import { pageAppRoot } from "../page";
import type { Cell } from "../shared";
import { confirmInPanel, declineInPanel } from "../talk";

const dashboardUrl = (id: number) => `${pageAppRoot()}/dashboard/${id}/`;

// What became of a proposal, once the person answered it.
const SETTLED: Record<string, string> = {
  done: "Done.",
  applying: "This may or may not have been done; check the dashboard.",
  replanned: "Planned again, below.",
  declined: "Not now.",
  partial: "Done in part; ask again for the rest.",
  chosen: "Answered, below.",
  stale: "Nothing was changed: the dashboard changed since this preview. Ask again.",
};

type Props = {
  api: PanelApi;
  messages: Message[];
  conversationId: string | null;
  /** The dashboard on screen, when the person may read it: a proposal is confirmed only from its own. */
  onScreenId: number | null;
  /** Sends a message again where the person chose, after a question back. */
  onChoose: (text: string, choose: Choose) => void;
  busy: boolean;
};

/** The conversation as mnemiq keeps it: each message, with the dashboard it was sent from when that changed, and
 * each proposal's answers while it waits. */
export function Messages({ api, messages, conversationId, onScreenId, onChoose, busy }: Props) {
  const { token } = theme.useToken();
  const muted: CSSProperties = { color: token.colorTextSecondary };
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: token.padding }}>
      {messages.map((message, position) => {
        const before = position > 0 ? messages[position - 1].dashboard?.id ?? null : undefined;
        const here = message.dashboard?.id ?? null;
        const moved = message.role === "person" && here !== null && here !== before;
        return (
          <div key={message.index} data-message={message.index}
            style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
            {moved ? <span style={{ ...muted, fontSize: token.fontSizeSM }}>on {message.dashboard?.title ?? "a dashboard"}</span> : null}
            <Shown api={api} message={message} previous={messages[position - 1]} conversationId={conversationId}
              onScreenId={onScreenId} onChoose={onChoose} busy={busy} muted={muted} />
          </div>
        );
      })}
    </div>
  );
}

function Shown({ api, message, previous, conversationId, onScreenId, onChoose, busy, muted }:
  Omit<Props, "messages"> & { message: Message; previous?: Message; muted: CSSProperties }) {
  const { token } = theme.useToken();
  if (message.role === "person") {
    return (
      <p style={{ margin: 0, alignSelf: "flex-end", maxWidth: "85%", padding: token.paddingXS,
        borderRadius: token.borderRadius, background: token.colorFillTertiary }}>{message.text}</p>
    );
  }
  if (message.kind === "answer" && message.answer) {
    return <KeptAnswer api={api} message={message} conversationId={conversationId} muted={muted} />;
  }
  if (message.kind === "clarify" && message.clarify) {
    const text = previous?.role === "person" ? previous.text : null;
    return (
      <section aria-label="Question back" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span>{message.clarify.question}</span>
        {text ? (
          <div style={{ display: "flex", gap: 8 }}>
            {(["ask", "build", "change"] as const).map((choose) => (
              <Button key={choose} size="small" disabled={busy} onClick={() => onChoose(text, choose)}>
                {choose === "ask" ? "Answer it" : choose === "build" ? "Build it" : "Change the dashboard"}
              </Button>
            ))}
          </div>
        ) : null}
      </section>
    );
  }
  if (message.proposal && isChangePreview(message.proposal)) {
    return <Preview api={api} message={message} preview={message.proposal} onScreenId={onScreenId} muted={muted} />;
  }
  if (message.proposal) {
    return <Proposal api={api} message={message} proposal={message.proposal} onScreenId={onScreenId} muted={muted} />;
  }
  if (message.offer) {
    const waiting = message.outcome?.state === "offered";
    return (
      <section aria-label="Offer" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span>{message.text}</span>
        {message.offer.reason ? <span style={muted}>{message.offer.reason}</span> : null}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <a href={dashboardUrl(message.offer.dashboard_id)}>Open it</a>
          {waiting ? (
            <>
              <Button size="small" onClick={() => void confirmInPanel(api, message, { choice: "add", dashboard_id: onScreenId })}>
                Add these charts to it
              </Button>
              <Button size="small" onClick={() => void confirmInPanel(api, message, { choice: "new", dashboard_id: onScreenId })}>
                Build a new one
              </Button>
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
 * The rows live in this card's state alone, so closing the panel drops them. */
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

/** A build waiting for "Build it": its plan, and Build it only on the dashboard it is for. */
function Proposal({ api, message, proposal, onScreenId, muted }:
  { api: PanelApi; message: Message; proposal: BuildProposal; onScreenId: number | null; muted: CSSProperties }) {
  const { token } = theme.useToken();
  const [sending, setSending] = useState(false);
  const state = message.outcome?.state;
  const here = proposal.dashboard_id === null || proposal.dashboard_id === onScreenId;
  const answer = async (confirm: boolean) => {
    setSending(true);
    try {
      if (confirm) await confirmInPanel(api, message, { dashboard_id: onScreenId });
      else await declineInPanel(api, message);
    } finally {
      setSending(false);
    }
  };
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
      {state === "proposed" ? (
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Button type="primary" size="small" disabled={!here || sending} onClick={() => void answer(true)}>Build it</Button>
          <Button size="small" disabled={sending} onClick={() => void answer(false)}>Not now</Button>
          {!here && proposal.dashboard_id !== null ? (
            <a href={dashboardUrl(proposal.dashboard_id)}>Open the dashboard this is for</a>
          ) : null}
        </div>
      ) : <span style={muted}>{SETTLED[state ?? ""] ?? ""}</span>}
    </section>
  );
}

/** A change waiting for "Apply": each chart's edit as it will be written, the charts it would remove (each confirmed
 * by its own tick), and Apply only on the dashboard it was previewed on. */
function Preview({ api, message, preview, onScreenId, muted }:
  { api: PanelApi; message: Message; preview: ChangePreview; onScreenId: number | null; muted: CSSProperties }) {
  const { token } = theme.useToken();
  const [removing, setRemoving] = useState<number[]>([]);
  const [sending, setSending] = useState(false);
  const state = message.outcome?.state;
  const here = preview.dashboard_id === onScreenId;
  const answer = async (confirm: boolean) => {
    setSending(true);
    try {
      if (confirm) {
        await confirmInPanel(api, message, { dashboard_id: onScreenId,
          ...(removing.length ? { remove_chart_ids: removing } : {}) });
      } else await declineInPanel(api, message);
    } finally {
      setSending(false);
    }
  };
  const nothing = preview.edits.length === 0 && removing.length === 0;
  return (
    <section aria-label="Preview" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS,
      padding: token.paddingSM, border: `1px solid ${token.colorBorderSecondary}`, borderRadius: token.borderRadius }}>
      {preview.edits.length ? (
        <ul aria-label="Edits" style={{ margin: 0, paddingLeft: 18 }}>
          {preview.edits.map((edit) => <li key={edit.chart_id}>{edit.said}</li>)}
        </ul>
      ) : null}
      {preview.would_remove.length && state === "proposed" ? (
        <section aria-label="To remove" style={{ display: "flex", flexDirection: "column" }}>
          {preview.would_remove.map((chart) => (
            <Checkbox key={chart.chart_id} checked={removing.includes(chart.chart_id)}
              onChange={(event) => setRemoving((now) => event.target.checked
                ? [...now, chart.chart_id] : now.filter((id) => id !== chart.chart_id))}>
              Remove {chart.title ?? `chart ${chart.chart_id}`}
            </Checkbox>
          ))}
        </section>
      ) : null}
      {preview.left.map((left) => <span key={`l${left.chart_id}`} style={muted}>Chart {left.chart_id} is left as it is: {left.reason}</span>)}
      {preview.not_editor.map((entry) => (
        <span key={`e${entry.chart_id}`} style={muted}>
          Chart {entry.chart_id} is not yours to edit{entry.editors.length ? `; its editors are ${entry.editors.join(", ")}` : ""}.
        </span>
      ))}
      {state === "proposed" ? (
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Button type="primary" size="small" disabled={!here || sending || nothing} onClick={() => void answer(true)}>Apply</Button>
          <Button size="small" disabled={sending} onClick={() => void answer(false)}>Not now</Button>
          {!here ? <a href={dashboardUrl(preview.dashboard_id)}>Open the dashboard this is for</a> : null}
        </div>
      ) : (
        <span style={muted}>
          {state === "partial"
            ? `Partly done: chart${(message.outcome?.changed ?? []).length === 1 ? "" : "s"} ${(message.outcome?.changed ?? []).join(", ")} changed; the rest changed since the preview. Ask again for those.`
            : SETTLED[state ?? ""] ?? ""}
        </span>
      )}
    </section>
  );
}
