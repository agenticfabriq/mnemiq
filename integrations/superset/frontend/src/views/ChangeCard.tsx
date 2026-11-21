import React, { useSyncExternalStore, type CSSProperties } from "react";
import { Alert, Button, Spin, Typography, theme } from "antd";
import type { PanelApi } from "../api";
import { confirmInPanel, keepProposed } from "../change";
import { getSnapshot, subscribe } from "../store";

const chart = (id: number) => `chart ${id}`;

/** The change by chat on this page: the request; mnemiq's words; what changed, what was left and why, what was
 * dropped, whose charts the person does not edit (with their editors), and what was removed; a proposed removal to
 * confirm or keep; and a chart added, which runs as a build. */
export function ChangeCard({ api }: { api: PanelApi }) {
  const { change } = useSyncExternalStore(subscribe, getSnapshot);
  const { token } = theme.useToken();
  if (change === null) return null;
  const muted: CSSProperties = { color: token.colorTextSecondary };
  const answer = change.answer;
  const lines: string[] = [];
  if (answer) {
    if (answer.changed.length) lines.push(`Changed: ${answer.changed.map(chart).join(", ")}`);
    for (const left of answer.left) lines.push(`Left: ${chart(left.chart_id)}: ${left.reason}`);
    for (const dropped of answer.dropped) lines.push(`Dropped: ${chart(dropped.chart_id)}: ${dropped.key}`);
    for (const entry of answer.not_editor) {
      lines.push(`You do not edit ${chart(entry.chart_id)}; its editors: ${(entry.editors ?? []).join(", ") || "none named"}`);
    }
    if (answer.removed.length) lines.push(`Removed: ${answer.removed.map(chart).join(", ")}`);
  }
  const added = answer?.added ?? null;
  const addedText = added === null ? null
    : "error" in added ? `The chart to add was not planned: ${typeof added.error?.detail === "string" ? added.error.detail : "mnemiq refused it"}.`
    : "build" in added ? "The chart to add is being built; its build shows above."
    : "offer" in added ? added.offer.message
    : "refused" in added ? added.refused.reason
    : "shown" in added ? added.shown.message
    : null;
  const madeChanges = answer !== null && (answer.changed.length > 0 || answer.removed.length > 0);
  return (
    <section aria-label="Change" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS,
      padding: token.paddingSM, border: `1px solid ${token.colorBorderSecondary}`, borderRadius: token.borderRadius }}>
      <Typography.Text strong>{change.request}</Typography.Text>
      {change.sending ? <div role="status" aria-label="Changing" style={muted}><Spin size="small" /> Changing the dashboard</div> : null}
      {change.refused ? <Alert type="error" showIcon message={change.refused} /> : null}
      {answer?.message ? <span>{answer.message}</span> : null}
      {lines.length ? <ul style={{ margin: 0, paddingLeft: 18 }}>{lines.map((line) => <li key={line}>{line}</li>)}</ul> : null}
      {answer && answer.would_remove.length ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span>This would remove: {answer.would_remove.map((c) => c.title ?? chart(c.chart_id)).join(", ")}</span>
          <div style={{ display: "flex", gap: 8 }}>
            <Button size="small" danger disabled={change.sending}
              onClick={() => confirmInPanel(api, change, answer.would_remove.map((c) => c.chart_id))}>Remove them</Button>
            <Button size="small" disabled={change.sending} onClick={() => keepProposed()}>Keep them</Button>
          </div>
        </div>
      ) : null}
      {addedText ? <span style={muted}>{addedText}</span> : null}
      {change.note ? <span style={muted}>{change.note}</span> : null}
      {madeChanges ? (
        <Button size="small" style={{ alignSelf: "flex-start" }} onClick={() => window.location.reload()}>
          Reload to see the changes
        </Button>
      ) : null}
    </section>
  );
}
