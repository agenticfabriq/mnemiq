import React, { type CSSProperties } from "react";
import { Alert, Table, Typography, theme } from "antd";
import { CONNECTION_ENDED_CODE, disposition, REASONS, type Cell, type Turn } from "../shared";

/** One answer of mnemiq's: its sentence; a declined or failed answer's reason and next step; the SQL, folded; and the
 * rows it read, each cell as written, a blank one shown as a muted dash and never as 0. No SQL Lab button. */
export function AnswerCard({ turn }: { turn: Turn }) {
  const { token } = theme.useToken();
  const outcome = disposition(turn);
  if (outcome === "broken") {
    const cut = turn.errorCode === CONNECTION_ENDED_CODE;
    return (
      <Alert type="error" role="alert" showIcon
        message={turn.error ?? "The answer failed."}
        description={cut ? "The connection was cut, not the question refused." : undefined} />
    );
  }
  const answer = turn.answer;
  const sentence = turn.text || answer?.answer || "";
  const muted: CSSProperties = { color: token.colorTextTertiary };
  const reason = answer?.reason_code ? REASONS[answer.reason_code] : undefined;
  return (
    <article style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
      {sentence ? <Typography.Paragraph style={{ margin: 0 }}>{sentence}</Typography.Paragraph> : null}
      {(outcome === "declined" || outcome === "failed") && reason ? (
        <Alert type={outcome === "failed" ? "error" : "warning"} showIcon message={reason.title} description={reason.next} />
      ) : null}
      {answer?.sql ? (
        <details>
          <summary style={muted}>SQL</summary>
          <Typography>
            <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}><code>{answer.sql}</code></pre>
          </Typography>
        </details>
      ) : null}
      {answer?.preview && answer.preview.columns.length > 0 ? (
        <>
          <Table
            size="small"
            pagination={false}
            scroll={{ x: true }}
            rowKey="key"
            columns={answer.preview.columns.map((name, i) => ({
              title: name,
              dataIndex: String(i),
              key: String(i),
              render: (value: Cell) => (value === null ? <span style={muted} title="no value">—</span> : String(value)),
            }))}
            dataSource={answer.preview.rows.map((row, r) => ({
              key: r,
              ...Object.fromEntries(row.map((value, i) => [String(i), value])),
            }))}
          />
          {answer.preview.truncated || answer.preview.capped ? (
            <span style={muted}>
              {answer.preview.rows.length} of {answer.preview.capped ? "at least " : ""}{answer.preview.row_count} rows shown
            </span>
          ) : null}
        </>
      ) : null}
    </article>
  );
}
