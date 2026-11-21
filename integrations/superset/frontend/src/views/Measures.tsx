import React, { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { Alert, Button, Form, Input, Modal, Select, Spin, theme } from "antd";
import type { PanelApi } from "../api";
import {
  editMeasure,
  listMeasures,
  measureValueText,
  removeMeasure,
  saveMeasure,
  type ActionAnswer,
  type ChartOfMeasure,
  type MeasureEdit,
  type MeasureItem,
} from "../measures";

const KINDS = ["currency", "percent", "count", "number"] as const;

type Result = { answer: ActionAnswer; charts: ChartOfMeasure[] } | { refused: string };

/** The dashboard's measures: read each time the section opens, held in this component's state only (a value is the
 * person's, computed now, and kept nowhere else), with Edit, Remove (the charts named first) and Save to dataset. */
export function Measures({ api, dashboardId }: { api: PanelApi; dashboardId: number }) {
  const { token } = theme.useToken();
  const [open, setOpen] = useState(false);
  const [measures, setMeasures] = useState<MeasureItem[] | null>(null);
  const [listRefused, setListRefused] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [removing, setRemoving] = useState<{ measure: MeasureItem; charts: ChartOfMeasure[]; note: string | null } | null>(null);
  const [editing, setEditing] = useState<MeasureItem | null>(null);
  const reading = useRef<AbortController | null>(null);

  const read = useCallback(() => {
    reading.current?.abort();
    const asking = new AbortController();
    reading.current = asking;
    setMeasures(null);
    setListRefused(null);
    listMeasures(api, dashboardId, asking.signal)
      .then((answer) => {
        if (asking.signal.aborted) return;
        if (answer.ok && Array.isArray(answer.body?.measures)) setMeasures(answer.body.measures);
        else setListRefused(answer.ok ? "mnemiq's list of measures could not be read." : answer.refusal.sentence);
      })
      .catch(() => {});
  }, [api, dashboardId]);

  useEffect(() => () => reading.current?.abort(), []);

  const toggle = () => {
    if (open) {
      // Closed, the values are dropped: the next opening reads them again.
      reading.current?.abort();
      setMeasures(null);
      setResult(null);
      setOpen(false);
      return;
    }
    setOpen(true);
    read();
  };

  const done = (measure: MeasureItem, answer: Awaited<ReturnType<typeof saveMeasure>>) => {
    setResult(answer.ok ? { answer: answer.body, charts: measure.charts } : { refused: answer.refusal.sentence });
    read();
  };

  const muted: CSSProperties = { color: token.colorTextSecondary };
  return (
    <section aria-label="Measures" style={{ display: "flex", flexDirection: "column", gap: token.paddingXS }}>
      <Button size="small" type="text" aria-expanded={open} onClick={toggle} style={{ alignSelf: "flex-start" }}>
        {open ? "▾" : "▸"} Measures
      </Button>
      {open ? (
        <>
          {result ? <ResultView result={result} /> : null}
          {listRefused ? <Alert type="warning" showIcon message={listRefused} /> : null}
          {open && measures === null && !listRefused ? <Spin size="small" /> : null}
          {measures && measures.length === 0 ? <span style={muted}>This dashboard keeps no measures.</span> : null}
          {measures && measures.length > 0 ? (
            <ul style={{ margin: 0, paddingLeft: 0, listStyle: "none", display: "flex", flexDirection: "column",
              gap: token.paddingXS }}>
              {measures.map((measure) => (
                <li key={measure.name} style={{ padding: token.paddingXS, borderRadius: token.borderRadius,
                  background: token.colorFillQuaternary }}>
                  <div style={{ display: "flex", gap: token.paddingXS, alignItems: "baseline" }}>
                    <strong style={{ flex: 1 }}>{measure.label}</strong>
                    <span>{measureValueText(measure.value, measure.kind, measure.currency, navigator.language)}</span>
                  </div>
                  {measure.value_note ? <div style={muted}>{measure.value_note}</div> : null}
                  <div style={muted}>{measure.charts.map((chart) => chart.title ?? `chart ${chart.chart_id}`).join(", ")}</div>
                  <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
                    <Button size="small" onClick={() => setEditing(measure)}>Edit</Button>
                    <Button size="small" onClick={() => setRemoving({ measure, charts: measure.charts, note: null })}>
                      Remove
                    </Button>
                    {!measure.saved_metric ? (
                      <Button size="small" onClick={async () =>
                        done(measure, await saveMeasure(api, dashboardId, measure.name))}>Save to dataset</Button>
                    ) : null}
                  </div>
                </li>
              ))}
            </ul>
          ) : null}
        </>
      ) : null}
      <Modal
        open={removing !== null}
        title={removing ? `Remove ${removing.measure.label}` : undefined}
        okText="Remove from these charts"
        okButtonProps={{ danger: true }}
        onCancel={() => setRemoving(null)}
        onOk={async () => {
          if (!removing) return;
          const { measure, charts } = removing;
          const answer = await removeMeasure(api, dashboardId, measure.name, charts.map((chart) => chart.chart_id));
          if ("chartsChanged" in answer) {
            // The measure's charts changed since they were shown: show the new list, and remove nothing yet.
            setRemoving({ measure, charts: answer.chartsChanged.charts, note: answer.chartsChanged.message });
            return;
          }
          setRemoving(null);
          done({ ...measure, charts }, answer);
        }}
      >
        {removing?.note ? <Alert type="warning" showIcon message={removing.note} style={{ marginBottom: 8 }} /> : null}
        <p>It is removed from these charts:</p>
        <ul>{removing?.charts.map((chart) => <li key={chart.chart_id}>{chart.title ?? `chart ${chart.chart_id}`}</li>)}</ul>
      </Modal>
      <Modal open={editing !== null} title={editing ? `Edit ${editing.label}` : undefined} footer={null}
        onCancel={() => setEditing(null)} destroyOnHidden>
        {editing ? (
          <EditForm measure={editing} onSave={async (edit) => {
            const measure = editing;
            setEditing(null);
            done(measure, await editMeasure(api, dashboardId, measure.name, edit));
          }} />
        ) : null}
      </Modal>
    </section>
  );
}

function EditForm({ measure, onSave }: { measure: MeasureItem; onSave: (edit: MeasureEdit) => void }) {
  const [form] = Form.useForm<MeasureEdit>();
  const kind = Form.useWatch("kind", form) ?? measure.kind;
  return (
    <Form form={form} name="measure-edit" layout="vertical"
      initialValues={{ label: measure.label, expression: measure.expression, kind: measure.kind,
        currency: measure.currency ?? undefined, description: measure.description }}
      onFinish={(values) => onSave({ label: values.label, expression: values.expression, kind: values.kind,
        currency: values.kind === "currency" ? (values.currency || null) : null, description: values.description ?? "" })}>
      <Form.Item label="Label" name="label" rules={[{ required: true }]}><Input /></Form.Item>
      <Form.Item label="Expression" name="expression" rules={[{ required: true }]}><Input.TextArea autoSize /></Form.Item>
      <Form.Item label="Kind" name="kind"><Select options={KINDS.map((k) => ({ value: k, label: k }))} /></Form.Item>
      {kind === "currency" ? (
        <Form.Item label="Currency" name="currency" rules={[{ pattern: /^[A-Z]{3}$/, message: "Three capital letters" }]}>
          <Input />
        </Form.Item>
      ) : null}
      <Form.Item label="Description" name="description"><Input.TextArea autoSize /></Form.Item>
      <Button type="primary" htmlType="submit">Save changes</Button>
    </Form>
  );
}

/** What an action did, its charts named by title. */
function ResultView({ result }: { result: Result }) {
  if ("refused" in result) {
    return <div role="status" aria-label="Result"><Alert type="error" showIcon message={result.refused} /></div>;
  }
  const { answer, charts } = result;
  const named = (ids: number[] = []) =>
    ids.map((id) => charts.find((chart) => chart.chart_id === id)?.title ?? `chart ${id}`).join(", ");
  const lines: string[] = [];
  if (answer.changed?.length) lines.push(`Changed: ${named(answer.changed)}`);
  if (answer.left_by_hand?.length) lines.push(`Left as changed by hand: ${named(answer.left_by_hand)}`);
  for (const entry of answer.not_editor ?? []) {
    const what = entry.chart_id !== undefined ? named([entry.chart_id]) : "this measure's dataset";
    lines.push(`You do not edit ${what}; its editors: ${(entry.editors ?? []).join(", ") || "none named"}`);
  }
  if (answer.blocked_by_report?.length) lines.push(`Kept, a report uses it: ${named(answer.blocked_by_report)}`);
  if (answer.could_not_read?.length) lines.push(`Could not be read: ${named(answer.could_not_read)}`);
  return (
    <div role="status" aria-label="Result">
      <Alert type={answer.not_editor?.length ? "warning" : "success"} showIcon message={answer.message}
        description={lines.length ? <ul style={{ margin: 0, paddingLeft: 18 }}>{lines.map((line) => <li key={line}>{line}</li>)}</ul> : undefined} />
    </div>
  );
}
