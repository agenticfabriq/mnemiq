import React, { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { Alert, Button, Form, Input, Modal, Select, Spin, theme } from "antd";
import type { PanelApi } from "../api";
import { CHECK_WORDS } from "../build";
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
// mnemiq's own limits for an edit (`EditBody`): longer is refused.
const MAX_LABEL = 120;
const MAX_EXPRESSION = 4000;
const MAX_DESCRIPTION = 500;

type Action = "edit" | "remove" | "save";
type Result = { action: Action; answer: ActionAnswer; charts: ChartOfMeasure[] } | { refused: string };

const chartName = (chart: ChartOfMeasure) => chart.title ?? `chart ${chart.chart_id}`;

/** The dashboard's measures: read each time the section opens, held in this component's state only (a value is the
 * person's, computed now, and kept nowhere else), each with its definition and its charts' checks, and with Edit,
 * Remove (which deletes the charts that show it, named first) and Save to dataset. One action runs at a time: the
 * others wait until it answers. */
export function Measures({ api, dashboardId }: { api: PanelApi; dashboardId: number }) {
  const { token } = theme.useToken();
  const [open, setOpen] = useState(false);
  const [measures, setMeasures] = useState<MeasureItem[] | null>(null);
  const [listRefused, setListRefused] = useState<string | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [removing, setRemoving] = useState<{ measure: MeasureItem; charts: ChartOfMeasure[]; note: string | null } | null>(null);
  const [editing, setEditing] = useState<MeasureItem | null>(null);
  const [pending, setPending] = useState(false);
  const reading = useRef<AbortController | null>(null);
  // Whether the section is open now: an action answered after it closed (or after the panel left this dashboard)
  // reads nothing more.
  const shown = useRef(false);
  const acting = useRef(false);

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

  useEffect(() => () => {
    shown.current = false;
    reading.current?.abort();
  }, []);

  const toggle = () => {
    if (open) {
      // Closed, the values are dropped: the next opening reads them again.
      shown.current = false;
      reading.current?.abort();
      setMeasures(null);
      setResult(null);
      setOpen(false);
      return;
    }
    shown.current = true;
    setOpen(true);
    read();
  };

  /** Runs one action, unless one is already running: the others wait, and a second click sends nothing. */
  const act = async (action: Action, measure: MeasureItem,
    send: () => Promise<Awaited<ReturnType<typeof saveMeasure>> | null>): Promise<boolean> => {
    if (acting.current) return false;
    acting.current = true;
    setPending(true);
    try {
      const answer = await send();
      if (answer === null) return false;
      if (shown.current) {
        setResult(answer.ok ? { action, answer: answer.body, charts: measure.charts } : { refused: answer.refusal.sentence });
        read();
      }
      return true;
    } finally {
      acting.current = false;
      setPending(false);
    }
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
          {measures === null && !listRefused ? <Spin size="small" /> : null}
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
                  <code style={{ ...muted, fontSize: token.fontSizeSM, wordBreak: "break-all" }}>{measure.expression}</code>
                  <div style={muted}>
                    {measure.charts.map((chart) => `${chartName(chart)}${chart.check
                      ? `: ${CHECK_WORDS[chart.check as keyof typeof CHECK_WORDS] ?? chart.check}` : ""}`).join("; ")}
                  </div>
                  <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
                    {/* A saved metric is Superset's to edit, on its dataset: mnemiq refuses to change it here. */}
                    {!measure.saved_metric ? (
                      <Button size="small" disabled={pending} onClick={() => setEditing(measure)}>Edit</Button>
                    ) : null}
                    <Button size="small" disabled={pending}
                      onClick={() => setRemoving({ measure, charts: measure.charts, note: null })}>Remove</Button>
                    {!measure.saved_metric ? (
                      <Button size="small" disabled={pending} loading={pending && editing === null && removing === null}
                        onClick={() => void act("save", measure, () => saveMeasure(api, dashboardId, measure.name))}>
                        Save to dataset
                      </Button>
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
        okText="Delete these charts"
        okButtonProps={{ danger: true }}
        confirmLoading={pending}
        onCancel={() => !pending && setRemoving(null)}
        onOk={() => {
          if (!removing) return;
          const { measure, charts } = removing;
          void act("remove", measure, async () => {
            const answer = await removeMeasure(api, dashboardId, measure.name, charts.map((chart) => chart.chart_id));
            if ("chartsChanged" in answer) {
              // The measure's charts changed since they were shown: show the new list, and delete nothing yet.
              setRemoving({ measure, charts: answer.chartsChanged.charts, note: answer.chartsChanged.message });
              return null;
            }
            setRemoving(null);
            return answer;
          });
        }}
      >
        {removing?.note ? <Alert type="warning" showIcon message={removing.note} style={{ marginBottom: 8 }} /> : null}
        <p>
          Removing {removing?.measure.label} deletes the charts that show it: each is moved to Superset's trash, or deleted
          for good where Superset keeps none. A chart you do not edit is taken off this dashboard instead.
        </p>
        <ul>{removing?.charts.map((chart) => <li key={chart.chart_id}>{chartName(chart)}</li>)}</ul>
      </Modal>
      <Modal open={editing !== null} title={editing ? `Edit ${editing.label}` : undefined} footer={null}
        onCancel={() => !pending && setEditing(null)} destroyOnHidden>
        {editing ? (
          <EditForm measure={editing} saving={pending} onSave={(edit) => {
            const measure = editing;
            void act("edit", measure, async () => {
              const answer = await editMeasure(api, dashboardId, measure.name, edit);
              setEditing(null);
              return answer;
            });
          }} />
        ) : null}
      </Modal>
    </section>
  );
}

function EditForm({ measure, saving, onSave }: { measure: MeasureItem; saving: boolean;
  onSave: (edit: MeasureEdit) => void }) {
  const [form] = Form.useForm<MeasureEdit>();
  const kind = Form.useWatch("kind", form) ?? measure.kind;
  return (
    <Form form={form} name="measure-edit" layout="vertical" disabled={saving}
      initialValues={{ label: measure.label, expression: measure.expression, kind: measure.kind,
        currency: measure.currency ?? undefined, description: measure.description }}
      onFinish={(values) => {
        if (saving) return;
        onSave({ label: values.label, expression: values.expression, kind: values.kind,
          currency: values.kind === "currency" ? (values.currency || null) : null, description: values.description ?? "" });
      }}>
      <Form.Item label="Label" name="label" rules={[{ required: true }, { max: MAX_LABEL }]}>
        <Input maxLength={MAX_LABEL} />
      </Form.Item>
      <Form.Item label="Expression" name="expression" rules={[{ required: true }, { max: MAX_EXPRESSION }]}>
        <Input.TextArea autoSize maxLength={MAX_EXPRESSION} />
      </Form.Item>
      <Form.Item label="Kind" name="kind"><Select options={KINDS.map((k) => ({ value: k, label: k }))} /></Form.Item>
      {kind === "currency" ? (
        <Form.Item label="Currency" name="currency" rules={[{ pattern: /^[A-Z]{3}$/, message: "Three capital letters" }]}>
          <Input maxLength={3} />
        </Form.Item>
      ) : null}
      <Form.Item label="Description" name="description" rules={[{ max: MAX_DESCRIPTION }]}>
        <Input.TextArea autoSize maxLength={MAX_DESCRIPTION} />
      </Form.Item>
      <Button type="primary" htmlType="submit" loading={saving} disabled={false}>Save changes</Button>
    </Form>
  );
}

/** What an action did, its charts named by title: for a Remove, the charts deleted and those taken off this dashboard
 * because the person does not edit them. */
function ResultView({ result }: { result: Result }) {
  if ("refused" in result) {
    return <div role="status" aria-label="Result"><Alert type="error" showIcon message={result.refused} /></div>;
  }
  const { action, answer, charts } = result;
  const named = (ids: number[] = []) =>
    ids.map((id) => chartName(charts.find((chart) => chart.chart_id === id) ?? { chart_id: id, title: null, check: null }))
      .join(", ");
  const lines: string[] = [];
  if (answer.changed?.length) lines.push(`${action === "remove" ? "Deleted" : "Changed"}: ${named(answer.changed)}`);
  if (answer.left_by_hand?.length) lines.push(`Left as changed by hand: ${named(answer.left_by_hand)}`);
  for (const entry of answer.not_editor ?? []) {
    const editors = (entry.editors ?? []).join(", ") || "none named";
    if (entry.chart_id === undefined) {
      lines.push(`You do not edit this measure's dataset; its editors: ${editors}`);
    } else if (action === "remove") {
      lines.push(`Taken off this dashboard (you do not edit it): ${named([entry.chart_id])}; its editors: ${editors}`);
    } else {
      lines.push(`You do not edit ${named([entry.chart_id])}; its editors: ${editors}`);
    }
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
