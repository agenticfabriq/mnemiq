// The dashboard's measures, as mnemiq's measures routes answer them for the person asking: each measure's
// definition, its value computed now under the person's own rules, and the charts that hold a copy of it. Values are
// read each time the section opens and kept only in its component's state, never in the panel's store.
import type { Answer, PanelApi } from "./api";

/** mnemiq's `MeasureKind` values. */
export type MeasureKind = "currency" | "percent" | "count" | "number";

export type ChartOfMeasure = { chart_id: number; title: string | null; check: string | null };

export type MeasureItem = {
  name: string;
  label: string;
  expression: string;
  kind: MeasureKind;
  currency: string | null;
  description: string;
  asked_by: string;
  dataset_id: number;
  /** The saved metric the measure is, when it is one (a metric Superset keeps on the dataset), else null. */
  saved_metric: string | null;
  value: number | null;
  value_note: string | null;
  /** What `value` covers (mnemiq's `ValueBasis`): "readable_rows", every row the person can read before any chart's
   * filters, or "chart", the one number the chart `value_chart_id` shows under that chart's own filters; null with no
   * value. An older mnemiq answers neither. */
  value_basis?: "readable_rows" | "chart" | null;
  value_chart_id?: number | null;
  charts: ChartOfMeasure[];
};

export type Editors = { chart_id?: number; dataset_id?: number; editors: string[] };

/** What an Edit, a Remove or a Save did: the charts changed (for a Remove, deleted), those left as changed by hand,
 * those the person does not edit (with who does; a Remove takes them off this dashboard instead), those a report
 * holds, those that could not be read, and mnemiq's words. */
export type ActionAnswer = {
  changed: number[];
  left_by_hand: number[];
  not_editor: Editors[];
  blocked_by_report: number[];
  could_not_read: number[];
  message: string;
};

/** A remove refused because the measure's charts are no longer the ones shown: mnemiq's words and the charts now. */
export type ChartsChanged = { chartsChanged: { message: string; charts: ChartOfMeasure[] } };

export type MeasureEdit = { label: string; expression: string; kind: MeasureKind; currency: string | null;
  description: string };

/** A chart as the measures section names it: its title, else its id. */
export const chartName = (chart: ChartOfMeasure) => chart.title ?? `chart ${chart.chart_id}`;

/** What a listed value covers, as one sentence, or null where there is no value or mnemiq does not say. mnemiq's own
 * statement covers every row the person can read, before any chart's filters, so a measure whose condition sits on
 * its charts' filters reads differently here than on them; a chart's one number is under that chart's own filters.
 * An older mnemiq answers no `value_basis`, and then nothing is said rather than a guess. */
export function valueBasisText(measure: MeasureItem): string | null {
  if (measure.value === null) return null;
  if (measure.value_basis === "readable_rows") return "Over every row you can read, before any chart's filters.";
  if (measure.value_basis === "chart" && typeof measure.value_chart_id === "number") {
    const id = measure.value_chart_id;
    const chart = measure.charts.find((c) => c.chart_id === id) ?? { chart_id: id, title: null, check: null };
    return `As ${chartName(chart)} shows it, with that chart's filters.`;
  }
  return null;
}

const path = (dashboardId: number, name?: string, action?: string) =>
  `/superset/dashboards/${dashboardId}/measures${name === undefined ? "" : `/${encodeURIComponent(name)}/${action}`}`;

/** `value` as `kind` reads, in `locale`: currency with its code and 2 decimals (2 decimals when it names none), a
 * percent (a fraction, as mnemiq keeps it) with 1 decimal, a count as a grouped whole number, any other number with 2
 * decimals; no value is "no value", never 0. */
export function measureValueText(value: number | null, kind: MeasureKind, currency: string | null,
  locale?: string): string {
  if (value === null || !Number.isFinite(value)) return "no value";
  const format = (options: Intl.NumberFormatOptions) => new Intl.NumberFormat(locale, options).format(value);
  if (kind === "currency" && currency) {
    try {
      return format({ style: "currency", currency, minimumFractionDigits: 2, maximumFractionDigits: 2 });
    } catch {
      // A code Intl does not know: the number, as for no code.
    }
  }
  if (kind === "percent") return format({ style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 });
  if (kind === "count") return format({ maximumFractionDigits: 0 });
  return format({ minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function listMeasures(api: PanelApi, dashboardId: number, signal?: AbortSignal): Promise<Answer<{ measures: MeasureItem[] }>> {
  return api.get(path(dashboardId), signal);
}

export function editMeasure(api: PanelApi, dashboardId: number, name: string, edit: MeasureEdit): Promise<Answer<ActionAnswer>> {
  return api.post(path(dashboardId, name, "edit"), edit);
}

export function saveMeasure(api: PanelApi, dashboardId: number, name: string): Promise<Answer<ActionAnswer>> {
  return api.post(path(dashboardId, name, "save"), {});
}

/** Removes the measure from the charts the person was shown, `chartIds`; when they are no longer its charts, mnemiq
 * refuses (409) with the charts now, answered as `chartsChanged`. */
export async function removeMeasure(api: PanelApi, dashboardId: number, name: string,
  chartIds: number[]): Promise<Answer<ActionAnswer> | ChartsChanged> {
  const answer = await api.post<ActionAnswer>(path(dashboardId, name, "remove"), { chart_ids: chartIds });
  if (!answer.ok && answer.status === 409) {
    const detail = (answer.body as { detail?: { message?: unknown; charts?: unknown } } | undefined)?.detail;
    if (detail && typeof detail === "object" && Array.isArray(detail.charts)) {
      return { chartsChanged: { message: typeof detail.message === "string" ? detail.message : "",
        charts: detail.charts as ChartOfMeasure[] } };
    }
  }
  return answer;
}
