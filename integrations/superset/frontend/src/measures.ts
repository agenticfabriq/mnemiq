// The dashboard's measures, as mnemiq answers them for the person asking (PR D's measures routes): each measure's
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
  saved_metric: boolean;
  value: number | null;
  value_note: string | null;
  charts: ChartOfMeasure[];
};

export type Editors = { chart_id?: number; dataset_id?: number; editors: string[] };

/** What an Edit, a Remove or a Save did: the charts changed, those left as changed by hand, those the person does not
 * edit (with who does), those a report holds, those that could not be read, and mnemiq's words. */
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
