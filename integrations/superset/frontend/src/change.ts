// Changing the dashboard on screen by chat (PR D's `POST v1/superset/dashboards/{id}/changes`): mnemiq reads the
// request against the dashboard's charts and makes each change as the person, leaves what it should not change with
// its reason, proposes any removal for the person to confirm, and plans a chart to add as a build, which the panel
// runs. The dashboard is the one on screen when the request is sent.
import type { Answer, PanelApi } from "./api";
import { driver, type StartAnswer } from "./build";
import { update } from "./store";

export type ChangeView = {
  changed: number[];
  left: { chart_id: number; reason: string }[];
  dropped: { chart_id: number; key: string }[];
  not_editor: { chart_id: number; editors: string[] }[];
  removed: number[];
  would_remove: { chart_id: number; title: string | null }[];
  added: StartAnswer | { error: { status: number; detail: unknown } } | null;
  message: string;
};

/** The change the panel shows: the request, the dashboard it went to, mnemiq's answer or the refusal, and whether it
 * is being sent. */
export type ChangeRecord = {
  request: string;
  dashboardId: number | null;
  answer: ChangeView | null;
  refused: string | null;
  note: string | null;
  sending: boolean;
};

export const NOT_ON_A_DASHBOARD = "Change works on a dashboard you can read: open one, then ask again.";
const UNASKED = "mnemiq could not be asked to change the dashboard; try again shortly.";

const list = <T,>(value: unknown): T[] => (Array.isArray(value) ? (value as T[]) : []);

/** mnemiq's answer read defensively: a list it leaves out is none, and its message is text or nothing. */
export function changeView(body: unknown): ChangeView {
  const found = (body ?? {}) as Record<string, unknown>;
  return {
    changed: list<number>(found.changed),
    left: list(found.left),
    dropped: list(found.dropped),
    not_editor: list(found.not_editor),
    removed: list<number>(found.removed),
    would_remove: list(found.would_remove),
    added: (found.added as ChangeView["added"]) ?? null,
    message: typeof found.message === "string" ? found.message : "",
  };
}

export function sendChange(api: PanelApi, request: string, dashboardId: number): Promise<Answer<unknown>> {
  return api.post(`/superset/dashboards/${dashboardId}/changes`, { request });
}

/** Removes the charts the person confirmed from a change's proposal, and nothing else. */
export function confirmRemoval(api: PanelApi, dashboardId: number, chartIds: number[]): Promise<Answer<unknown>> {
  return api.post(`/superset/dashboards/${dashboardId}/changes`, { remove_chart_ids: chartIds });
}

let sending = false;

function write(fn: (record: ChangeRecord) => ChangeRecord): void {
  update((state) => (state.change ? { ...state, change: fn(state.change) } : state));
}

/** Sends `request` to the dashboard `dashboardOnScreen` names when called (none: nothing is sent), one change at a
 * time, outside the panel's component; a chart the change adds is run as a build. Answers whether it was sent. */
export function changeInPanel(api: PanelApi, request: string,
  dashboardOnScreen: () => Promise<number | null>): boolean {
  if (sending) return false;
  sending = true;
  update((state) => ({ ...state, change: { request, dashboardId: null, answer: null, refused: null, note: null,
    sending: true } }));
  void (async () => {
    try {
      const dashboardId = await dashboardOnScreen();
      if (dashboardId === null) return write((record) => ({ ...record, refused: NOT_ON_A_DASHBOARD }));
      write((record) => ({ ...record, dashboardId }));
      settle(await sendChange(api, request, dashboardId), request, dashboardId);
    } catch {
      write((record) => ({ ...record, refused: UNASKED }));
    } finally {
      sending = false;
      write((record) => ({ ...record, sending: false }));
    }
  })();
  return true;
}

/** Confirms the proposed removal of `chartIds` from the change on screen. */
export function confirmInPanel(api: PanelApi, record: ChangeRecord, chartIds: number[]): boolean {
  if (sending || record.dashboardId === null) return false;
  const dashboardId = record.dashboardId;
  sending = true;
  write((current) => ({ ...current, sending: true }));
  void confirmRemoval(api, dashboardId, chartIds)
    .then((answer) => settle(answer, record.request, dashboardId))
    .catch(() => write((current) => ({ ...current, refused: UNASKED })))
    .finally(() => {
      sending = false;
      write((current) => ({ ...current, sending: false }));
    });
  return true;
}

/** Keeps the charts a change proposed to remove: the proposal is dropped, and nothing is sent. */
export function keepProposed(): void {
  write((record) => (record.answer ? { ...record, answer: { ...record.answer, would_remove: [] } } : record));
}

function settle(answer: Answer<unknown>, request: string, dashboardId: number): void {
  if (!answer.ok) return write((record) => ({ ...record, refused: answer.refusal.sentence }));
  const view = changeView(answer.body);
  let note: string | null = null;
  const added = view.added;
  if (added && "build" in added && added.build?.status === "building") {
    // The chart the change adds is a build onto this dashboard: run it, unless one is already running here, in which
    // case it waits, resumable, among the person's unfinished builds.
    if (!driver.resume(request, { dashboardId }, added.build.build_id)) {
      note = "Another build is running here; the chart's build waits among your unfinished builds.";
    }
  }
  write((record) => ({ ...record, answer: view, refused: null, note }));
}

/** For tests: no change being sent. */
export function resetChanges(): void {
  sending = false;
}
