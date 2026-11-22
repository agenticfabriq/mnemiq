// The panel's conversations with mnemiq, through the forwarder (`v1/superset/conversations`): each person's own, kept
// by mnemiq, so they are there after a reload, in another tab and on any dashboard. The panel keeps the conversation on
// screen in module memory only; mnemiq is where it lives.
import type { Answer, PanelApi } from "./api";
import type { PlanSummary } from "./build";
import type { AguiEvent } from "./shared";

export type ConversationSummary = { id: string; title: string; updated_at: string };

export type DashboardOf = { id: number; title: string | null };

/** A build waiting for "Build it", as mnemiq keeps it. */
export type BuildProposal = {
  build_id: string;
  goal: string;
  dashboard_id: number | null;
  plan: PlanSummary | null;
  left_out?: { part: string; chart_id: number; title: string | null }[];
};

/** A change's preview, waiting for "Apply" (`ChangePreview.to_json` in mnemiq's superset/chart_change.py). */
export type ChangePreview = {
  dashboard_id: number;
  dashboard_changed_on: string | null;
  edits: { chart_id: number; title: string | null; changed_on: string | null; said: string; digest: string }[];
  would_remove: { chart_id: number; title: string | null; changed_on?: string | null }[];
  add_chart: boolean;
  left: { chart_id: number; reason: string }[];
  dropped: { chart_id: number; key: string }[];
  not_editor: { chart_id: number | null; editors: string[] }[];
  message: string;
  reason: string | null;
};

/** What came of a proposal: `proposed` until the person answers it, then what their answer did. */
export type Outcome = {
  state: string;
  build_id?: string;
  changed?: number[];
  left?: { chart_id: number; reason: string }[];
  removed?: number[];
};

/** One message of a conversation, as mnemiq keeps it: never a row of results. */
export type Message = {
  index: number;
  role: "person" | "mnemiq";
  kind: "ask" | "answer" | "build" | "change" | "clarify" | "note";
  text: string;
  at?: string;
  dashboard?: DashboardOf | null;
  intent?: string;
  goal?: string;
  /** mnemiq's message: the person's message it replies to. */
  replies_to?: number;
  // An answer mnemiq declined to give (`deferred`) or could not give (`failed`) keeps which, and its reason's code.
  answer?: { sentence: string; sql: string; grant_fingerprint: string | null; tables_used: string[];
    deferred?: boolean; failed?: boolean; reason_code?: string | null };
  build_id?: string;
  proposal?: BuildProposal | ChangePreview;
  offer?: { dashboard_id: number; title: string | null; reason: string | null };
  refused?: { reason: string; parts: { part: string; reason: string }[] };
  shown?: { dashboard_id: number; left_out: { part: string; chart_id: number; title: string | null }[] };
  clarify?: { question: string; choices: string[] };
  outcome?: Outcome;
};

export type Conversation = { id: string; title: string; created_at: string; updated_at: string; messages: Message[] };

/** The page a message is sent from: the dashboard on screen, as Superset answered it to the person, or none. */
export type Page = { dashboardId: number | null; dashboardTitle: string | null; dashboardHasCharts: boolean | null };

export type Choose = "ask" | "build" | "change";

export type ConfirmBody = { dashboard_id?: number | null; remove_chart_ids?: number[]; choice?: "add" | "new" };

/** A kept answer's table, run again as the person now, or the sentence that says why it was not. */
export type Results =
  | { columns: string[]; rows: (string | number | boolean | null)[][]; row_count: number; truncated: boolean;
      capped: boolean }
  | { refused: string };

const BASE = "/superset/conversations";
const at = (id: string) => `${BASE}/${encodeURIComponent(id)}`;

export function isChangePreview(proposal: Message["proposal"]): proposal is ChangePreview {
  return !!proposal && Array.isArray((proposal as ChangePreview).edits);
}

/** The person's conversations, newest first; `before` is the `next` of the page before. */
export function listConversations(api: PanelApi, before?: string | null):
  Promise<Answer<{ conversations: ConversationSummary[]; next: string | null }>> {
  return api.get(before ? `${BASE}?before=${encodeURIComponent(before)}` : BASE);
}

export function startConversation(api: PanelApi): Promise<Answer<{ conversation: Conversation }>> {
  return api.post(BASE, {});
}

export function readConversation(api: PanelApi, id: string): Promise<Answer<{ conversation: Conversation }>> {
  return api.get(at(id));
}

export function deleteConversation(api: PanelApi, id: string): Promise<Answer<null>> {
  return api.post(`${at(id)}/delete`, {});
}

/** A message, answered as a stream: what it was routed to, then an answer's frames, a proposal, a preview, or a
 * question back. `choose` is the person's own answer to a question back. */
export function sendMessage(api: PanelApi, id: string, message: { text: string; page: Page; choose?: Choose },
  signal?: AbortSignal): AsyncGenerator<AguiEvent> {
  const { text, page, choose } = message;
  const body = {
    text,
    ...(page.dashboardId !== null ? { dashboard_id: page.dashboardId, dashboard_title: page.dashboardTitle,
      dashboard_has_charts: page.dashboardHasCharts } : {}),
    ...(choose ? { choose } : {}),
  };
  return api.stream(`${at(id)}/messages`, body, signal);
}

/** The person's "Build it", "Apply" or answer to an offer, sent with the dashboard on screen. */
export function confirmMessage(api: PanelApi, id: string, index: number, body: ConfirmBody): Promise<Answer<unknown>> {
  return api.post(`${at(id)}/messages/${index}/confirm`, body);
}

/** The person's "Not now". */
export function declineMessage(api: PanelApi, id: string, index: number): Promise<Answer<{ state: string }>> {
  return api.post(`${at(id)}/messages/${index}/decline`, {});
}

/** A kept answer's table, run again as the person, now. */
export function showResults(api: PanelApi, id: string, index: number): Promise<Answer<Results>> {
  return api.post(`${at(id)}/messages/${index}/results`, {});
}
