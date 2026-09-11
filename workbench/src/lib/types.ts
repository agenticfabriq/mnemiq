/** The wire contract. Mirrors `answer_payload` in src/mnemiq/server/serialize.py. */

export const MODES = ["instant", "thinking", "deep"] as const;
export type Mode = (typeof MODES)[number];
export const DEFAULT_MODE: Mode = "thinking";

/**
 * Why there is no answer, categorised by what the caller should do next.
 * Mirrors `DeferralReason` in src/mnemiq/contract/seams.py.
 *
 * The last three are deliberately in this list and are NOT deferrals: they ride with
 * `failed: true` and `deferred: false`, and the UI must keep them apart. Each names a
 * different thing that broke -- the source, the model provider, the verifier -- so the card
 * has to say which; "Source failed" over a verifier outage is a false statement.
 */
export type DeferralReason =
  | "authorization"
  | "policy_unavailable"
  | "no_tables"
  | "unanswerable"
  | "undefined_term"
  | "invalid_query"
  | "verification"
  | "disagreement"
  | "execution_failed"
  | "model_unavailable"
  | "verifier_unavailable"
  | "ungovernable";

/** Postgres numerics arrive as strings; nulls stay null rather than becoming "". */
export type Cell = string | number | boolean | null;

export type ResultPreview = {
  columns: string[];
  rows: Cell[][];
  /** True count in the result, not the capped length of `rows`. */
  row_count: number;
  truncated: boolean;
};

/** One prior turn, echoed back so a follow-up can resolve against it. */
export type HistoryTurn = {
  question: string;
  sql: string;
  tables_used: string[];
  columns: string[];
  rows: Cell[][];
  /** The authorization boundary the engine answered under; it re-checks this. */
  grant_fingerprint: string;
};

export type AnswerPayload = {
  answer: string;
  deferred: boolean;
  failed: boolean;
  reason_code: DeferralReason | null;
  mode: string | null;
  cached: boolean;
  agreement: number | null;
  judge_engaged: boolean | null;
  judge_override: boolean | null;
  /** Whether the engaged judge FELL BACK to the majority vote instead of picking. The selector
   *  fails closed on an outage, an unreadable reply or a pick outside the clusters, and the
   *  fallback returns the same index a judge agreeing with the majority returns -- so without
   *  this, `judge_engaged` alone claims a judgement that may never have happened. `null` means
   *  the engine did not report it, which is not evidence of a fallback. */
  judge_fell_back: boolean | null;
  candidates_executed: number | null;
  /** What the mode spent: outer attempts (the database rejected the SQL) and whether the
   *  corrector's surgical pass carried it (the decider rejected it). Two different judges. */
  attempts: number | null;
  corrected: boolean | null;
  sql: string | null;
  tables_used: string[] | null;
  enrichment_version: string | null;
  timing: Record<string, number> | null;
  preview: ResultPreview | null;
  grant_fingerprint?: string;
};

/**
 * One phase of an answer. `id` is what a finish is matched to its start by -- matching
 * on name would break as soon as two steps of the same name overlap.
 */
export type Step = {
  id: string;
  name: string;
  status: "running" | "done" | "failed";
  ms?: number;
  /** plan: which repair attempt. candidate: which of N. `of` is the total for either. */
  attempt?: number;
  index?: number;
  of?: number;
};

/** The AG-UI frames /v1/chat emits. Unlisted types are ignored, not errors. */
export type AguiEvent =
  | { type: "RUN_STARTED"; runId: string }
  // Stage-specific extras (attempt/index/of) ride along untyped and are read
  // defensively, so a new field on the engine side cannot break an older client.
  | { type: "STEP_STARTED"; stepName: string; stepId: string }
  | { type: "STEP_FINISHED"; stepName: string; stepId: string; durationMs?: number;
      ok?: boolean }
  | { type: "TEXT_MESSAGE_START"; messageId: string; role: string }
  | { type: "TEXT_MESSAGE_CONTENT"; messageId: string; delta: string }
  | { type: "TEXT_MESSAGE_END"; messageId: string }
  | { type: "CUSTOM"; name: string; value: unknown }
  | { type: "RUN_FINISHED"; runId: string }
  | { type: "RUN_ERROR"; message: string; runId?: string }
  | { type: string & Record<never, never> };

export type SchemaTable = { object_id: string; card: string };
