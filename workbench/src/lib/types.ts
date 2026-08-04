/** The wire contract. Mirrors `answer_payload` in src/mnemiq/server/serialize.py. */

export const MODES = ["instant", "thinking", "deep"] as const;
export type Mode = (typeof MODES)[number];
export const DEFAULT_MODE: Mode = "thinking";

/**
 * Why there is no answer, categorised by what the caller should do next.
 * Mirrors `DeferralReason` in src/mnemiq/contract/seams.py.
 *
 * `execution_failed` is deliberately in this list but is NOT a deferral: it rides
 * with `failed: true` and `deferred: false`, and the UI must keep them apart.
 */
export type DeferralReason =
  | "authorization"
  | "policy_unavailable"
  | "no_tables"
  | "unanswerable"
  | "invalid_query"
  | "verification"
  | "disagreement"
  | "execution_failed";

/** Postgres numerics arrive as strings; nulls stay null rather than becoming "". */
export type Cell = string | number | boolean | null;

export type ResultPreview = {
  columns: string[];
  rows: Cell[][];
  /** True count in the result, not the capped length of `rows`. */
  row_count: number;
  truncated: boolean;
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
  candidates_executed: number | null;
  sql: string | null;
  tables_used: string[] | null;
  enrichment_version: string | null;
  timing: Record<string, number> | null;
  preview: ResultPreview | null;
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
