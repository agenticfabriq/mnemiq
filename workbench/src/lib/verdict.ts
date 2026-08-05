/**
 * What the engine decided, and what the reader should do about it.
 *
 * The copy for each reason comes from what the caller is expected to do next --
 * the axis `DeferralReason` is categorised on. A refusal is a verdict with an
 * action attached, not an apology.
 */

import type { Turn } from "./store";
import type { DeferralReason } from "./types";

export type Disposition = "answered" | "declined" | "failed" | "working" | "broken";

export function disposition(turn: Turn): Disposition {
  if (turn.status === "error") return "broken";
  if (!turn.answer) return turn.status === "running" ? "working" : "answered";
  if (turn.answer.failed) return "failed";
  if (turn.answer.deferred) return "declined";
  return "answered";
}

export const DISPOSITION_LABEL: Record<Disposition, string> = {
  answered: "Answered",
  declined: "Declined",
  failed: "Source failed",
  working: "Working",
  broken: "Stream failed",
};

export const REASONS: Record<DeferralReason, { title: string; next: string }> = {
  authorization: {
    title: "Out of scope for this identity",
    next: "Request a grant for the tables this question needs.",
  },
  policy_unavailable: {
    title: "The access policy could not be read",
    next: "Page an operator. The engine refuses rather than guess at access.",
  },
  no_tables: {
    title: "Nothing retrieved for this question",
    next: "Rephrase it, or check that enrichment has run over this source.",
  },
  unanswerable: {
    title: "Not answerable from the tables in scope",
    next: "Ask about a column the data actually holds.",
  },
  invalid_query: {
    title: "No valid query within the budget",
    next: "Narrow the question, or retry in a higher mode.",
  },
  verification: {
    title: "Withdrawn by the verifier",
    next: "The engine answered, then declined to stand behind it. Narrow the question.",
  },
  disagreement: {
    title: "Candidates disagreed",
    next: "Too much divergence to pick one answer. Retry, or narrow the question.",
  },
  execution_failed: {
    title: "The source rejected every attempt",
    next: "Page an operator. This is a source failure, not a refusal.",
  },
  model_unavailable: {
    title: "The model provider did not respond",
    next: "An outage, not a judgement about your data. Try again.",
  },
};

/** 412 ms / 8.5 s / 2 m 04 s -- always legible, never more precision than is useful. */
export function duration(ms: number | undefined): string | null {
  if (ms === undefined || Number.isNaN(ms)) return null;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const minutes = Math.floor(ms / 60_000);
  return `${minutes} m ${String(Math.round((ms % 60_000) / 1000)).padStart(2, "0")} s`;
}

export const plural = (n: number, one: string, many = `${one}s`) =>
  `${n} ${n === 1 ? one : many}`;
