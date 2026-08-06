/**
 * The turns a follow-up is asked against.
 *
 * Structure, not prose: the previous query, the tables it read, and a few rows of its
 * result. The rows are the part that matters -- "and how many claims does it have?"
 * resolves "it" against the *result*, because the previous SQL computed the top region
 * without ever naming it.
 *
 * Each turn carries the `grant_fingerprint` the engine stamped on the answer. The engine
 * replays a turn only to that same authorization boundary and re-checks it every time;
 * this is a faithful echo, never a claim about what the caller may see.
 */

import type { Turn } from "./store";
import type { HistoryTurn } from "./types";

// The server bounds these again. Keeping the request small is the client's reason.
const MAX_TURNS = 3;
const MAX_ROWS = 5;

/** The completed, answered turns before the current one, oldest first. */
export function historyFrom(turns: Turn[]): HistoryTurn[] {
  const out: HistoryTurn[] = [];
  for (const turn of turns) {
    const answer = turn.answer;
    // A deferral or a failure has no result to resolve a pronoun against, and its
    // question was never answered -- carrying it forward would only add noise.
    if (turn.role !== "assistant" || !answer || answer.deferred || answer.failed) continue;

    const preview = answer.preview;
    out.push({
      question: questionFor(turns, turn),
      sql: answer.sql ?? "",
      tables_used: answer.tables_used ?? [],
      columns: preview?.columns ?? [],
      rows: (preview?.rows ?? []).slice(0, MAX_ROWS),
      grant_fingerprint: answer.grant_fingerprint ?? "",
    });
  }
  return out.slice(-MAX_TURNS);
}

/** The user message this answer replied to -- the words a pronoun refers back to. */
function questionFor(turns: Turn[], answer: Turn): string {
  const at = turns.indexOf(answer);
  for (let i = at - 1; i >= 0; i--) {
    if (turns[i]!.role === "user") return turns[i]!.text;
  }
  return "";
}
