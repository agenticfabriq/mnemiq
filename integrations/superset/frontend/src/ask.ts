// Asking about the data: one question streamed from mnemiq's `/v1/chat` through the forwarder, read into the
// conversation's turns by the workbench's own reducer, and ended, as the workbench ends one, even when the stream
// stops without its last frame.
import type { PanelApi } from "./api";
import { closeRun, DEFAULT_MODE, historyFrom, isRunning, reduce, userTurn, type Turn } from "./shared";
import { getSnapshot, update } from "./store";

/** Asks `question`, with the history of `prior` (each earlier answer's question, SQL and columns, never its rows),
 * writing each turn as it changes through `update`. */
export async function askQuestion(
  api: PanelApi,
  question: string,
  prior: Turn[],
  update: (fn: (turns: Turn[]) => Turn[]) => void,
  signal: AbortSignal,
): Promise<void> {
  const history = historyFrom(prior).map((turn) => ({ ...turn, rows: [] }));
  update((turns) => [...turns, userTurn(question)]);
  let sawEnd = false;
  try {
    for await (const event of api.stream("/chat", { question, mode: DEFAULT_MODE, history }, signal)) {
      if (event.type === "RUN_FINISHED" || event.type === "RUN_ERROR") sawEnd = true;
      update((turns) => reduce(turns, event));
    }
  } catch (error) {
    // A stream that broke, or was stopped: the answer in flight ends as an error below, never left running.
    if ((error as { name?: string })?.name !== "AbortError") console.error("mnemiq panel: an answer's stream broke", error);
  } finally {
    update((turns) => closeRun(turns, sawEnd));
  }
}

let asking: AbortController | null = null;

/** Asks `question` in the panel's conversation, unless an ask is already running there: one at a time, as the
 * workbench asks. It runs outside the panel's component, so closing the panel (which rc2 unmounts) mid-answer leaves
 * the answer to arrive in the store for the next opening. Answers whether the ask was sent. */
export function askInPanel(api: PanelApi, question: string): boolean {
  if (asking !== null || isRunning(getSnapshot().turns)) return false;
  const controller = new AbortController();
  asking = controller;
  void askQuestion(api, question, getSnapshot().turns,
    (fn) => update((state) => ({ ...state, turns: fn(state.turns) })), controller.signal)
    .finally(() => {
      if (asking === controller) asking = null;
    });
  return true;
}

/** For tests: no ask running. */
export function resetAsking(): void {
  asking?.abort();
  asking = null;
}
