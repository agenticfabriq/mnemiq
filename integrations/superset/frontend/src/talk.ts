// The conversation on screen, driven outside the panel's component: rc2 unmounts the panel on every close, and a
// message in flight, or a build started from a confirm, must outlive that. mnemiq keeps the conversation; the panel
// holds only what is on screen (`store.ts`): the kept messages as last read, and the turn of the message in flight,
// read from its stream by the workbench's own reducer while it streams.
import type { Answer, PanelApi } from "./api";
import { driver } from "./build";
import {
  confirmMessage,
  declineMessage,
  isChangePreview,
  readConversation,
  sendMessage,
  startConversation,
  type Choose,
  type ConfirmBody,
  type Message,
  type Page,
} from "./conversation";
import { closeRun, reduce, userTurn, type AguiEvent } from "./shared";
import { getSnapshot, update } from "./store";

let sending: AbortController | null = null;

/** Reads conversation `id` from mnemiq onto the screen; answers the refusal's sentence when it could not be read. */
export async function openConversation(api: PanelApi, id: string): Promise<string | null> {
  const found = await readConversation(api, id);
  if (!found.ok) {
    update((state) => ({ ...state, notice: found.refusal.sentence }));
    return found.refusal.sentence;
  }
  const { conversation } = found.body;
  update((state) => ({ ...state, conversationId: conversation.id, title: conversation.title,
    messages: conversation.messages, turns: state.sending ? state.turns : [], notice: null }));
  return null;
}

/** A new conversation: the screen cleared, and the conversation made by mnemiq when its first message is sent. */
export function newConversation(): boolean {
  if (sending !== null) return false;
  update((state) => ({ ...state, conversationId: null, title: "", messages: [], turns: [], notice: null }));
  return true;
}

/** Sends `text` from `page` in the conversation on screen (a new one when none is), unless a message is already in
 * flight: one at a time. Answers whether it was sent. */
export function sendInPanel(api: PanelApi, text: string, page: Page, choose?: Choose): boolean {
  if (sending !== null) return false;
  const controller = new AbortController();
  sending = controller;
  void send(api, text, page, choose, controller.signal).finally(() => {
    if (sending === controller) sending = null;
    update((state) => ({ ...state, sending: false }));
  });
  return true;
}

async function send(api: PanelApi, text: string, page: Page, choose: Choose | undefined, signal: AbortSignal) {
  update((state) => ({ ...state, sending: true, notice: null, turns: [userTurn(text)] }));
  const read = (event: AguiEvent) => update((state) => ({ ...state, turns: reduce(state.turns, event) }));
  let id = getSnapshot().conversationId;
  if (id === null) {
    const started = await startConversation(api);
    if (!started.ok) {
      read({ type: "RUN_STARTED", runId: "" });
      read({ type: "RUN_ERROR", message: started.refusal.sentence, code: started.refusal.kind });
      return;
    }
    id = started.body.conversation.id;
    update((state) => ({ ...state, conversationId: id }));
  }
  const before = getSnapshot().messages.length;
  let sawEnd = false;
  try {
    for await (const event of sendMessage(api, id, { text, page, choose }, signal)) {
      if (event.type === "RUN_FINISHED" || event.type === "RUN_ERROR") sawEnd = true;
      read(event);
    }
  } catch (error) {
    if ((error as { name?: string })?.name !== "AbortError") console.error("mnemiq panel: a message's stream broke", error);
  } finally {
    update((state) => ({ ...state, turns: closeRun(state.turns, sawEnd) }));
  }
  // What mnemiq kept is what the conversation shows from here. A message mnemiq kept nothing of (refused before
  // its stream started) keeps its turn on screen, saying why.
  await openConversation(api, id);
  if (getSnapshot().messages.length > before) update((state) => ({ ...state, turns: [] }));
}

/** The proposal at `index` answered: "Build it", "Apply", or an offered dashboard's choice, from the dashboard on
 * screen. A build confirmed is run here, one step per request, as any build is. */
export async function confirmInPanel(api: PanelApi, message: Message, body: ConfirmBody): Promise<Answer<unknown>> {
  const id = getSnapshot().conversationId;
  if (id === null) throw new Error("no conversation is on screen");
  const answer = await confirmMessage(api, id, message.index, body);
  const built = answer.ok ? (answer.body as { build?: { build_id?: unknown } })?.build : undefined;
  if (built && typeof built.build_id === "string" && message.proposal && !isChangePreview(message.proposal)) {
    const dashboardId = message.proposal.dashboard_id;
    driver.resume(message.proposal.goal, dashboardId !== null ? { dashboardId } : {}, built.build_id);
  }
  update((state) => ({ ...state, notice: answer.ok ? null : answer.refusal.sentence }));
  await openConversation(api, id);
  if (!answer.ok) update((state) => ({ ...state, notice: answer.refusal.sentence }));
  return answer;
}

/** The proposal at `index` set aside: the person's "Not now". */
export async function declineInPanel(api: PanelApi, message: Message): Promise<void> {
  const id = getSnapshot().conversationId;
  if (id === null) return;
  const answer = await declineMessage(api, id, message.index);
  await openConversation(api, id);
  if (!answer.ok) update((state) => ({ ...state, notice: answer.refusal.sentence }));
}

/** For tests: nothing in flight. */
export function resetTalk(): void {
  sending?.abort();
  sending = null;
}
