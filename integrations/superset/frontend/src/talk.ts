// The conversation on screen, driven outside the panel's component: rc2 unmounts the panel on every close, and a
// message in flight, or a build started from a confirm, must outlive that. mnemiq keeps the conversation; the panel
// holds what is on screen (`store.ts`): the kept messages as last read, the turn of the message in flight, read from
// its stream by the workbench's own reducer while it streams, and the answer just given, with its rows, until the
// person moves on.
//
// One thing happens on the screen at a time: a message, a confirm or a decline holds it (`busy`), and while it does
// the person cannot switch conversations or start a new one. Every read of a conversation is bound to the screen it
// was asked for (`screen`), so one that answers after the person moved on changes nothing.
import type { Answer, PanelApi } from "./api";
import { ALREADY_BUILDING, driver } from "./build";
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
import { closeRun, reduce, userTurn, type AguiEvent, type Turn } from "./shared";
import { getSnapshot, update } from "./store";

// mnemiq's sentence for a conversation that is not, or no longer, the person's (deleted in another tab).
export const NOT_YOURS = "This person has no conversation of that id.";
export const OPEN_IT = "Open the dashboard this was for, then confirm.";

let holding: AbortController | null = null;
// Bumped each time the person moves to another conversation, or a new one: what a late read is checked against.
let screen = 0;

function hold(): AbortController | null {
  if (holding !== null) return null;
  holding = new AbortController();
  update((state) => ({ ...state, sending: true }));
  return holding;
}

function release(controller: AbortController): void {
  if (holding === controller) holding = null;
  update((state) => ({ ...state, sending: false }));
}

/** Reads conversation `id` onto the screen, when the screen is still `on` (the screen it was asked for). A conversation
 * mnemiq no longer has for this person clears the screen. Answers the refusal's sentence when it could not be read. */
async function reread(api: PanelApi, id: string, on: number): Promise<string | null> {
  const found = await readConversation(api, id);
  if (screen !== on) return null;
  if (!found.ok) {
    if (found.refusal.sentence === NOT_YOURS) {
      update((state) => ({ ...state, conversationId: null, title: "", messages: [], turns: [], fresh: null,
        notice: NOT_YOURS }));
    } else {
      update((state) => ({ ...state, notice: found.refusal.sentence }));
    }
    return found.refusal.sentence;
  }
  const { conversation } = found.body;
  update((state) => ({ ...state, conversationId: conversation.id, title: conversation.title,
    messages: conversation.messages, notice: null }));
  return null;
}

/** Opens conversation `id` from the list, unless something is in flight on the screen. */
export async function openConversation(api: PanelApi, id: string): Promise<string | null> {
  if (holding !== null) return null;
  const on = ++screen;
  update((state) => ({ ...state, turns: [], fresh: null, notice: null }));
  return reread(api, id, on);
}

/** Reopens conversation `id` after the panel's own Reload (`reload.ts`), the screen held until it is read, so nothing is
 * sent into a new conversation meanwhile, and a read that lands later replaces nothing the person did. */
export async function reopenConversation(api: PanelApi, id: string): Promise<string | null> {
  const controller = hold();
  if (controller === null) return null;
  const on = ++screen;
  update((state) => ({ ...state, turns: [], fresh: null, notice: null }));
  try {
    return await reread(api, id, on);
  } finally {
    release(controller);
  }
}

/** A new conversation: the screen cleared, and the conversation made by mnemiq when its first message is sent. */
export function newConversation(): boolean {
  if (holding !== null) return false;
  screen += 1;
  update((state) => ({ ...state, conversationId: null, title: "", messages: [], turns: [], fresh: null, notice: null }));
  return true;
}

/** Sends `text` in the conversation on screen (a new one when none is), from the page `pageNow` resolves when it is
 * called, unless something is already in flight: one at a time, the screen held from the moment it is accepted.
 * Answers whether it was accepted. */
export function sendInPanel(api: PanelApi, text: string, pageNow: () => Promise<Page>, choose?: Choose): boolean {
  const controller = hold();
  if (controller === null) return false;
  const on = screen;
  const id = getSnapshot().conversationId;
  void send(api, text, pageNow, choose, controller.signal, on, id).finally(() => release(controller));
  return true;
}

async function send(api: PanelApi, text: string, pageNow: () => Promise<Page>, choose: Choose | undefined,
  signal: AbortSignal, on: number, startedIn: string | null) {
  update((state) => ({ ...state, notice: null, fresh: null, turns: [userTurn(text)] }));
  const read = (event: AguiEvent) => update((state) => ({ ...state, turns: reduce(state.turns, event) }));
  const failed = (sentence: string, code: string) => {
    read({ type: "RUN_STARTED", runId: "" });
    read({ type: "RUN_ERROR", message: sentence, code });
  };
  const page = await pageNow();
  let id = startedIn;
  if (id === null) {
    const started = await startConversation(api);
    if (!started.ok) {
      failed(started.refusal.sentence, started.refusal.kind);
      return;
    }
    id = started.body.conversation.id;
    update((state) => ({ ...state, conversationId: id }));
  }
  let sawEnd = false;
  let recorded: number | null = null;
  try {
    for await (const event of sendMessage(api, id, { text, page, choose }, signal)) {
      if (event.type === "RUN_FINISHED" || event.type === "RUN_ERROR") sawEnd = true;
      if (event.type === "CUSTOM" && (event as { name?: string }).name === "mnemiq.recorded") {
        const index = ((event as { value?: { index?: unknown } }).value ?? {}).index;
        if (typeof index === "number") recorded = index;
      }
      read(event);
    }
  } catch (error) {
    if ((error as { name?: string })?.name !== "AbortError") console.error("mnemiq panel: a message's stream broke", error);
  } finally {
    update((state) => ({ ...state, turns: closeRun(state.turns, sawEnd) }));
  }
  // What mnemiq kept is what the conversation shows from here. The answer just given keeps its card, rows and all,
  // while the person stays on it; a message mnemiq kept nothing of (refused before its stream started) keeps its turn,
  // saying why.
  const before = getSnapshot().messages.length;
  const live = [...getSnapshot().turns].reverse().find((turn: Turn) => turn.role === "assistant") ?? null;
  await reread(api, id, on);
  if (screen !== on) return;
  if (getSnapshot().messages.length > before || recorded !== null) {
    update((state) => ({ ...state, turns: [],
      fresh: recorded !== null && live?.answer ? { conversationId: id!, index: recorded, turn: live } : null }));
  }
}

/** Holds the screen while `work` answers a proposal in the conversation on screen, then reads it again. */
async function acting<T>(api: PanelApi, work: (id: string) => Promise<T>): Promise<T | null> {
  const id = getSnapshot().conversationId;
  if (id === null) return null;
  const controller = hold();
  if (controller === null) return null;
  const on = screen;
  try {
    const answered = await work(id);
    await reread(api, id, on);
    return answered;
  } finally {
    release(controller);
  }
}

/** The proposal `message` answered: "Build it", "Apply", or an offered dashboard's choice. The dashboard on screen is
 * read from the address when the person clicks (the page context can be a second behind it), and a proposal for a
 * dashboard is sent only from that dashboard. A build is not confirmed while this page plans or runs another, and a
 * build confirmed is run here, one step per request, as any build is. */
export async function confirmInPanel(api: PanelApi, message: Message, body: Omit<ConfirmBody, "dashboard_id">,
  pageNow: () => Promise<Page>): Promise<Answer<unknown> | null> {
  const proposal = message.proposal;
  const building = proposal !== undefined && !isChangePreview(proposal);
  if ((building || body.choice !== undefined) && driver.busy()) {
    update((state) => ({ ...state, notice: ALREADY_BUILDING }));
    return null;
  }
  return acting(api, async (id) => {
    const page = await pageNow();
    const wanted = proposal ? proposal.dashboard_id : null;
    if (wanted !== null && page.dashboardId !== wanted) {
      update((state) => ({ ...state, notice: OPEN_IT }));
      return null;
    }
    const answer = await confirmMessage(api, id, message.index, { ...body, dashboard_id: page.dashboardId });
    const built = answer.ok ? (answer.body as { build?: { build_id?: unknown } })?.build : undefined;
    if (built && typeof built.build_id === "string" && building) {
      driver.resume(proposal.goal, proposal.dashboard_id !== null ? { dashboardId: proposal.dashboard_id } : {},
        built.build_id);
    }
    update((state) => ({ ...state, notice: answer.ok ? null : answer.refusal.sentence }));
    return answer;
  }).then((answer) => {
    // A notice set by the answer outlives the reread that follows it.
    if (answer && !answer.ok) update((state) => ({ ...state, notice: answer.refusal.sentence }));
    return answer;
  });
}

/** The proposal `message` set aside: the person's "Not now". */
export async function declineInPanel(api: PanelApi, message: Message): Promise<void> {
  const answer = await acting(api, (id) => declineMessage(api, id, message.index));
  if (answer && !answer.ok) update((state) => ({ ...state, notice: answer.refusal.sentence }));
}

/** For tests: nothing in flight. */
export function resetTalk(): void {
  holding?.abort();
  holding = null;
  screen = 0;
}
