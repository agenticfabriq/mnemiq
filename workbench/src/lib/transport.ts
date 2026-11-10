/**
 * The client for /v1. Same-origin: the engine serves this bundle, so the browser holds
 * no engine address. A deployment that verifies its users puts a sign-in proxy in front
 * of the same origin, which adds each user's token to these requests on the way through.
 */

import { readSse } from "./sse";
import type { AguiEvent, HistoryTurn, Mode, SchemaTable, Whoami } from "./types";

/**
 * What a /v1 answer says about the session rather than the question asked. Kept apart
 * because each calls for something different: signed out means sign in again; not
 * permitted means signing in again would change nothing; unavailable means the service
 * is down, and the user's session (and saved threads) are fine.
 */
export type Refusal = "signed-out" | "not-permitted" | "insufficient-scope" | "unavailable";

/** `endSession`: where a refused sign-in's Sign out ends, as a 403 on whoami names it; null otherwise. */
export class Refused extends Error {
  constructor(
    readonly refusal: Refusal,
    readonly endSession: string | null = null,
  ) {
    super(refusal);
    this.name = "Refused";
  }
}

/** What the reader is told for each refusal short of signing out. */
export const NOTICE: Record<Exclude<Refusal, "signed-out">, string> = {
  "not-permitted": "Not permitted: the engine refused this sign-in.",
  "insufficient-scope": "Not permitted: this sign-in lacks the access the workbench needs.",
  unavailable: "The service is unavailable. Try again shortly.",
};

function refusalOf(response: Response): Refusal | null {
  // A redirect means the proxy's session expired and it wants the browser at the identity
  // provider: the same as a 401.
  if (response.status === 401 || response.type === "opaqueredirect") return "signed-out";
  if (response.status === 403) {
    const challenge = response.headers.get("www-authenticate") ?? "";
    return /\binsufficient_scope\b/.test(challenge) ? "insufficient-scope" : "not-permitted";
  }
  if (response.status === 503) return "unavailable";
  return null;
}

/**
 * fetch for /v1. A redirect is never followed: followed to the identity provider, it is a
 * cross-origin request the browser refuses, and that refusal is indistinguishable from the
 * network failing -- a signed-out user would be told the service is down.
 */
async function request(path: string, init: RequestInit = {}): Promise<Response> {
  try {
    return await fetch(path, { ...init, redirect: "manual" });
  } catch (error) {
    // fetch rejects with a TypeError when the request never got an answer. A cancelled
    // request rejects with an AbortError instead, and stays the caller's business.
    if (error instanceof TypeError) throw new Refused("unavailable");
    throw error;
  }
}

/** `request`, throwing `Refused` for an answer about the session. */
async function v1(path: string, init: RequestInit = {}): Promise<Response> {
  const response = await request(path, init);
  const refusal = refusalOf(response);
  if (refusal) throw new Refused(refusal);
  return response;
}

/**
 * How long a 403's body may take to arrive. The refusal is known from the status alone and the page shows
 * nothing until it is reported, while the body, a few dozen bytes the engine sends with the headers, only
 * says where Sign out ends: one still missing after this has stalled, and Sign out is left to the proxy.
 */
export const REFUSAL_BODY_MS = 2_000;

/** A response's body as text, or null when it fails or is not all in within `ms`, its read then cancelled. */
async function textWithin(response: Response, ms: number): Promise<string | null> {
  if (response.body === null) return null;
  const reader = response.body.getReader();
  const read = (async () => {
    const decoder = new TextDecoder();
    let text = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return text + decoder.decode();
      text += decoder.decode(value, { stream: true });
    }
  })();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const late = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), ms);
  });
  try {
    const text = await Promise.race([read, late]);
    // Cancelling ends the pending read, which then settles. Not awaited: a source that is slow to
    // cancel must not hold the refusal either.
    if (text === null) reader.cancel().catch(() => {});
    return text;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * The end-session URL a 403's body names beside its detail, `{"detail": ..., "end_session_url": ...}`,
 * read as whoami's 200 reads it. A body that cannot be read in time, or names none, reads as none: the
 * 403 is the answer either way, and reading its body never turns it into another refusal.
 */
async function endSessionOfRefusal(response: Response): Promise<string | null> {
  const text = await textWithin(response, REFUSAL_BODY_MS);
  if (text === null) return null;
  let body: { end_session_url?: unknown } | null;
  try {
    body = JSON.parse(text) as typeof body;
  } catch {
    return null;
  }
  return endSessionOf(body?.end_session_url);
}

/**
 * Who the engine says is asking, or `null` from an engine without the route, which serves
 * everyone as one identity. Anything that names nobody is the service failing: without an
 * identity there is no telling whose threads to show.
 */
export async function fetchWhoami(): Promise<Whoami | null> {
  const response = await request("/v1/whoami");
  const refusal = refusalOf(response);
  if (refusal) {
    // A 403 refuses this sign-in, so the user's way on is another account: the page offers a Sign
    // out, and needs to know where it ends. No other refusal names one.
    throw new Refused(refusal, response.status === 403 ? await endSessionOfRefusal(response) : null);
  }
  if (response.status === 404) return null;
  if (!response.ok) throw new Refused("unavailable");
  let body: { principal_id?: unknown; tenant_id?: unknown; verified?: unknown; end_session_url?: unknown };
  try {
    body = (await response.json()) as typeof body;
  } catch {
    throw new Refused("unavailable");
  }
  if (typeof body?.principal_id !== "string") throw new Refused("unavailable");
  // A string or an explicit null, nothing else. Folded to null, a missing or malformed tenant
  // would put the same principal in two tenants into one namespace, and would be acted on
  // (another identity's threads purged) before anyone noticed the answer was wrong.
  if (body.tenant_id !== null && typeof body.tenant_id !== "string") throw new Refused("unavailable");
  return {
    principal_id: body.principal_id,
    tenant_id: body.tenant_id,
    // Only an explicit false is unverified. Read the other way, an engine that left the field
    // out would hand a verified user the history of whoever used this browser before.
    verified: body.verified !== false,
    end_session_url: endSessionOf(body.end_session_url),
  };
}

/** The end-session URL the engine names, kept only as an absolute http(s) URL. The browser is sent
 *  there on Sign out, so anything else (a script URL, a relative path, a non-string, or nothing, from
 *  an engine before the field) reads as none, and Sign out stays the proxy's own. */
function endSessionOf(value: unknown): string | null {
  if (typeof value !== "string") return null;
  // As the engine's own startup check: printable ASCII and no backslash, which a browser reads as `/`,
  // and a host as written, since a browser percent-decodes a host before it reads it.
  if (/[^\x21-\x7e]|\\/.test(value)) return null;
  const authority = /^[A-Za-z][A-Za-z0-9+.-]*:\/\/([^/?#]*)/.exec(value)?.[1] ?? "";
  if (authority.includes("%")) return null;
  try {
    const { protocol, host } = new URL(value);
    if (protocol !== "https:" && protocol !== "http:") return null;
    // The backstop: a host the browser rewrites (a default port dropped, a hex or bracketed IP made canonical)
    // would not go where the engine checked, so keep the URL only when the browser reads its authority as written.
    return host === authority ? value : null;
  } catch {
    return null;
  }
}

/** A run that failed before the engine started one. `code` "unavailable" marks a service
 *  problem and "not-permitted" a refusal of this sign-in, so the card tells the reader which
 *  instead of saying the engine is still serving. */
function* failedRun(message: string, code?: "unavailable" | "not-permitted"): Generator<AguiEvent> {
  // The reducer patches the last assistant turn, and this run has none yet: without a start,
  // the error would land on the previous question's answer, or nowhere. No run id, since the
  // engine never issued one; the reducer names the turn itself.
  yield { type: "RUN_STARTED", runId: "" };
  yield { type: "RUN_ERROR", message, ...(code && { code }) };
}

export async function* streamChat(
  question: string,
  mode: Mode,
  history: HistoryTurn[] = [],
  signal?: AbortSignal,
): AsyncGenerator<AguiEvent> {
  let response: Response;
  try {
    response = await v1("/v1/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, mode, history }),
      signal,
    });
  } catch (error) {
    // Signed out is thrown on to the caller: the page purges this user's threads next, and
    // no event may reach a thread after that.
    if (!(error instanceof Refused) || error.refusal === "signed-out") throw error;
    yield* failedRun(NOTICE[error.refusal], error.refusal === "unavailable" ? "unavailable" : "not-permitted");
    return;
  }

  if (!response.ok || !response.body) {
    yield* failedRun(`the engine returned ${response.status} ${response.statusText}`.trim());
    return;
  }

  for await (const data of readSse(response.body)) {
    let event: AguiEvent;
    try {
      event = JSON.parse(data) as AguiEvent;
    } catch {
      continue; // a frame we cannot parse is not a reason to drop the whole run
    }
    yield event;
  }
}

export type Scope = { tables: SchemaTable[]; starters: string[] };

export async function fetchScope(signal?: AbortSignal): Promise<Scope> {
  const response = await v1("/v1/schema", { signal });
  if (!response.ok) return { tables: [], starters: [] };
  const body = (await response.json()) as { tables?: SchemaTable[]; starters?: string[] };
  return { tables: body.tables ?? [], starters: body.starters ?? [] };
}
