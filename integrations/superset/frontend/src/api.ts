// The panel's client for mnemiq, through the extension's forwarder (`/extensions/agenticfabriq/mnemiq/v1/...`), which
// mints a token for the signed-in person on each request. The browser holds no mnemiq token and never sets
// `Authorization`; Superset's own session cookie rides along (same origin), and a POST carries the session's CSRF
// token, which the forwarder requires.
import { readSse, type AguiEvent } from "./shared";

export type RefusalKind =
  | "signed-out"
  | "not-a-builder"
  | "not-for-building"
  | "not-permitted"
  | "busy"
  | "unavailable"
  | "too-old"
  | "refused";

export type PanelRefusal = { kind: RefusalKind; sentence: string };

export type Answer<T> = { ok: true; status: number; body: T } | { ok: false; status: number; refusal: PanelRefusal };

// mnemiq's sentence for a person Superset will not run queries for (`NOT_A_BUILDER` in mnemiq's superset/grants.py).
const NOT_A_BUILDER_OPENING = "Superset refused this user's queries.";

export const TOO_OLD = "This mnemiq does not offer that yet; update mnemiq.";

// The panel's own words, for a refusal that carries no sentence of mnemiq's.
const SAID: Record<RefusalKind, string> = {
  "signed-out": "You are signed out of Superset; sign in again.",
  "not-a-builder": "Superset does not let this user build with mnemiq.",
  "not-for-building": "This sign-in may ask mnemiq, but not build with it.",
  "not-permitted": "Superset does not permit that for this user.",
  busy: "Another step of this build is running.",
  unavailable: "mnemiq is unavailable; try again shortly.",
  "too-old": TOO_OLD,
  refused: "mnemiq could not do that.",
};

/** What a refused answer means to the panel, with mnemiq's sentence (`detail`) when it gave one, or null when the
 * answer is not one of these: a 2xx, and any status the table leaves to the caller (another 404, 400, 413, 422). */
export function refusalOf(response: Response, detail: string | null): PanelRefusal | null {
  const said = (kind: RefusalKind, sentence = detail) => ({ kind, sentence: sentence?.trim() || SAID[kind] });
  // A redirect to Superset's sign-in, which `redirect: "manual"` leaves as an opaque answer.
  if (response.type === "opaqueredirect") return said("signed-out", null);
  const status = response.status;
  if (status === 401) {
    // mnemiq's own 401 asks for its Bearer token again: the extension's token was refused, which signing in again
    // cannot fix. Superset's 401 (an ended session) asks for nothing.
    return /\bbearer\b/i.test(response.headers.get("www-authenticate") ?? "") ? said("unavailable") : said("signed-out", null);
  }
  if (status === 403) {
    if (detail?.startsWith(NOT_A_BUILDER_OPENING)) return said("not-a-builder");
    if (/\binsufficient_scope\b/.test(response.headers.get("www-authenticate") ?? "")) return said("not-for-building");
    return said("not-permitted");
  }
  if (status === 409) return said("busy");
  if (status === 502 || status === 503 || status === 504) return said("unavailable");
  // FastAPI's own answer for a path it has no route for: a mnemiq older than the panel.
  if (status === 404 && detail === "Not Found") return said("too-old", TOO_OLD);
  return null;
}

/** mnemiq's sentence in an answer's body (`{"detail": "..."}`), or null. */
async function detailOf(response: Response): Promise<{ detail: string | null; body: unknown }> {
  let body: unknown = null;
  try {
    body = await response.json();
  } catch {
    return { detail: null, body: null };
  }
  const detail = (body as { detail?: unknown } | null)?.detail;
  return { detail: typeof detail === "string" ? detail : null, body };
}

export class PanelApi {
  readonly base: string;

  constructor(
    root: string,
    private readonly csrf: () => Promise<string | undefined>,
    private readonly fetchFn: typeof fetch = (...args) => fetch(...args),
  ) {
    this.base = `${root}/extensions/agenticfabriq/mnemiq/v1`;
  }

  /** `GET` mnemiq's `path` (without `/v1`). */
  get<T>(path: string, signal?: AbortSignal): Promise<Answer<T>> {
    return this.send<T>(path, { method: "GET", headers: { Accept: "application/json" }, signal });
  }

  /** `POST` `body` to mnemiq's `path` (without `/v1`), with the session's CSRF token. */
  async post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<Answer<T>> {
    const token = await this.csrf();
    const headers: Record<string, string> = { Accept: "application/json", "Content-Type": "application/json" };
    if (token) headers["X-CSRFToken"] = token;
    return this.send<T>(path, { method: "POST", headers, body: JSON.stringify(body), signal });
  }

  /** `POST` `body` to mnemiq's streaming `path` (`/chat`), yielding each event as it arrives. A refusal before the
   * stream starts is a run of its own that failed with the refusal's sentence, as the workbench's transport makes it,
   * so the conversation reads it as an answer that could not be given. */
  async *stream(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<AguiEvent> {
    const token = await this.csrf();
    const headers: Record<string, string> = { Accept: "text/event-stream", "Content-Type": "application/json" };
    if (token) headers["X-CSRFToken"] = token;
    let response: Response;
    try {
      response = await this.request(path, { method: "POST", headers, body: JSON.stringify(body), signal });
    } catch (error) {
      if ((error as { name?: string })?.name === "AbortError") throw error;
      yield* failedRun(SAID.unavailable, "unavailable");
      return;
    }
    if (!response.ok || response.body === null) {
      const { detail } = await detailOf(response);
      const refusal = refusalOf(response, detail) ?? { kind: "refused", sentence: detail?.trim() || SAID.refused };
      yield* failedRun(refusal.sentence, refusal.kind);
      return;
    }
    for await (const data of readSse(response.body)) {
      try {
        yield JSON.parse(data) as AguiEvent;
      } catch {
        // A frame that is not JSON carries nothing the conversation reads.
      }
    }
  }

  /** The request as every call makes it: same origin, Superset's session, and no redirect followed (a redirect is a
   * sign-in page, which the panel reads as signed out). */
  request(path: string, init: RequestInit): Promise<Response> {
    return this.fetchFn(`${this.base}${path}`, { ...init, credentials: "same-origin", redirect: "manual" });
  }

  private async send<T>(path: string, init: RequestInit): Promise<Answer<T>> {
    let response: Response;
    try {
      response = await this.request(path, init);
    } catch (error) {
      if ((error as { name?: string })?.name === "AbortError") throw error;
      return { ok: false, status: 0, refusal: { kind: "unavailable", sentence: SAID.unavailable } };
    }
    const { detail, body } = await detailOf(response);
    const refusal = refusalOf(response, detail);
    if (refusal) return { ok: false, status: response.status, refusal };
    if (!response.ok) {
      return { ok: false, status: response.status, refusal: { kind: "refused", sentence: detail?.trim() || SAID.refused } };
    }
    return { ok: true, status: response.status, body: body as T };
  }
}

/** A run that failed before it started: the events the conversation reads as an answer that could not be given. */
function* failedRun(sentence: string, code: string): Generator<AguiEvent> {
  yield { type: "RUN_STARTED", runId: "" };
  yield { type: "RUN_ERROR", message: sentence, code };
}
