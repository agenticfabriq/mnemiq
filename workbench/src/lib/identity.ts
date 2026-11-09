/**
 * Whose saved threads these are.
 *
 * Threads (answers, rows included) and the mode live in localStorage, which belongs to the
 * browser, not to whoever signed in. Once a deployment verifies each request's user, one
 * shared browser would show the next user the last one's answers. So each identity's items
 * live under a namespace of its own and only the current one is ever read.
 *
 * A namespace is named by a SHA-256 of the identity, never the id itself: the id is often an
 * email address, and every key sits in plain view in the browser's storage panel.
 */

import type { Identity } from "./types";

const PREFIX = "mnemiq.ns.";

/** The namespace of whoever loaded the page last, so a load can tell the user changed. */
export const MARKER = "mnemiq.identity.v1";

/** Where threads and the mode lived before they were per identity. */
export const LEGACY = { threads: "mnemiq.threads.v1", mode: "mnemiq.mode.v1" } as const;

export type Item = keyof typeof LEGACY;

/**
 * The one namespace of a deployment that verifies nobody: an engine without the whoami route,
 * or the settings identity. Both are the one identity every caller had before an issuer, so
 * they share it -- keyed apart, upgrading the engine to one with the route would read as a
 * change of user and purge the very history a deployment without an issuer must keep.
 */
export const SINGLE = "single";

/**
 * Where the one identity's items stay while its legacy history cannot be moved (storage full):
 * the legacy keys themselves. Read and written there as before, and moved on a later load.
 */
export const LEGACY_SHELF = "legacy";

const ITEMS = Object.keys(LEGACY) as Item[];

export const RELOAD_GUARD_MS = 30_000;
const RELOADED_AT = "mnemiq.signin-reload.v1";

export const keyFor = (ns: string, item: Item): string =>
  ns === LEGACY_SHELF ? LEGACY[item] : `${PREFIX}${ns}.${item}.v1`;

/**
 * The namespace for the identity the engine reported, or `null` when there is no safe name for
 * it. `crypto.subtle` exists only in a secure context; a verified user on a page without it
 * gets no namespace, so nothing of theirs is read or written.
 */
export async function namespaceOf(who: Identity | null): Promise<string | null> {
  if (who === null || !who.verified) return SINGLE;
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) return null;
  // Both parts, as JSON, so neither can be shifted into the other: ("a.b", "c") and
  // ("a", "b.c") must not name the same namespace.
  const digest = await subtle.digest(
    "SHA-256",
    new TextEncoder().encode(JSON.stringify([who.tenant_id, who.principal_id])),
  );
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function removeKeys(storage: Storage, doomed: (key: string) => boolean): void {
  const keys: string[] = [];
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i);
    if (key !== null && doomed(key)) keys.push(key);
  }
  for (const key of keys) storage.removeItem(key);
}

/** Remove every namespaced item except those of `keep`. */
function purgeNamespaces(storage: Storage, keep: string | null): void {
  const own = keep === null ? null : `${PREFIX}${keep}.`;
  removeKeys(storage, (key) => key.startsWith(PREFIX) && !(own && key.startsWith(own)));
}

function purgeLegacy(storage: Storage): void {
  for (const key of Object.values(LEGACY)) storage.removeItem(key);
}

/** No history to keep: absent, or an empty list of threads. */
const holdsNothing = (item: Item, raw: string | null): boolean =>
  raw === null || (item === "threads" && raw === "[]");

/**
 * Move the legacy items into the one identity's namespace, all or none. The legacy items are
 * removed only after every copy is written; when one cannot be (storage full), the copies made
 * so far are taken back and the legacy items stay exactly as they were. Answers whether the
 * history now lives in the namespace.
 */
function moveLegacy(storage: Storage): boolean {
  const copied: string[] = [];
  try {
    for (const item of ITEMS) {
      const legacy = storage.getItem(LEGACY[item]);
      if (legacy === null) continue;
      // Unless the namespace already has its own history: a tab still on the old bundle can
      // write the legacy key after the move. An empty list is not history, so it never
      // stands in the way.
      const target = keyFor(SINGLE, item);
      if (!holdsNothing(item, storage.getItem(target))) continue;
      storage.setItem(target, legacy);
      copied.push(target);
    }
  } catch {
    for (const key of copied) storage.removeItem(key);
    return false;
  }
  purgeLegacy(storage);
  return true;
}

/**
 * Make `ns` the page's namespace. Runs before anything is rendered, so another identity's
 * threads are gone before they could be shown. Answers where this identity's items are read
 * and written: `ns`, or `LEGACY_SHELF` while the one identity's legacy history cannot be moved.
 */
export function enter(ns: string | null): string | null {
  let storage: Storage | undefined;
  try {
    storage = globalThis.localStorage;
  } catch {
    return ns; // storage denied: nothing was stored to leak, and nothing can be
  }
  if (!storage) return ns;

  try {
    // A different identity than last time, or no record of one: whatever other namespaces
    // hold belongs to someone else.
    if (ns === null || storage.getItem(MARKER) !== ns) purgeNamespaces(storage, ns);
  } catch {
    // Storage denied part way: what could not be read could not be shown either.
  }

  let shelf = ns;
  try {
    if (ns === SINGLE) {
      // The legacy items are this identity's own history.
      if (!moveLegacy(storage)) shelf = LEGACY_SHELF;
    } else {
      // Whoever used this browser before an issuer: never handed to a verified user.
      purgeLegacy(storage);
    }
  } catch {
    shelf = ns === SINGLE ? LEGACY_SHELF : ns;
  }

  try {
    if (ns === null) storage.removeItem(MARKER);
    else storage.setItem(MARKER, ns);
  } catch {
    // Full: the next load compares against the older marker and purges again, which is safe.
  }
  return shelf;
}

/** The sign-in proxy's own sign-out, which ends its session and then sends the browser to `rd`. */
export const SIGN_OUT = "/oauth2/sign_out";

/**
 * Where Sign out sends the browser. Without an end-session URL, the proxy's sign-out, back to the
 * page signed out from, exactly as before. With one, `rd` is the identity provider's end-session URL,
 * carrying the workbench's root as `post_logout_redirect_uri` (one exact address the provider can
 * register), so the provider's session ends even when the proxy could not end it, and nobody is
 * signed straight back in.
 */
export function signOutHref(endSession: string | null, origin: string, back: string): string {
  if (endSession === null) return `${SIGN_OUT}?rd=${encodeURIComponent(back)}`;
  const end = new URL(endSession);
  end.searchParams.set("post_logout_redirect_uri", `${origin}/`);
  return `${SIGN_OUT}?rd=${encodeURIComponent(end.toString())}`;
}

/**
 * An explicit sign-out: the signing-out user's threads and mode go, and nothing else. Nothing
 * but the Sign out control purges on signing out; an expired session keeps the threads (the
 * page's handling of a refusal says why).
 */
export function signOut(ns: string | null): void {
  if (ns === null) return;
  try {
    const storage = globalThis.localStorage;
    if (storage) removeKeys(storage, (key) => key.startsWith(`${PREFIX}${ns}.`));
  } catch {
    // Storage denied: there is nothing in it to purge.
  }
}

/**
 * Whether the page may reload now to have the sign-in proxy sign the user in again, recording
 * the reload when it may. At most one per guard window: a proxy that answers every request
 * with a 401 would otherwise reload the page forever. When the time cannot be recorded the
 * answer is no, for the same reason.
 */
export function mayReload(now: number = Date.now()): boolean {
  try {
    const last = Number(globalThis.sessionStorage.getItem(RELOADED_AT));
    if (last && now - last < RELOAD_GUARD_MS) return false;
    globalThis.sessionStorage.setItem(RELOADED_AT, String(now));
    return true;
  } catch {
    return false;
  }
}
