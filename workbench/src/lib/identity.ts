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

import type { Whoami } from "./types";

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
export async function namespaceOf(who: Whoami | null): Promise<string | null> {
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

/** Remove every namespaced item except those of `keep`. */
function purgeNamespaces(storage: Storage, keep: string | null): void {
  const own = keep === null ? null : `${PREFIX}${keep}.`;
  const doomed: string[] = [];
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i);
    if (key?.startsWith(PREFIX) && !(own && key.startsWith(own))) doomed.push(key);
  }
  for (const key of doomed) storage.removeItem(key);
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

/** Signed out: nobody's threads should remain. The remembered namespace stays, so the next
 *  load can tell whether the same user came back. */
export function signOut(): void {
  try {
    const storage = globalThis.localStorage;
    if (!storage) return;
    purgeNamespaces(storage, null);
    purgeLegacy(storage);
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
