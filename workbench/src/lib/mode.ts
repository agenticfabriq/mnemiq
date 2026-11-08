/**
 * The chosen mode, remembered.
 *
 * Mode was plain component state, so every reload put the workbench back on `thinking`.
 * That is worse than a forgotten preference: mode is what a question *costs* -- `deep`
 * runs five candidates and judges them -- so a reader who picked it, refreshed, and asked
 * again silently got a cheaper answer than the one they were comparing against.
 *
 * Kept in localStorage beside the threads, in the same identity's namespace (identity.ts).
 */

import { keyFor } from "./identity";
import { DEFAULT_MODE, MODES, type Mode } from "./types";

const isMode = (value: unknown): value is Mode =>
  typeof value === "string" && (MODES as readonly string[]).includes(value);

/** `ns` null: a user with no namespace (identity.ts), for whom nothing is kept. */
export function loadMode(ns: string | null): Mode {
  if (ns === null) return DEFAULT_MODE;
  try {
    const raw = globalThis.localStorage?.getItem(keyFor(ns, "mode"));
    // Validated against the closed set rather than cast. Storage is the user's to edit and
    // outlives a release that renames a mode; an unrecognised value must not reach the
    // engine, which would refuse the question and make it look like the question's fault.
    return isMode(raw) ? raw : DEFAULT_MODE;
  } catch {
    return DEFAULT_MODE; // an unreadable preference is not a reason to fail to start
  }
}

export function saveMode(ns: string | null, mode: Mode): void {
  if (ns === null) return;
  try {
    globalThis.localStorage?.setItem(keyFor(ns, "mode"), mode);
  } catch {
    // Over quota or storage denied. The session keeps the mode it is holding; only the
    // memory of it is lost, and losing it silently is better than refusing the click.
  }
}
