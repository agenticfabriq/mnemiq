/**
 * The chosen mode, remembered.
 *
 * Mode was plain component state, so every reload put the workbench back on `thinking`.
 * That is worse than a forgotten preference: mode is what a question *costs* -- `deep`
 * runs five candidates and judges them -- so a reader who picked it, refreshed, and asked
 * again silently got a cheaper answer than the one they were comparing against.
 *
 * Kept in localStorage beside the threads, and the consequence is the same one stated
 * there: this is per browser, not per principal.
 */

import { DEFAULT_MODE, MODES, type Mode } from "./types";

const KEY = "mnemiq.mode.v1";

const isMode = (value: unknown): value is Mode =>
  typeof value === "string" && (MODES as readonly string[]).includes(value);

export function loadMode(): Mode {
  try {
    const raw = globalThis.localStorage?.getItem(KEY);
    // Validated against the closed set rather than cast. Storage is the user's to edit and
    // outlives a release that renames a mode; an unrecognised value must not reach the
    // engine, which would refuse the question and make it look like the question's fault.
    return isMode(raw) ? raw : DEFAULT_MODE;
  } catch {
    return DEFAULT_MODE; // an unreadable preference is not a reason to fail to start
  }
}

export function saveMode(mode: Mode): void {
  try {
    globalThis.localStorage?.setItem(KEY, mode);
  } catch {
    // Over quota or storage denied. The session keeps the mode it is holding; only the
    // memory of it is lost, and losing it silently is better than refusing the click.
  }
}
