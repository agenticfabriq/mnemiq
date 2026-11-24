// The panel's "Reload to see the changes": the page reloaded so Superset draws what a build or a change wrote, and
// the conversation on screen reopened after it. Its id rides in the URL's fragment, which the browser never sends to
// a server, and is read once, and cleared, when the panel next opens: nothing reaches browser storage, and a reload of
// the browser's own starts empty, as it always has.

const KEY = "mnemiq-conversation";
// mnemiq's conversation ids are UUIDs (Postgres's `uuid`); anything else in the fragment is no conversation to open.
const ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** The page's own reload, held apart so a test can stand in for it (jsdom does not navigate). */
export const page = { reload: (): void => window.location.reload() };

/** Reloads the page, carrying `conversationId`, the conversation on screen if there is one, for the panel to reopen. */
export function reloadKeeping(conversationId: string | null): void {
  if (conversationId !== null && ID.test(conversationId)) {
    const { pathname, search } = window.location;
    history.replaceState(history.state, "", `${pathname}${search}#${KEY}=${encodeURIComponent(conversationId)}`);
  }
  page.reload();
}

/** The conversation a panel Reload carried, read once: the fragment is cleared whether or not it names one. Any other
 * fragment is left as it is. */
export function takeReloaded(): string | null {
  const { hash, pathname, search } = window.location;
  const prefix = `#${KEY}=`;
  if (!hash.startsWith(prefix)) return null;
  history.replaceState(history.state, "", `${pathname}${search}`);
  let id: string;
  try {
    id = decodeURIComponent(hash.slice(prefix.length));
  } catch {
    return null;
  }
  return ID.test(id) ? id : null;
}
