import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { SCALAR_ANSWER } from "./test/fixtures";
import { enter, keyFor, LEGACY, MARKER, mayReload, namespaceOf, SINGLE } from "./lib/identity";
import { NOTICE } from "./lib/transport";
import type { Thread } from "./lib/threads";

/**
 * The page as a whole, driven only through what the engine answers. Each storage rule is
 * tested on its own in identity.test.ts; these check that the page applies them before it
 * renders, and that every /v1 door routes its answer to the same handling.
 */

const ALICE = { principal_id: "alice@example.com", tenant_id: "tenant-a", verified: true };
const BOB = { principal_id: "bob@example.com", tenant_id: "tenant-a", verified: true };
const STARTER = "how many claims are there?";

const thread = (question: string): Thread => ({
  id: `t-${question}`,
  title: question,
  createdAt: 0,
  turns: [
    { id: `u-${question}`, role: "user", text: question, status: "complete" },
    { id: `a-${question}`, role: "assistant", text: SCALAR_ANSWER.answer, status: "complete", answer: SCALAR_ANSWER },
  ],
});

const stored = (question: string) => JSON.stringify([thread(question)]);

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });

const OPAQUE_REDIRECT = {
  type: "opaqueredirect",
  status: 0,
  ok: false,
  statusText: "",
  headers: new Headers(),
  body: null,
} as unknown as Response;

const SCOPE = () => json({ tables: [{ object_id: "claim", card: "" }], starters: [STARTER] });

type Route = "/v1/whoami" | "/v1/schema" | "/v1/chat";

function serve(routes: Partial<Record<Route, (init?: RequestInit) => Response | Promise<Response>>>) {
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const route = routes[url as Route];
    if (!route) throw new Error(`no route for ${url}`);
    return route(init);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const snapshot = (): Record<string, string> =>
  Object.fromEntries(
    Array.from({ length: localStorage.length }, (_, i) => {
      const key = localStorage.key(i)!;
      return [key, localStorage.getItem(key)!];
    }),
  );

const namespaced = () => Object.keys(snapshot()).filter((key) => key.startsWith("mnemiq.ns."));

/** A browser where `who` was the last to load the page and left a thread behind. */
async function signedInBefore(who: typeof ALICE, question: string): Promise<string> {
  const ns = (await namespaceOf(who))!;
  enter(ns);
  localStorage.setItem(keyFor(ns, "threads"), stored(question));
  localStorage.setItem(keyFor(ns, "mode"), "deep");
  return ns;
}

/** Wait until the workbench itself is on screen. */
const workbench = () => screen.findByRole("button", { name: "+ New chat" });

/** Let pending effects and microtasks run, so a write that would happen does. */
const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

const calls = (fetchMock: ReturnType<typeof serve>, route: Route) =>
  fetchMock.mock.calls.filter(([url]) => url === route).length;

/** Hide or show the tab, as switching to another tab or window does. */
function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, "visibilityState", { configurable: true, get: () => state });
  document.dispatchEvent(new Event("visibilitychange"));
}

beforeAll(() => {
  // assistant-ui's viewport observes its own size; jsdom has no ResizeObserver.
  globalThis.ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  // Its auto-scroll calls scrollTo on the viewport, which jsdom does not implement.
  Element.prototype.scrollTo ??= function () {};
});

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  delete (document as unknown as Record<string, unknown>).visibilityState;
});

describe("whose threads the page shows", () => {
  it("keeps a deployment's history when the engine has no whoami route", async () => {
    localStorage.setItem(LEGACY.threads, stored("how many policies are there?"));
    localStorage.setItem(LEGACY.mode, "deep");
    serve({ "/v1/whoami": () => new Response("", { status: 404 }), "/v1/schema": SCOPE });
    const reload = vi.fn();

    render(<App reload={reload} />);

    expect(await screen.findAllByText("how many policies are there?")).not.toHaveLength(0);
    expect(screen.getByRole("radio", { name: "deep" })).toHaveAttribute("aria-checked", "true");
    expect(localStorage.getItem(LEGACY.threads)).toBeNull();
    expect(localStorage.getItem(keyFor(SINGLE, "threads"))).toContain("how many policies are there?");
    expect(reload).not.toHaveBeenCalled();
  });

  it("keeps it when the engine reports the deployment's settings identity", async () => {
    localStorage.setItem(LEGACY.threads, stored("how many policies are there?"));
    serve({
      "/v1/whoami": () => json({ principal_id: "analyst", tenant_id: "local", verified: false }),
      "/v1/schema": SCOPE,
    });

    render(<App reload={vi.fn()} />);

    expect(await screen.findAllByText("how many policies are there?")).not.toHaveLength(0);
    expect(localStorage.getItem(LEGACY.threads)).toBeNull();
  });

  it("never shows the legacy history to a verified user", async () => {
    localStorage.setItem(LEGACY.threads, stored("how many policies are there?"));
    localStorage.setItem(LEGACY.mode, "deep");
    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": SCOPE });

    render(<App reload={vi.fn()} />);

    await workbench();
    expect(screen.queryAllByText("how many policies are there?")).toHaveLength(0);
    expect(screen.getByRole("radio", { name: "thinking" })).toHaveAttribute("aria-checked", "true");
    expect(localStorage.getItem(LEGACY.threads)).toBeNull();
    expect(localStorage.getItem(LEGACY.mode)).toBeNull();
  });

  it("never shows the legacy history when the engine leaves out whether it verified", async () => {
    localStorage.setItem(LEGACY.threads, stored("how many policies are there?"));
    serve({
      "/v1/whoami": () => json({ principal_id: "alice@example.com", tenant_id: "tenant-a" }),
      "/v1/schema": SCOPE,
    });

    render(<App reload={vi.fn()} />);

    await workbench();
    expect(screen.queryAllByText("how many policies are there?")).toHaveLength(0);
    expect(localStorage.getItem(LEGACY.threads)).toBeNull();
  });

  it("shows each user their own threads and purges the last user's", async () => {
    const alice = await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => json(BOB), "/v1/schema": SCOPE });

    render(<App reload={vi.fn()} />);

    await workbench();
    expect(screen.queryAllByText("alice's question")).toHaveLength(0);
    expect(localStorage.getItem(keyFor(alice, "threads"))).toBeNull();
    expect(localStorage.getItem(keyFor(alice, "mode"))).toBeNull();
  });

  it("shows the same user their threads again", async () => {
    await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": SCOPE });

    render(<App reload={vi.fn()} />);

    expect(await screen.findAllByText("alice's question")).not.toHaveLength(0);
  });

  it("stores no raw principal id under any key", async () => {
    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": SCOPE });
    render(<App reload={vi.fn()} />);
    fireEvent.click(await screen.findByRole("radio", { name: "deep" }));
    await settle();

    expect(Object.keys(snapshot()).length).toBeGreaterThan(0);
    for (const key of Object.keys(snapshot())) expect(key).not.toContain("alice");
  });
});

describe("a tab left open across a change of user", () => {
  it("drops the old user's threads and cancels their question when another tab loads as someone else", async () => {
    const alice = await signedInBefore(ALICE, "alice's question");
    let who = ALICE;
    let chatSignal: AbortSignal | undefined;
    const fetchMock = serve({
      "/v1/whoami": () => json(who),
      "/v1/schema": SCOPE,
      "/v1/chat": (init) => {
        chatSignal = init?.signal ?? undefined;
        return new Promise<Response>(() => {}); // still being answered
      },
    });
    const tabA = within(render(<App reload={vi.fn()} />).container);
    expect(await tabA.findAllByText("alice's question")).not.toHaveLength(0);
    fireEvent.click(await tabA.findByRole("button", { name: "+ New chat" }));
    fireEvent.click(await tabA.findByRole("button", { name: STARTER }));
    await waitFor(() => expect(chatSignal).toBeDefined());

    // Bob signs in through another tab, which loads the workbench as him.
    who = BOB;
    const tabB = within(render(<App reload={vi.fn()} />).container);
    await tabB.findByRole("button", { name: "+ New chat" });
    // What the browser tells every other tab when that one writes the identity marker.
    const bob = (await namespaceOf(BOB))!;
    window.dispatchEvent(
      new StorageEvent("storage", { key: MARKER, oldValue: alice, newValue: bob, storageArea: localStorage }),
    );

    await waitFor(() => expect(tabA.queryAllByText("alice's question")).toHaveLength(0));
    expect(chatSignal!.aborted).toBe(true);
    // Tab A asks again who is asking, and is Bob's now.
    await tabA.findByRole("button", { name: "+ New chat" });
    expect(calls(fetchMock, "/v1/whoami")).toBe(3);
    await settle();
    expect(localStorage.getItem(keyFor(alice, "threads"))).toBeNull();
  });

  it("checks who is asking before sending from a tab that was hidden, and sends nothing when it changed", async () => {
    await signedInBefore(ALICE, "alice's question");
    let who = ALICE;
    const chat = vi.fn(() => new Response("", { status: 500 }));
    const fetchMock = serve({ "/v1/whoami": () => json(who), "/v1/schema": SCOPE, "/v1/chat": chat });
    render(<App reload={vi.fn()} />);
    expect(await screen.findAllByText("alice's question")).not.toHaveLength(0);

    // Hidden while Bob signs in somewhere that never loads the workbench, then shown again.
    setVisibility("hidden");
    who = BOB;
    setVisibility("visible");
    fireEvent.click(await workbench());
    fireEvent.click(await screen.findByRole("button", { name: STARTER }));

    await waitFor(() => expect(screen.queryAllByText("alice's question")).toHaveLength(0));
    await workbench();
    await settle();
    expect(chat).not.toHaveBeenCalled();
    expect(calls(fetchMock, "/v1/whoami")).toBe(3); // the load, the check, and asking again
  });

  it("sends once the check finds the same user, after the tab sat idle", async () => {
    await signedInBefore(ALICE, "alice's question");
    const chat = vi.fn(() => new Response("", { status: 500 }));
    const fetchMock = serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": SCOPE, "/v1/chat": chat });
    render(<App reload={vi.fn()} />);
    expect(await screen.findAllByText("alice's question")).not.toHaveLength(0);

    const later = Date.now() + 61_000;
    vi.spyOn(Date, "now").mockReturnValue(later);
    fireEvent.click(await workbench());
    fireEvent.click(await screen.findByRole("button", { name: STARTER }));

    await waitFor(() => expect(chat).toHaveBeenCalledTimes(1));
    expect(calls(fetchMock, "/v1/whoami")).toBe(2);
    expect(screen.getAllByText("alice's question")).not.toHaveLength(0);
  });

  it("clears the tab and reloads, sending nothing, when the check finds the session expired", async () => {
    await signedInBefore(ALICE, "alice's question");
    let whoami = () => json(ALICE);
    const chat = vi.fn(() => new Response("", { status: 500 }));
    serve({ "/v1/whoami": () => whoami(), "/v1/schema": SCOPE, "/v1/chat": chat });
    const reload = vi.fn();
    render(<App reload={reload} />);
    expect(await screen.findAllByText("alice's question")).not.toHaveLength(0);

    setVisibility("hidden");
    whoami = () => new Response("", { status: 401 });
    setVisibility("visible");
    fireEvent.click(await workbench());
    fireEvent.click(await screen.findByRole("button", { name: STARTER }));

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    expect(chat).not.toHaveBeenCalled();
    expect(screen.queryAllByText("alice's question")).toHaveLength(0);
  });

  it("adds no round trip before a question from a tab in use", async () => {
    const chat = vi.fn(() => new Response("", { status: 500 }));
    const fetchMock = serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": SCOPE, "/v1/chat": chat });
    render(<App reload={vi.fn()} />);

    fireEvent.click(await screen.findByRole("button", { name: STARTER }));

    await waitFor(() => expect(chat).toHaveBeenCalledTimes(1));
    expect(calls(fetchMock, "/v1/whoami")).toBe(1);
  });
});

describe("the legacy history near the storage quota", () => {
  it("keeps showing and keeping it until it can be moved, never an empty list in its place", async () => {
    localStorage.setItem(LEGACY.threads, stored("how many policies are there?"));
    // Full storage: no room to copy the history into its namespace, room for an empty list.
    const real = Storage.prototype.setItem;
    Storage.prototype.setItem = function (key: string, value: string) {
      if (key.startsWith("mnemiq.ns.") && value.length > 2) {
        throw new DOMException("quota", "QuotaExceededError");
      }
      return real.call(this, key, value);
    };
    try {
      serve({ "/v1/whoami": () => new Response("", { status: 404 }), "/v1/schema": SCOPE });
      render(<App reload={vi.fn()} />);

      expect(await screen.findAllByText("how many policies are there?")).not.toHaveLength(0);
      await settle();
      expect(localStorage.getItem(LEGACY.threads)).toContain("how many policies are there?");
      expect(localStorage.getItem(keyFor(SINGLE, "threads"))).toBeNull();
    } finally {
      Storage.prototype.setItem = real;
    }
    cleanup();

    // The next load, with room again, moves it.
    render(<App reload={vi.fn()} />);

    expect(await screen.findAllByText("how many policies are there?")).not.toHaveLength(0);
    expect(localStorage.getItem(LEGACY.threads)).toBeNull();
    expect(localStorage.getItem(keyFor(SINGLE, "threads"))).toContain("how many policies are there?");
  });
});

describe("a malformed identity", () => {
  for (const [what, tenant_id] of [
    ["no tenant", undefined],
    ["a numeric tenant", 7],
    ["an object for a tenant", { id: "tenant-a" }],
  ] as const) {
    it(`reads ${what} as the service unavailable, touching no stored history`, async () => {
      await signedInBefore(ALICE, "alice's question");
      localStorage.setItem(LEGACY.threads, stored("how many policies are there?"));
      const before = snapshot();
      serve({ "/v1/whoami": () => json({ principal_id: "bob@example.com", tenant_id, verified: true }) });
      const reload = vi.fn();

      render(<App reload={reload} />);

      expect(await screen.findByText(NOTICE.unavailable)).toBeInTheDocument();
      await settle();
      expect(snapshot()).toEqual(before);
      expect(reload).not.toHaveBeenCalled();
    });
  }
});

describe("signed out", () => {
  // An expired session is not a sign-out: the screen and the tab's memory are cleared, the
  // saved threads are kept, and the page reloads to sign in again.
  it("clears the screen and reloads, keeping the saved threads, when whoami answers 401", async () => {
    await signedInBefore(ALICE, "alice's question");
    const before = snapshot();
    serve({ "/v1/whoami": () => new Response("", { status: 401 }) });
    const reload = vi.fn();

    render(<App reload={reload} />);

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    await settle();
    expect(snapshot()).toEqual(before);
    expect(screen.queryAllByText("alice's question")).toHaveLength(0);
  });

  it("clears the screen and reloads, keeping the saved threads, when the schema request is redirected", async () => {
    const alice = await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": () => OPAQUE_REDIRECT });
    const reload = vi.fn();

    render(<App reload={reload} />);

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    await settle();
    expect(screen.queryAllByText("alice's question")).toHaveLength(0);
    expect(localStorage.getItem(keyFor(alice, "threads"))).toContain("alice's question");
  });

  it("clears the screen and reloads, keeping the saved threads, when a question is answered 401", async () => {
    const alice = await signedInBefore(ALICE, "alice's question");
    serve({
      "/v1/whoami": () => json(ALICE),
      "/v1/schema": SCOPE,
      "/v1/chat": () => new Response("", { status: 401 }),
    });
    const reload = vi.fn();
    render(<App reload={reload} />);
    expect(await screen.findAllByText("alice's question")).not.toHaveLength(0);

    fireEvent.click(await workbench());
    fireEvent.click(await screen.findByRole("button", { name: STARTER }));

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    await settle();
    expect(screen.queryAllByText("alice's question")).toHaveLength(0);
    expect(localStorage.getItem(keyFor(alice, "threads"))).toContain("alice's question");
  });

  it("gives the same user their threads back once they have signed in again", async () => {
    await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => new Response("", { status: 401 }) });
    const reload = vi.fn();
    render(<App reload={reload} />);
    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    cleanup();

    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": SCOPE });
    render(<App reload={vi.fn()} />);

    expect(await screen.findAllByText("alice's question")).not.toHaveLength(0);
  });

  it("never lets a late refusal, from a tab opened before someone else signed in, erase their history", async () => {
    await signedInBefore(ALICE, "alice's question");
    serve({
      "/v1/whoami": () => json(ALICE),
      "/v1/schema": SCOPE,
      "/v1/chat": () => new Response("", { status: 401 }),
    });
    const reload = vi.fn();
    render(<App reload={reload} />);
    await workbench();

    // Bob signs in through another tab, which loads the workbench as him.
    const bob = await signedInBefore(BOB, "bob's question");
    // Alice's old tab asks a question; the proxy now holds no session for her.
    fireEvent.click(await workbench());
    fireEvent.click(await screen.findByRole("button", { name: STARTER }));

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    await settle();
    expect(localStorage.getItem(keyFor(bob, "threads"))).toContain("bob's question");
    expect(localStorage.getItem(keyFor(bob, "mode"))).toBe("deep");
  });

  it("asks the user to sign in instead of reloading again within 30 seconds", async () => {
    mayReload(Date.now() - 5_000); // the page reloaded to sign in five seconds ago
    const alice = await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => new Response("", { status: 401 }) });
    const reload = vi.fn();

    render(<App reload={reload} />);

    const button = await screen.findByRole("button", { name: "Sign in again" });
    expect(screen.getByText(/session has ended/i)).toBeInTheDocument();
    expect(reload).not.toHaveBeenCalled();
    expect(localStorage.getItem(keyFor(alice, "threads"))).toContain("alice's question");

    fireEvent.click(button);
    expect(reload).toHaveBeenCalledTimes(1);
  });
});

describe("signing out", () => {
  it("offers a verified user a Sign out that clears their threads and mode, then signs out at the proxy", async () => {
    const alice = await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": SCOPE });
    const navigate = vi.fn();
    const reload = vi.fn();
    render(<App reload={reload} navigate={navigate} />);
    expect(await screen.findAllByText("alice's question")).not.toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: "Sign out" }));

    // The proxy's own sign-out, sent back to this page afterwards.
    await waitFor(() => expect(navigate).toHaveBeenCalledWith("/oauth2/sign_out?rd=%2F"));
    await settle();
    expect(localStorage.getItem(keyFor(alice, "threads"))).toBeNull();
    expect(localStorage.getItem(keyFor(alice, "mode"))).toBeNull();
    expect(screen.queryAllByText("alice's question")).toHaveLength(0);
    expect(reload).not.toHaveBeenCalled();
  });

  for (const [what, whoami] of [
    ["an engine without the whoami route", () => new Response("", { status: 404 })],
    ["the settings identity", () => json({ principal_id: "analyst", tenant_id: "local", verified: false })],
  ] as const) {
    it(`offers no Sign out to ${what}, which has no sign-in to end`, async () => {
      serve({ "/v1/whoami": whoami, "/v1/schema": SCOPE });

      render(<App reload={vi.fn()} navigate={vi.fn()} />);

      await workbench();
      expect(screen.queryByRole("button", { name: "Sign out" })).not.toBeInTheDocument();
    });
  }
});

describe("refusals that are not a sign-out", () => {
  it("says a 403 is not permitted, without reloading or purging", async () => {
    await signedInBefore(ALICE, "alice's question");
    const before = snapshot();
    serve({ "/v1/whoami": () => new Response("", { status: 403 }) });
    const reload = vi.fn();

    render(<App reload={reload} />);

    expect(await screen.findByText(NOTICE["not-permitted"])).toBeInTheDocument();
    await settle();
    expect(reload).not.toHaveBeenCalled();
    expect(snapshot()).toEqual(before);
  });

  it("says a sign-in lacks the access it needs when the 403 names a missing scope", async () => {
    serve({
      "/v1/whoami": () =>
        new Response("", {
          status: 403,
          headers: { "www-authenticate": 'Bearer error="insufficient_scope", scope="db.read"' },
        }),
    });
    const reload = vi.fn();

    render(<App reload={reload} />);

    expect(await screen.findByText(NOTICE["insufficient-scope"])).toBeInTheDocument();
    expect(reload).not.toHaveBeenCalled();
  });

  it("shows a 403 on the schema above the user's own workbench", async () => {
    await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": () => new Response("", { status: 403 }) });
    const reload = vi.fn();

    render(<App reload={reload} />);

    expect(await screen.findByText(NOTICE["not-permitted"])).toBeInTheDocument();
    expect(screen.getAllByText("alice's question")).not.toHaveLength(0);
    expect(reload).not.toHaveBeenCalled();
  });

  for (const [what, failure] of [
    ["a 503", () => serve({ "/v1/whoami": () => new Response("", { status: 503 }) })],
    [
      "a network failure",
      () => vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch"))),
    ],
  ] as const) {
    it(`says ${what} is the service being unavailable, without reloading or purging`, async () => {
      await signedInBefore(ALICE, "alice's question");
      const before = snapshot();
      failure();
      const reload = vi.fn();

      render(<App reload={reload} />);

      expect(await screen.findByText(NOTICE.unavailable)).toBeInTheDocument();
      await settle();
      expect(reload).not.toHaveBeenCalled();
      expect(snapshot()).toEqual(before);
      expect(screen.queryByText(/sign in/i)).not.toBeInTheDocument();
    });
  }

  it("says the service is unavailable when the schema answers 503, keeping the workbench", async () => {
    await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": () => new Response("", { status: 503 }) });
    const reload = vi.fn();

    render(<App reload={reload} />);

    expect(await screen.findByText(NOTICE.unavailable)).toBeInTheDocument();
    expect(screen.getAllByText("alice's question")).not.toHaveLength(0);
    // An outage is not an empty grant: the table list must not claim nothing is granted.
    expect(screen.queryByText(/No tables are granted/)).not.toBeInTheDocument();
    expect(reload).not.toHaveBeenCalled();
  });

  it("answers a question asked during an outage with the service message, keeping the threads", async () => {
    await signedInBefore(ALICE, "alice's question");
    serve({
      "/v1/whoami": () => json(ALICE),
      "/v1/schema": SCOPE,
      "/v1/chat": () => new Response("", { status: 503 }),
    });
    const reload = vi.fn();
    render(<App reload={reload} />);

    fireEvent.click(await workbench());
    fireEvent.click(await screen.findByRole("button", { name: STARTER }));

    expect(await screen.findByText(NOTICE.unavailable)).toBeInTheDocument();
    expect(screen.getByText(/service problem/)).toBeInTheDocument();
    expect(screen.queryByText(/still serving/)).not.toBeInTheDocument();
    await settle();
    expect(reload).not.toHaveBeenCalled();
    expect(screen.getAllByText("alice's question")).not.toHaveLength(0);
    expect(namespaced().length).toBeGreaterThan(0);
  });
});

describe("a page that cannot keep users apart", () => {
  it("says saved threads are off, and why, when it cannot hash a verified user's identity", async () => {
    // crypto.subtle exists only in a secure context. Without it the user gets no saved
    // threads, and that must not happen silently.
    const real = globalThis.crypto;
    vi.stubGlobal("crypto", { randomUUID: () => real.randomUUID() });
    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": SCOPE });

    render(<App reload={vi.fn()} />);

    expect(await screen.findByText(/Saved threads are off on this page/)).toBeInTheDocument();
    expect(screen.getByText(/secure context, such as HTTPS/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "deep" }));
    await settle();
    expect(namespaced()).toEqual([]);
  });

  it("says nothing of the kind where it can", async () => {
    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": SCOPE });

    render(<App reload={vi.fn()} />);

    await workbench();
    expect(screen.queryByText(/Saved threads are off/)).not.toBeInTheDocument();
  });

  it("says nothing of the kind to a deployment that verifies nobody, which needs no hash", async () => {
    const real = globalThis.crypto;
    vi.stubGlobal("crypto", { randomUUID: () => real.randomUUID() });
    serve({ "/v1/whoami": () => new Response("", { status: 404 }), "/v1/schema": SCOPE });

    render(<App reload={vi.fn()} />);

    await workbench();
    expect(screen.queryByText(/Saved threads are off/)).not.toBeInTheDocument();
  });
});
