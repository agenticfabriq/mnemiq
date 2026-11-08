import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { SCALAR_ANSWER } from "./test/fixtures";
import { enter, keyFor, LEGACY, mayReload, namespaceOf, SINGLE } from "./lib/identity";
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

function serve(routes: Partial<Record<Route, () => Response>>) {
  const fetchMock = vi.fn(async (url: string) => {
    const route = routes[url as Route];
    if (!route) throw new Error(`no route for ${url}`);
    return route();
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
  it("purges the user's threads and reloads when whoami answers 401", async () => {
    await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => new Response("", { status: 401 }) });
    const reload = vi.fn();

    render(<App reload={reload} />);

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    expect(namespaced()).toEqual([]);
    expect(screen.queryAllByText("alice's question")).toHaveLength(0);
  });

  it("purges and reloads when the schema request is redirected to sign in", async () => {
    await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => json(ALICE), "/v1/schema": () => OPAQUE_REDIRECT });
    const reload = vi.fn();

    render(<App reload={reload} />);

    await waitFor(() => expect(reload).toHaveBeenCalledTimes(1));
    await settle();
    expect(namespaced()).toEqual([]);
    expect(screen.queryAllByText("alice's question")).toHaveLength(0);
  });

  it("purges and reloads when a question is answered 401, and writes nothing back", async () => {
    await signedInBefore(ALICE, "alice's question");
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
    expect(namespaced()).toEqual([]);
    expect(screen.queryAllByText("alice's question")).toHaveLength(0);
  });

  it("asks the user to sign in instead of reloading again within 30 seconds", async () => {
    mayReload(Date.now() - 5_000); // the page reloaded to sign in five seconds ago
    await signedInBefore(ALICE, "alice's question");
    serve({ "/v1/whoami": () => new Response("", { status: 401 }) });
    const reload = vi.fn();

    render(<App reload={reload} />);

    const button = await screen.findByRole("button", { name: "Sign in again" });
    expect(screen.getByText(/session has ended/i)).toBeInTheDocument();
    expect(reload).not.toHaveBeenCalled();
    expect(namespaced()).toEqual([]);

    fireEvent.click(button);
    expect(reload).toHaveBeenCalledTimes(1);
  });
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
