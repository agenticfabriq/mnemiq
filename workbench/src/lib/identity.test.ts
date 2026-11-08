import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  LEGACY,
  LEGACY_SHELF,
  RELOAD_GUARD_MS,
  SINGLE,
  enter,
  keyFor,
  mayReload,
  namespaceOf,
  signOut,
} from "./identity";

const ALICE = { principal_id: "alice@example.com", tenant_id: "tenant-a", verified: true };
const BOB = { principal_id: "bob@example.com", tenant_id: "tenant-a", verified: true };

const keys = (): string[] =>
  Array.from({ length: localStorage.length }, (_, i) => localStorage.key(i)!);

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("namespaceOf", () => {
  it("reads an engine without the whoami route as one identity", async () => {
    expect(await namespaceOf(null)).toBe(SINGLE);
  });

  it("reads the settings identity as that same one identity", async () => {
    // Unverified is the deployment's one configured identity, as it was before any issuer.
    // Keyed apart from the no-route namespace, an engine upgrade that adds the route would
    // read as an identity change and purge the history it is meant to keep.
    const settings = { principal_id: "analyst", tenant_id: "local", verified: false };
    expect(await namespaceOf(settings)).toBe(SINGLE);
  });

  it("gives each verified user a namespace of their own", async () => {
    const alice = await namespaceOf(ALICE);
    const bob = await namespaceOf(BOB);
    expect(alice).toMatch(/^[0-9a-f]{64}$/);
    expect(bob).toMatch(/^[0-9a-f]{64}$/);
    expect(alice).not.toBe(bob);
  });

  it("keeps the same user in two tenants apart", async () => {
    const other = { ...ALICE, tenant_id: "tenant-b" };
    expect(await namespaceOf(ALICE)).not.toBe(await namespaceOf(other));
  });

  it("never carries the raw id", async () => {
    const ns = (await namespaceOf(ALICE))!;
    expect(ns).not.toContain("alice");
    expect(keyFor(ns, "threads")).not.toContain("alice");
  });

  it("keeps nothing for a verified user when the page cannot hash", async () => {
    // crypto.subtle exists only in a secure context. Without it there is no safe name for
    // a user's namespace, so nothing is read or written for them.
    vi.stubGlobal("crypto", {});
    expect(await namespaceOf(ALICE)).toBeNull();
    expect(await namespaceOf(null)).toBe(SINGLE);
  });
});

describe("enter", () => {
  it("moves the legacy history into the one identity's namespace, once", () => {
    localStorage.setItem(LEGACY.threads, '[{"id":"t1","turns":[]}]');
    localStorage.setItem(LEGACY.mode, "deep");

    enter(SINGLE);

    expect(localStorage.getItem(keyFor(SINGLE, "threads"))).toBe('[{"id":"t1","turns":[]}]');
    expect(localStorage.getItem(keyFor(SINGLE, "mode"))).toBe("deep");
    expect(localStorage.getItem(LEGACY.threads)).toBeNull();
    expect(localStorage.getItem(LEGACY.mode)).toBeNull();
  });

  it("never overwrites a namespace's own history with a late legacy copy", () => {
    // A tab still running the old bundle can write the legacy key after the move.
    localStorage.setItem(keyFor(SINGLE, "threads"), "[newer]");
    localStorage.setItem(LEGACY.threads, "[older]");

    enter(SINGLE);

    expect(localStorage.getItem(keyFor(SINGLE, "threads"))).toBe("[newer]");
    expect(localStorage.getItem(LEGACY.threads)).toBeNull();
  });

  it("purges the legacy history for a verified user rather than handing it over", async () => {
    // The legacy key belonged to whoever used this browser before an issuer: not this user.
    localStorage.setItem(LEGACY.threads, '[{"id":"t1","turns":[]}]');
    localStorage.setItem(LEGACY.mode, "deep");
    const ns = (await namespaceOf(ALICE))!;

    enter(ns);

    expect(localStorage.getItem(LEGACY.threads)).toBeNull();
    expect(localStorage.getItem(LEGACY.mode)).toBeNull();
    expect(localStorage.getItem(keyFor(ns, "threads"))).toBeNull();
    expect(localStorage.getItem(keyFor(ns, "mode"))).toBeNull();
  });

  it("purges every other namespace when a different identity loads", async () => {
    const alice = (await namespaceOf(ALICE))!;
    const bob = (await namespaceOf(BOB))!;
    enter(alice);
    localStorage.setItem(keyFor(alice, "threads"), "[alice's]");
    localStorage.setItem(keyFor(alice, "mode"), "deep");

    enter(bob);

    expect(localStorage.getItem(keyFor(alice, "threads"))).toBeNull();
    expect(localStorage.getItem(keyFor(alice, "mode"))).toBeNull();
  });

  it("keeps the namespace of the identity that loaded last time", async () => {
    const alice = (await namespaceOf(ALICE))!;
    enter(alice);
    localStorage.setItem(keyFor(alice, "threads"), "[alice's]");

    enter(alice);

    expect(localStorage.getItem(keyFor(alice, "threads"))).toBe("[alice's]");
  });

  it("purges a namespace it cannot vouch for when nobody was remembered", async () => {
    // No record of who loaded last: a namespace already here is of unknown origin.
    const alice = (await namespaceOf(ALICE))!;
    localStorage.setItem(keyFor(SINGLE, "threads"), "[someone's]");

    enter(alice);

    expect(localStorage.getItem(keyFor(SINGLE, "threads"))).toBeNull();
  });

  it("purges everything when a verified user has no namespace", async () => {
    localStorage.setItem(keyFor(SINGLE, "threads"), "[someone's]");
    localStorage.setItem(LEGACY.threads, "[legacy]");
    enter(SINGLE);

    enter(null);

    expect(keys().filter((key) => key.includes("threads"))).toEqual([]);
  });

  it("leaves storage it does not own alone", () => {
    localStorage.setItem("mnemiq.scope-open.v1", "1");
    localStorage.setItem(keyFor(SINGLE, "threads"), "[x]");

    enter(null);

    expect(localStorage.getItem("mnemiq.scope-open.v1")).toBe("1");
  });

  it("stores no raw principal id under any key", async () => {
    localStorage.setItem(LEGACY.threads, "[legacy]");
    const ns = (await namespaceOf(ALICE))!;
    enter(ns);
    localStorage.setItem(keyFor(ns, "threads"), "[x]");
    localStorage.setItem(keyFor(ns, "mode"), "deep");

    for (const key of keys()) {
      expect(key).not.toContain(ALICE.principal_id);
      expect(key).not.toContain("alice");
      expect(localStorage.getItem(key)).not.toContain("alice");
    }
  });
});

/** Storage that is full: it refuses any write to a key `refuse` picks. Returns the undo. */
function fullFor(refuse: (key: string, value: string) => boolean): () => void {
  const real = Storage.prototype.setItem;
  Storage.prototype.setItem = function (key: string, value: string) {
    if (refuse(key, value)) throw new DOMException("quota", "QuotaExceededError");
    return real.call(this, key, value);
  };
  return () => {
    Storage.prototype.setItem = real;
  };
}

describe("moving the legacy history near the storage quota", () => {
  it("answers where the history lives: the namespace once moved", () => {
    localStorage.setItem(LEGACY.threads, "[legacy]");
    expect(enter(SINGLE)).toBe(SINGLE);
  });

  it("leaves the legacy history exactly as it was when the move cannot be written, and says so", () => {
    localStorage.setItem(LEGACY.threads, "[legacy]");
    localStorage.setItem(LEGACY.mode, "deep");
    const undo = fullFor((key) => key.startsWith("mnemiq.ns."));
    let shelf: string | null;
    try {
      shelf = enter(SINGLE);
    } finally {
      undo();
    }

    expect(shelf).toBe(LEGACY_SHELF);
    expect(localStorage.getItem(LEGACY.threads)).toBe("[legacy]");
    expect(localStorage.getItem(LEGACY.mode)).toBe("deep");
    expect(keys().filter((key) => key.startsWith("mnemiq.ns."))).toEqual([]);
  });

  it("leaves no half-moved copy when only the second item cannot be written", () => {
    localStorage.setItem(LEGACY.threads, "[legacy]");
    localStorage.setItem(LEGACY.mode, "deep");
    const undo = fullFor((key) => key === keyFor(SINGLE, "mode"));
    try {
      expect(enter(SINGLE)).toBe(LEGACY_SHELF);
    } finally {
      undo();
    }

    expect(keys().filter((key) => key.startsWith("mnemiq.ns."))).toEqual([]);
    expect(localStorage.getItem(LEGACY.threads)).toBe("[legacy]");
  });

  it("moves it on a later load, once there is room", () => {
    localStorage.setItem(LEGACY.threads, "[legacy]");
    const undo = fullFor((key) => key.startsWith("mnemiq.ns."));
    try {
      enter(SINGLE);
    } finally {
      undo();
    }

    expect(enter(SINGLE)).toBe(SINGLE);
    expect(localStorage.getItem(keyFor(SINGLE, "threads"))).toBe("[legacy]");
    expect(localStorage.getItem(LEGACY.threads)).toBeNull();
  });

  it("moves the legacy history over an empty list rather than deleting it", () => {
    // An empty list is no history: it must never be what stops the move and lets the
    // original be deleted.
    localStorage.setItem(keyFor(SINGLE, "threads"), "[]");
    localStorage.setItem(LEGACY.threads, "[legacy]");

    enter(SINGLE);

    expect(localStorage.getItem(keyFor(SINGLE, "threads"))).toBe("[legacy]");
  });

  it("reads and writes the legacy keys themselves while the history waits to move", () => {
    expect(keyFor(LEGACY_SHELF, "threads")).toBe(LEGACY.threads);
    expect(keyFor(LEGACY_SHELF, "mode")).toBe(LEGACY.mode);
  });
});

describe("signOut", () => {
  it("purges the signing-out user's threads and mode", async () => {
    const ns = (await namespaceOf(ALICE))!;
    enter(ns);
    localStorage.setItem(keyFor(ns, "threads"), "[alice's]");
    localStorage.setItem(keyFor(ns, "mode"), "deep");

    signOut(ns);

    expect(localStorage.getItem(keyFor(ns, "threads"))).toBeNull();
    expect(localStorage.getItem(keyFor(ns, "mode"))).toBeNull();
  });

  it("touches no other namespace", async () => {
    const alice = (await namespaceOf(ALICE))!;
    const bob = (await namespaceOf(BOB))!;
    localStorage.setItem(keyFor(alice, "threads"), "[alice's]");
    localStorage.setItem(keyFor(bob, "threads"), "[bob's]");

    signOut(alice);

    expect(localStorage.getItem(keyFor(bob, "threads"))).toBe("[bob's]");
  });
});

describe("mayReload", () => {
  it("allows one reload and records when it happened", () => {
    expect(mayReload(1_000_000)).toBe(true);
    expect(mayReload(1_000_000 + RELOAD_GUARD_MS - 1)).toBe(false);
  });

  it("allows the next once the guard has passed", () => {
    expect(mayReload(1_000_000)).toBe(true);
    expect(mayReload(1_000_000 + RELOAD_GUARD_MS)).toBe(true);
  });

  it("refuses when it cannot record the reload, so a page cannot loop", () => {
    const real = Storage.prototype.setItem;
    Storage.prototype.setItem = () => {
      throw new DOMException("denied", "SecurityError");
    };
    try {
      expect(mayReload(1_000_000)).toBe(false);
    } finally {
      Storage.prototype.setItem = real;
    }
  });
});
