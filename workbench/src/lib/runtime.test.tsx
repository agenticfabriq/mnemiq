import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { keyFor } from "./identity";
import { CONNECTION_ENDED } from "./store";
import type { AguiEvent } from "./types";

vi.mock("./transport", async (original) => {
  const actual = await original<typeof import("./transport")>();
  return { ...actual, streamChat: vi.fn() };
});

const { streamChat } = await import("./transport");
const { useWorkbench } = await import("./runtime");

function mounted(shelf: string | null = null, confirm: () => Promise<boolean> = async () => true) {
  return renderHook(() => useWorkbench(shelf, () => {}, confirm));
}

function lastTurn(result: ReturnType<typeof mounted>["result"]) {
  const active = result.current.threads.find((t) => t.id === result.current.activeId)!;
  return active.turns.at(-1)!;
}

afterEach(() => {
  vi.mocked(streamChat).mockReset();
});

describe("an answer whose stream stops early", () => {
  it("a stream that starts a run and ends leaves the turn an error, never running", async () => {
    vi.mocked(streamChat).mockImplementation(async function* (): AsyncGenerator<AguiEvent> {
      yield { type: "RUN_STARTED", runId: "r1" };
    });
    const { result } = mounted();
    await act(async () => {
      await result.current.ask("how many claims?");
    });
    expect(lastTurn(result).status).toBe("error");
    expect(lastTurn(result).error).toBe(CONNECTION_ENDED);
  });

  it("a stream that breaks after the run started does the same, with no unhandled rejection", async () => {
    vi.mocked(streamChat).mockImplementation(async function* (): AsyncGenerator<AguiEvent> {
      yield { type: "RUN_STARTED", runId: "r1" };
      throw new TypeError("network error");
    });
    const { result } = mounted();
    await act(async () => {
      await result.current.ask("how many claims?");
    });
    expect(lastTurn(result).status).toBe("error");
    expect(lastTurn(result).error).toBe(CONNECTION_ENDED);
  });
});

describe("the end of an ask", () => {
  it("a stream that yields nothing and ends gives the question an error turn", async () => {
    vi.mocked(streamChat).mockImplementation(async function* (): AsyncGenerator<AguiEvent> {});
    const { result } = mounted();
    await act(async () => {
      await result.current.ask("how many claims?");
    });
    expect(lastTurn(result).role).toBe("assistant");
    expect(lastTurn(result).status).toBe("error");
    expect(lastTurn(result).error).toBe(CONNECTION_ENDED);
  });

  it("a second ask while one runs in the thread sends nothing and fails nothing", async () => {
    let release: () => void = () => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    vi.mocked(streamChat).mockImplementation(async function* (): AsyncGenerator<AguiEvent> {
      yield { type: "RUN_STARTED", runId: "r1" };
      await held;
      yield { type: "RUN_FINISHED", runId: "r1" };
    });
    const { result } = mounted();
    let first: Promise<void> = Promise.resolve();
    await act(async () => {
      first = result.current.ask("first");
      await Promise.resolve();
    });
    await act(async () => {
      await result.current.ask("second");
    });
    expect(vi.mocked(streamChat)).toHaveBeenCalledTimes(1);
    await act(async () => {
      release();
      await first;
    });
    expect(lastTurn(result).status).toBe("complete");
  });

  it("a thread saved while an answer ran comes back with that answer ended", () => {
    const shelf = "u-test";
    const running = { id: "a1", role: "assistant", text: "", status: "running" };
    localStorage.setItem(keyFor(shelf, "threads"), JSON.stringify([
      { id: "t1", title: "q", turns: [{ id: "u1", role: "user", text: "q", status: "complete" }, running] },
    ]));
    const { result } = mounted(shelf);
    expect(lastTurn(result).status).toBe("error");
    expect(lastTurn(result).error).toBe(CONNECTION_ENDED);
    localStorage.clear();
  });
});
