import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CONNECTION_ENDED } from "./store";
import type { AguiEvent } from "./types";

vi.mock("./transport", async (original) => {
  const actual = await original<typeof import("./transport")>();
  return { ...actual, streamChat: vi.fn() };
});

const { streamChat } = await import("./transport");
const { useWorkbench } = await import("./runtime");

function mounted() {
  return renderHook(() => useWorkbench(null, () => {}, async () => true));
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
