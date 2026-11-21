import React from "react";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { host } from "./test/core";
import { resetStore, update } from "./store";
import { Trigger } from "./Trigger";

describe("the trigger", () => {
  beforeEach(() => {
    host.reset();
    resetStore();
  });
  afterEach(cleanup);

  it("the trigger toggles the panel and shows a running build", () => {
    render(<Trigger />);
    const button = screen.getByRole("button", { name: "mnemiq" });
    fireEvent.click(button);
    expect(host.calls).toEqual(["open"]);
    fireEvent.click(button);
    expect(host.calls).toEqual(["open", "close"]);
    expect(screen.queryByTestId("mnemiq-running")).toBeNull();
    act(() => update((state) => ({ ...state, runningBuild: "0f6d0c1e-6b8f-4a39-9a4e-3d5f5a9c2b10" })));
    expect(screen.getByTestId("mnemiq-running")).toBeTruthy();
    expect(screen.getByRole("button", { name: "mnemiq, a build is running" })).toBeTruthy();
    act(() => update((state) => ({ ...state, runningBuild: null })));
    expect(screen.queryByTestId("mnemiq-running")).toBeNull();
  });
});
