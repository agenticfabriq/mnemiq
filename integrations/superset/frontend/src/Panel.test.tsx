import React from "react";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { chat, host } from "./test/core";
import { Panel } from "./Panel";

const root = () => document.querySelector<HTMLElement>("[data-mnemiq-panel]")!;

describe("the panel's shell", () => {
  beforeEach(() => host.reset());
  afterEach(cleanup);

  it("floating mode sizes itself and panel mode fills the host", () => {
    render(<Panel />);
    expect(root().dataset.mnemiqPanel).toBe("floating");
    expect(root().style.width).toBe("420px");
    expect(root().style.height).toBe("640px");
    expect(root().style.maxHeight).toBe("80vh");
    act(() => chat.setDisplayMode("panel"));
    expect(root().dataset.mnemiqPanel).toBe("panel");
    expect(root().style.height).toBe("100%");
    expect(root().style.width).toBe("100%");
    expect(root().style.maxHeight).toBe("");
  });

  it("Close and Dock call the host", () => {
    host.open = true;
    render(<Panel />);
    fireEvent.click(screen.getByRole("button", { name: "Dock" }));
    expect(host.calls).toEqual(["setDisplayMode:panel"]);
    fireEvent.click(screen.getByRole("button", { name: "Float" }));
    expect(host.calls).toEqual(["setDisplayMode:panel", "setDisplayMode:floating"]);
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(host.calls.at(-1)).toBe("close");
    expect(host.open).toBe(false);
  });
});
