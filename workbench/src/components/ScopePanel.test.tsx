import { beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { ScopePanel } from "./ScopePanel";
import type { SchemaTable } from "../lib/types";

const tables = (...names: string[]): SchemaTable[] =>
  names.map((object_id) => ({ object_id, card: "" }));

const PAGILA = tables(
  "actor", "actor_info", "address", "category", "city", "country", "customer",
  "customer_list", "film", "film_actor", "film_category", "film_list", "inventory",
  "language", "payment", "rental", "staff", "store",
);

beforeEach(() => {
  localStorage.clear();
  cleanup();
});

describe("ScopePanel filter", () => {
  it("remembers the filter across a reload", () => {
    const { unmount } = render(<ScopePanel tables={PAGILA} onHide={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("Filter tables"), { target: { value: "film" } });
    unmount();

    // A reload is a fresh mount reading the same storage.
    render(<ScopePanel tables={PAGILA} onHide={vi.fn()} />);

    expect(screen.getByLabelText("Filter tables")).toHaveValue("film");
    expect(screen.queryByText("payment")).toBeNull();
    expect(screen.getByText("film_actor")).toBeTruthy();
  });

  it("says how many of the tables it is showing while a filter is on", () => {
    // The trap persistence creates. The header read `${tables.length} tables` -- the TOTAL --
    // whatever the filter did, so a filtered panel claimed 18 tables while listing 4. Harmless
    // when the filter dies on reload; a standing lie once it survives one, because the reader
    // arrives with no memory of having typed it.
    render(<ScopePanel tables={PAGILA} onHide={vi.fn()} />);
    expect(screen.getByText("18 tables")).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Filter tables"), { target: { value: "film" } });

    expect(screen.getByText("4 of 18 tables")).toBeTruthy();
    expect(screen.queryByText("18 tables")).toBeNull();
  });

  it("keeps the filter box reachable when a stored filter narrows below the threshold", () => {
    // The box only renders above 12 tables. A filter stored against a wide scope and reloaded
    // against a narrow one would then apply invisibly -- a list silently shortened with no
    // control to clear it. Whenever a filter is active the box is shown, whatever the count.
    localStorage.setItem("mnemiq.scope-filter.v1", "st");

    // "customer" contains "st" -- a fixture that matches everything proves nothing.
    render(<ScopePanel tables={tables("store", "staff", "film")} onHide={vi.fn()} />);

    expect(screen.getByLabelText("Filter tables")).toHaveValue("st");
    expect(screen.getByText("2 of 3 tables")).toBeTruthy();
  });

  it("forgets a filter that has been cleared", () => {
    render(<ScopePanel tables={PAGILA} onHide={vi.fn()} />);
    const box = screen.getByLabelText("Filter tables");
    fireEvent.change(box, { target: { value: "film" } });
    fireEvent.change(box, { target: { value: "" } });
    cleanup();

    render(<ScopePanel tables={PAGILA} onHide={vi.fn()} />);

    expect(screen.getByLabelText("Filter tables")).toHaveValue("");
    expect(screen.getByText("18 tables")).toBeTruthy();
  });

  it("still says nothing is granted rather than nothing matched", () => {
    // Non-vacuity for the count: an empty scope and a filter that matches nothing are different
    // facts, and the fail-closed message belongs only to the first.
    render(<ScopePanel tables={[]} onHide={vi.fn()} />);
    expect(screen.getByText(/No tables are granted/)).toBeTruthy();
  });
});
