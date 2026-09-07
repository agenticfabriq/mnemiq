import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SCALAR_ANSWER } from "../test/fixtures";
import type { AnswerPayload } from "../lib/types";
import { Badges } from "./Badges";

/**
 * M11's screen half. `judge_engaged` says the clusters disagreed enough to ASK the selector; the
 * selector fails closed to the majority vote on an outage, so this chip claimed a judgement had
 * happened whenever one had merely been attempted. The engine now ships `judge_fell_back` beside
 * it, and this is the surface a person actually reads.
 */
function answer(over: Partial<AnswerPayload>): AnswerPayload {
  return { ...SCALAR_ANSWER, agreement: 0.6, candidates_executed: 3, ...over };
}

describe("Badges", () => {
  it("says a judge was engaged when one actually judged", () => {
    render(<Badges answer={answer({ judge_engaged: true, judge_fell_back: false })} />);
    expect(screen.getByText("judge engaged")).toBeInTheDocument();
  });

  it("does not claim a judgement when the judge could not be reached", () => {
    render(<Badges answer={answer({ judge_engaged: true, judge_fell_back: true })} />);
    expect(screen.queryByText("judge engaged")).not.toBeInTheDocument();
    expect(screen.getByText("judge unavailable")).toBeInTheDocument();
  });

  it("still says engaged when the engine reports no verdict on the question", () => {
    // An older engine, or any surface that never learned the field: absence of the fact is not
    // evidence of a fallback, and reporting one would invent an outage.
    render(<Badges answer={answer({ judge_engaged: true, judge_fell_back: null })} />);
    expect(screen.getByText("judge engaged")).toBeInTheDocument();
  });

  it("says nothing about a judge that was never consulted", () => {
    render(<Badges answer={answer({ judge_engaged: false, judge_fell_back: null })} />);
    expect(screen.queryByText(/judge/)).not.toBeInTheDocument();
  });
});
