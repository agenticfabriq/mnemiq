import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { DEFERRAL } from "../test/fixtures";
import type { AnswerPayload, DeferralReason } from "../lib/types";
import { REASONS } from "../lib/verdict";
import { DeferralCard } from "./DeferralCard";

const REASON_CODES = Object.keys(REASONS) as DeferralReason[];

const withReason = (code: DeferralReason, failed = false): AnswerPayload => ({
  ...DEFERRAL,
  reason_code: code,
  deferred: !failed,
  failed,
});

describe("DeferralCard", () => {
  it("renders the engine's own stated reason", () => {
    render(<DeferralCard answer={DEFERRAL} />);
    expect(screen.getByText(/claim_status_code, is entirely NULL/)).toBeInTheDocument();
  });

  it.each(REASON_CODES)("gives %s a title and a next action", (code) => {
    render(<DeferralCard answer={withReason(code)} />);
    expect(screen.getByText(REASONS[code].title)).toBeInTheDocument();
    expect(screen.getByText(REASONS[code].next)).toBeInTheDocument();
    expect(screen.getByText(code)).toBeInTheDocument();
  });

  it("labels a deferral and a failure differently, and does not blame the source for either", () => {
    const { unmount } = render(<DeferralCard answer={withReason("unanswerable")} />);
    expect(screen.getByLabelText("Deferral")).toBeInTheDocument();
    unmount();

    // Three codes ride `failed` and only one of them IS the source. The label said
    // "Source failure" over a model-provider outage before a verifier outage could reach it
    // too, so the region is named for the kind and the heading names which one broke.
    for (const code of ["execution_failed", "model_unavailable", "verifier_unavailable"] as const) {
      const view = render(<DeferralCard answer={withReason(code, true)} />);
      expect(screen.getByLabelText("Failure")).toBeInTheDocument();
      expect(screen.queryByLabelText("Source failure")).toBeNull();
      view.unmount();
    }
  });

  it("still renders when the engine sends no reason code", () => {
    render(<DeferralCard answer={{ ...DEFERRAL, reason_code: null }} />);
    expect(screen.getByText("No answer")).toBeInTheDocument();
  });
});
