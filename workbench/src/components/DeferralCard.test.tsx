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

  it("labels a deferral and a source failure differently", () => {
    const { unmount } = render(<DeferralCard answer={withReason("unanswerable")} />);
    expect(screen.getByLabelText("Deferral")).toBeInTheDocument();
    unmount();

    render(<DeferralCard answer={withReason("execution_failed", true)} />);
    expect(screen.getByLabelText("Source failure")).toBeInTheDocument();
  });

  it("still renders when the engine sends no reason code", () => {
    render(<DeferralCard answer={{ ...DEFERRAL, reason_code: null }} />);
    expect(screen.getByText("No answer")).toBeInTheDocument();
  });
});
