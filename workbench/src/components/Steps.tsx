/**
 * What the engine is doing, phase by phase.
 *
 * Stages get human labels rather than the engine's internal names: this is the one
 * place in the interface where the reader is not being shown data, they are being told
 * what is happening, so it reads as a sentence rather than an identifier.
 *
 * While a run is open this is the progress display. Once it closes the same list is the
 * timing breakdown, which is the only place that breakdown exists -- the engine records
 * one total, not a per-phase split.
 */

import type { Step } from "../lib/types";
import { duration } from "../lib/verdict";

const LABEL: Record<string, string> = {
  retrieve: "Finding tables in scope",
  plan: "Writing SQL",
  candidate: "Generating a candidate",
  execute: "Running the query",
  verify: "Checking the result",
  synthesize: "Writing the answer",
};

function detail(step: Step): string | null {
  if (step.of === undefined || step.of <= 1) return null;
  const n = step.index ?? step.attempt;
  return n === undefined ? null : `${n} of ${step.of}`;
}

function Row({ step }: { step: Step }) {
  const mark =
    step.status === "running" ? "·" : step.status === "failed" ? "✕" : "✓";
  const tone =
    step.status === "failed"
      ? "text-crimson"
      : step.status === "running"
        ? "text-brass"
        : "text-graphite";

  return (
    <li className="flex items-baseline gap-2">
      <span
        className={`w-2 shrink-0 ${tone} ${step.status === "running" ? "animate-pulse" : ""}`}
        aria-hidden="true"
      >
        {mark}
      </span>
      <span className={step.status === "running" ? "text-ink" : "text-graphite"}>
        {LABEL[step.name] ?? step.name}
      </span>
      {detail(step) && <span className="text-graphite/70">{detail(step)}</span>}
      <span className="tabular ml-auto text-graphite/70">
        {step.ms === undefined ? "" : duration(step.ms)}
      </span>
    </li>
  );
}

export function Steps({ steps, live }: { steps: Step[]; live?: boolean }) {
  if (steps.length === 0) return null;
  return (
    <ol
      className="meta flex max-w-sm flex-col gap-1"
      {...(live ? { role: "status", "aria-live": "polite" as const } : {})}
    >
      {steps.map((step) => (
        <Row key={step.id} step={step} />
      ))}
    </ol>
  );
}
