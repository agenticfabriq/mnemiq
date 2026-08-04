/**
 * A refusal, rendered as a verdict rather than an error.
 *
 * `failed` is a separate state from `deferred` on purpose: a source outage is
 * something that happened to the engine, a deferral is a decision it made. The two
 * get different hues and different next actions.
 */

import type { AnswerPayload, DeferralReason } from "../lib/types";
import { REASONS } from "../lib/verdict";

const UNKNOWN = {
  title: "No answer",
  next: "The engine declined without a recognised reason code.",
};

export function DeferralCard({ answer }: { answer: AnswerPayload }) {
  const failed = answer.failed;
  const code = answer.reason_code as DeferralReason | null;
  const copy = (code && REASONS[code]) || UNKNOWN;

  const edge = failed ? "border-crimson/45" : "border-brass/45";
  const wash = failed ? "bg-crimson-wash" : "bg-brass-wash";
  const accent = failed ? "text-crimson" : "text-brass";

  return (
    <section
      className={`border-l-2 ${edge} ${wash} px-3.5 py-3`}
      aria-label={failed ? "Source failure" : "Deferral"}
    >
      <div className="flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
        <h3 className={`text-[13px] font-medium ${accent}`}>{copy.title}</h3>
        {code && <code className="meta">{code}</code>}
      </div>

      <p className="prose-answer mt-2 text-[14px]">{answer.answer}</p>

      <p className="label mt-2.5 flex gap-1.5">
        <span aria-hidden="true">→</span>
        <span className="normal-case tracking-normal text-[11px]">{copy.next}</span>
      </p>
    </section>
  );
}
