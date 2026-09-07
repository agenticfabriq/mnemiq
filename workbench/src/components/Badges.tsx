/**
 * Signals that only exist in some modes. Each renders only when the engine actually
 * reported it -- `agreement`, `judge_*` and `candidates_executed` are null outside
 * multi-candidate runs, and an absent signal must not read as a zero.
 *
 * Only `judge_override` gets the reserved hue: it is the one signal here that
 * changed the outcome.
 */

import type { AnswerPayload } from "../lib/types";

function Chip({ children, accent }: { children: React.ReactNode; accent?: boolean }) {
  return (
    <span
      className={`label border px-1.5 py-0.5 ${
        accent ? "border-brass/50 text-brass" : "border-rule"
      }`}
    >
      {children}
    </span>
  );
}

export function Badges({ answer }: { answer: AnswerPayload }) {
  const chips: React.ReactNode[] = [];

  if (answer.cached) chips.push(<Chip key="cached">cached</Chip>);
  if (answer.agreement !== null) {
    chips.push(<Chip key="agree">agreement {answer.agreement.toFixed(2)}</Chip>);
  }
  if (answer.candidates_executed !== null) {
    chips.push(<Chip key="cand">{answer.candidates_executed} candidates</Chip>);
  }
  // `judge_engaged` says the judge was ASKED. The selector fails closed to the majority vote, so
  // a chip reading "judge engaged" was the same words whether one judged or the endpoint was
  // down -- and this is the surface a person reads. `null` is not a fallback: an engine that does
  // not report the fact must not be rendered as an outage.
  if (answer.judge_engaged) {
    chips.push(
      <Chip key="judge">{answer.judge_fell_back ? "judge unavailable" : "judge engaged"}</Chip>,
    );
  }
  if (answer.judge_override) {
    chips.push(
      <Chip key="override" accent>
        judge overrode the majority
      </Chip>,
    );
  }

  if (chips.length === 0) return null;
  return <div className="flex flex-wrap items-center gap-1.5">{chips}</div>;
}
