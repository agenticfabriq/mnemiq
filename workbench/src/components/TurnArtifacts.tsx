/**
 * Everything the engine returned alongside the prose, rendered from the
 * `mnemiq.turn` data part.
 *
 * A deferral carries its own text inside the card, so the prose part is dropped
 * upstream in `convertMessage` -- the reason is stated once, in the place that
 * explains what to do about it.
 */

import { useEffect, useState } from "react";
import type { DataMessagePartProps } from "@assistant-ui/react";

import type { Turn } from "../lib/store";
import type { Step } from "../lib/types";
import { duration } from "../lib/verdict";
import { Steps } from "./Steps";
import { Badges } from "./Badges";
import { DeferralCard } from "./DeferralCard";
import { ResultTable } from "./ResultTable";
import { RunDetails } from "./RunDetails";
import { SqlBlock } from "./SqlBlock";

/**
 * The engine answers in one shot -- there are no tokens to stream -- so progress is
 * the phases it reports plus how long it has been going. Deep mode runs for minutes;
 * a bare spinner would say nothing about that.
 */
function Working({ steps }: { steps: Step[] }) {
  const [elapsed, setElapsed] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const timer = globalThis.setInterval(() => setElapsed(Date.now() - started), 250);
    return () => globalThis.clearInterval(timer);
  }, []);

  return (
    <div className="flex flex-col gap-1.5">
      <Steps steps={steps} live />
      <p className="meta tabular">Working… {duration(elapsed)}</p>
    </div>
  );
}

export function TurnArtifacts({ data }: DataMessagePartProps<Turn>) {
  const turn = data;

  if (turn.status === "error") {
    return (
      <section className="border-l-2 border-crimson/45 bg-crimson-wash px-3.5 py-3">
        <h3 className="text-[13px] font-medium text-crimson">The run did not finish</h3>
        <p className="mt-1.5 text-[12px]">{turn.error}</p>
        <p className="label mt-2.5 normal-case tracking-normal text-[11px]">
          → The engine is still serving. Ask again.
        </p>
      </section>
    );
  }

  const answer = turn.answer;
  const steps = turn.steps ?? [];
  if (!answer) {
    return turn.status === "running" ? <Working steps={steps} /> : null;
  }

  const refused = answer.deferred || answer.failed;

  return (
    <div className="flex flex-col gap-2.5">
      {refused && <DeferralCard answer={answer} />}
      {answer.sql && <SqlBlock sql={answer.sql} />}
      {answer.preview && <ResultTable preview={answer.preview} />}
      <Badges answer={answer} />
      <RunDetails answer={answer} steps={steps} />
    </div>
  );
}
