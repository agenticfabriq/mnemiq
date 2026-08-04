/**
 * The provenance of one answer, and where its time went.
 *
 * The phase breakdown comes from the step events, not from the trace: the engine
 * records one total and the execution time, so this list is the only place the split
 * exists. It is also usually a surprise -- writing the SQL dominates, and running it
 * is noise.
 */

import type { AnswerPayload, Step } from "../lib/types";
import { duration } from "../lib/verdict";
import { Steps } from "./Steps";

function Row({ term, children }: { term: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 py-1">
      <dt className="label w-28 shrink-0 pt-0.5">{term}</dt>
      <dd className="min-w-0 text-[12px]">{children}</dd>
    </div>
  );
}

export function RunDetails({
  answer,
  steps = [],
}: {
  answer: AnswerPayload;
  steps?: Step[];
}) {
  const execute = duration(answer.timing?.["execute_ms"]);
  const total = duration(answer.timing?.["total_ms"]);
  const tables = answer.tables_used ?? [];
  if (tables.length === 0 && !total && !answer.enrichment_version && steps.length === 0) {
    return null;
  }

  return (
    <details className="border border-rule">
      <summary className="label cursor-pointer px-2.5 py-1.5 hover:text-ink">
        Run details
      </summary>
      <dl className="border-t border-rule px-2.5 py-1.5">
        {steps.length > 0 && (
          <Row term="Phases">
            <Steps steps={steps} />
          </Row>
        )}
        {tables.length > 0 && (
          <Row term="Tables read">
            <span className="break-all">{tables.join(", ")}</span>
          </Row>
        )}
        {total && (
          <Row term="Elapsed">
            <span className="tabular">
              {total}
              {execute && <span className="text-graphite"> · {execute} in the source</span>}
            </span>
          </Row>
        )}
        {answer.enrichment_version && (
          <Row term="Enrichment">
            <span className="tabular">{answer.enrichment_version}</span>
          </Row>
        )}
      </dl>
    </details>
  );
}
