/** An empty screen is an invitation to act, so it hands over questions to try. */

import { Mark } from "./Mark";

/**
 * The questions come from the engine, composed from the source this deployment is actually
 * connected to and scoped to what the asker may see. They used to be three hardcoded
 * questions about claims and policyholders, so every other source opened by inviting you to
 * ask about a database that was not there (M32).
 *
 * None is a valid answer. A source with nothing worth offering gets the invitation and no
 * buttons -- quieter than three confident questions about the wrong data.
 */
export function EmptyState({
  starters,
  onPick,
}: {
  starters: string[];
  onPick: (question: string) => void;
}) {
  return (
    <section className="flex flex-col gap-4 py-6">
      <div>
        <Mark size={30} className="mb-3 text-graphite" />
        <h2 className="text-[15px] font-medium">Ask a question about your data.</h2>
        <p className="mt-1.5 max-w-lg text-[12px] text-graphite">
          Answers arrive with the SQL that produced them. When the data cannot support
          an answer, the engine says so instead of guessing.
        </p>
      </div>

      <ul className="flex flex-col items-start gap-1.5">
        {starters.map((question) => (
          <li key={question}>
            <button
              type="button"
              onClick={() => onPick(question)}
              className="border border-rule px-2.5 py-1.5 text-left text-[12px] hover:border-brass hover:text-brass"
            >
              {question}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
