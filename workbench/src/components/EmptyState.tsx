/** An empty screen is an invitation to act, so it hands over questions to try. */

// One aggregate, one grouping, and one the engine should decline -- a refusal is as
// much a demonstration of the engine as an answer is.
const STARTERS = [
  "Which region has the highest total claim amount?",
  "How many claims are there by status?",
  "List five policyholders by name with their policy number",
];

export function EmptyState({ onPick }: { onPick: (question: string) => void }) {
  return (
    <section className="flex flex-col gap-4 py-6">
      <div>
        <h2 className="text-[15px] font-medium">Ask a question about your data.</h2>
        <p className="mt-1.5 max-w-lg text-[12px] text-graphite">
          Answers arrive with the SQL that produced them. When the data cannot support
          an answer, the engine says so instead of guessing.
        </p>
      </div>

      <ul className="flex flex-col items-start gap-1.5">
        {STARTERS.map((question) => (
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
