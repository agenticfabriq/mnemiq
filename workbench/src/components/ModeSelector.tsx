/** instant | thinking | deep -- the engine's own modes, spent effort per question. */

import { MODES, type Mode } from "../lib/types";

const EFFORT: Record<Mode, string> = {
  instant: "One candidate, no correction. Fastest.",
  thinking: "One candidate with SQL correction. The default.",
  deep: "Five candidates, judged and verified. Slowest.",
};

export function ModeSelector({
  mode,
  onChange,
}: {
  mode: Mode;
  onChange: (mode: Mode) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="label hidden sm:inline">Mode</span>
      <div role="radiogroup" aria-label="Mode" className="flex border border-rule">
        {MODES.map((option) => (
          <button
            key={option}
            type="button"
            role="radio"
            aria-checked={mode === option}
            title={EFFORT[option]}
            onClick={() => onChange(option)}
            className={`label border-r border-rule px-2 py-1 last:border-r-0 ${
              mode === option ? "bg-ink text-paper" : "hover:text-ink"
            }`}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
}
