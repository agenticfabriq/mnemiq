/**
 * The mnemiq mark: a Q inside brackets.
 *
 * The letter is the house (Agentic Fabriq, whose products share it); the brackets are
 * this product -- scope closing on what an identity may see, which is the engine's job.
 * Squared and mitred like everything else here; the tail is the one diagonal in the
 * system, which is what makes it a Q rather than a box.
 *
 * Strokes are `currentColor`, so it takes the ink of whatever it sits in and needs no
 * light and dark variants. `public/favicon.svg` carries the same geometry with explicit
 * values, because a favicon has no inherited colour to take.
 */

export function Mark({ size = 20, className }: { size?: number; className?: string }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      className={className}
      role="img"
      aria-label="mnemiq"
      fill="none"
      stroke="currentColor"
      strokeWidth={3.2}
      strokeLinecap="butt"
      strokeLinejoin="miter"
    >
      <path d="M10 4H4v24h6M22 4h6v24h-6" />
      <rect x="11.6" y="9.6" width="9" height="10.6" />
      <path d="M17 17.4l5.2 5.6" />
    </svg>
  );
}
