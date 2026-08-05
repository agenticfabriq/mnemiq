/**
 * The mnemiq mark: a q inside brackets.
 *
 * The letter is the house (Agentic Fabriq, whose products share it); the brackets are
 * this product -- scope closing on what an identity may see, which is the engine's job.
 * Lowercase to match the wordmark and the rest of the type system, and squared like
 * everything else here: the bowl is drawn as one stroke whose right side keeps going to
 * become the descender, so the letter is a single continuous path.
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
      <path d="M19.4 19.6h-8V9.4h8v16.4" />
    </svg>
  );
}
