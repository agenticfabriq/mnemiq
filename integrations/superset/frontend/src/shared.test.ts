import { readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

// The workbench modules the panel shares (through `src/shared.ts`). The panel runs on the host's React 18 and the
// workbench on its own React 19, so these must import nothing but each other: no React, no library at all.
const SHARED = ["store", "types", "sse", "history", "verdict"];
// Paths from the package root, where vitest runs: under jsdom `import.meta.url` is not a file URL.
const LIB = resolve(process.cwd(), "../../../workbench/src/lib");

/** The import specifiers of `source` that are not one of the shared modules. */
function foreignImports(source: string): string[] {
  const specifiers = [...source.matchAll(/^\s*(?:import|export)\b[^'"]*?\bfrom\s*["']([^"']+)["']|^\s*import\s*["']([^"']+)["']/gm)]
    .map((m) => m[1] ?? m[2]);
  return specifiers.filter((s) => !SHARED.some((name) => s === `./${name}`));
}

describe("the shared modules", () => {
  it("the shared modules import only each other", () => {
    for (const name of SHARED) {
      const source = readFileSync(resolve(LIB, `${name}.ts`), "utf8");
      expect(foreignImports(source), name).toEqual([]);
    }
    // The planted twin: a shared module that reached for React is caught.
    expect(foreignImports('import type { Turn } from "./store";\nimport { useState } from "react";\n'))
      .toEqual(["react"]);
  });

  it("src/shared.ts is the only file that reaches into the workbench", () => {
    const reaching = (source: string) => /from\s*["'][^"']*workbench\/src\//.test(source);
    const files = readdirSync(resolve(process.cwd(), "src"), { recursive: true, encoding: "utf8" })
      .filter((f) => /\.tsx?$/.test(f) && !/\.test\.tsx?$/.test(f));
    expect(files.filter((f) => reaching(readFileSync(resolve(process.cwd(), "src", f), "utf8")))).toEqual(["shared.ts"]);
    // The planted twin: a panel file that imports the workbench directly is caught.
    expect(reaching('import { reduce } from "../../../../workbench/src/lib/store";')).toBe(true);
  });
});
