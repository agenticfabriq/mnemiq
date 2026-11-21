import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

// `pnpm test` runs the unit tests, `@apache-superset/core` being the fake host in `src/test/core.ts`; `pnpm
// test:bundle` (MNEMIQ_PANEL_BUNDLE set) runs only `src/bundle.test.ts`, against the built `dist/`, where the host's
// modules come from the share scope the test builds, as Superset's loader builds it.
const bundle = Boolean(process.env.MNEMIQ_PANEL_BUNDLE);

export default defineConfig({
  resolve: {
    alias: { "@apache-superset/core": fileURLToPath(new URL("./src/test/core.ts", import.meta.url)) },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["src/test/setup.ts"],
    include: bundle ? ["src/bundle.test.ts"] : ["src/**/*.test.{ts,tsx}"],
    exclude: bundle ? [] : ["src/bundle.test.ts"],
  },
});
