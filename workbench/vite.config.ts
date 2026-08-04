import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The bundle lands inside the Python package, so `mnemiq serve --http` can serve the
// workbench from the same process as the API and the browser never leaves the origin.
const OUT_DIR = "../src/mnemiq/server/static";

// In dev, Vite serves the app and forwards the API to a separately running engine.
const ENGINE = process.env.MNEMIQ_ENGINE_ORIGIN ?? "http://127.0.0.1:8080";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { outDir: OUT_DIR, emptyOutDir: true },
  server: {
    proxy: {
      // SSE must not be buffered by the dev proxy, or /v1/chat arrives all at once.
      "/v1": { target: ENGINE, changeOrigin: true },
      "/healthz": { target: ENGINE, changeOrigin: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
