// The built panel (`dist/`, after `pnpm build`), loaded as Superset 7.0.0rc2 loads an extension:
// `ExtensionsLoader.loadModule` in superset-frontend/src/extensions/ExtensionsLoader.ts appends a <script> for the
// remote entry and waits for it to load, then calls `container.init(customScope)` with a share scope whose entries are
// `{get, loaded, eager}`, then `container.get("./index")` and the factory. Here each <script> the page is given (the
// remote entry, then the chunk it asks for) is answered from `dist/` and evaluated in the page, as the browser would.
import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import React from "react";
import * as ReactDOM from "react-dom";
import { createRoot } from "react-dom/client";
import * as antd from "antd";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import * as core from "./test/core";

const DIST = resolve(process.cwd(), "dist");
const ROOT = resolve(process.cwd(), "../../..");
// Where rc2 serves an extension's files: `/api/v1/extensions/<publisher>/<name>/<file>`.
const BASE = "http://superset.test/api/v1/extensions/agenticfabriq/mnemiq/";
const files = () => readdirSync(DIST);
const scripts = () => files().filter((f) => f.endsWith(".js")).map((f) => readFileSync(join(DIST, f), "utf8"));

/** Answers a <script> under `BASE` from `dist/`, evaluated in the page (an indirect eval runs it as a classic script
 * does, in the global scope), then fires its load event; anything else is left as jsdom leaves it. */
function serveDist(): () => void {
  const append = document.head.appendChild.bind(document.head);
  document.head.appendChild = <T extends Node>(node: T): T => {
    const added = append(node);
    if (node instanceof HTMLScriptElement && node.src.startsWith(BASE)) {
      const source = readFileSync(join(DIST, node.src.slice(BASE.length)), "utf8");
      setTimeout(() => {
        (0, eval)(source);
        node.dispatchEvent(new Event("load"));
      }, 0);
    }
    return added;
  };
  return () => void (document.head.appendChild = append);
}

type Container = { init(scope: object): Promise<void> | void; get(module: string): Promise<() => unknown> };
const asked = { react: 0 };

/** What `ExtensionsLoader.loadModule` does for this extension, the host's modules given through the share scope. */
async function loadAsSuperset(): Promise<void> {
  const [entry] = files().filter((f) => /^remoteEntry\.[0-9a-f]{8,}\.js$/.test(f));
  await new Promise<void>((done, failed) => {
    const element = document.createElement("script");
    element.src = BASE + entry;
    element.onload = () => done();
    element.onerror = () => failed(new Error("the remote entry did not load"));
    document.head.appendChild(element);
  });
  const container = (globalThis as Record<string, unknown>).agenticfabriq_mnemiq as Container;
  expect(typeof container?.init).toBe("function");
  const shared = (version: string, module: unknown, count?: () => void) => ({
    [version]: { get: () => { count?.(); return Promise.resolve(() => module); }, loaded: true, eager: true },
  });
  await container.init({
    react: shared("18.3.1", React, () => asked.react++),
    "react-dom": shared("18.3.1", ReactDOM),
    antd: shared("6.6.8", antd),
    "@apache-superset/core": shared("0.2.0", { chat: core.chat, navigation: core.navigation,
      authentication: core.authentication }),
  });
  const factory = await container.get("./index");
  factory();
}

/** Whether `source` carries a React of its own: React's internals, which only React's own code names. */
const carriesReact = (source: string) => source.includes("__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED");
/** Whether `source` evaluates a string as code. */
const evaluatesStrings = (source: string) => /\beval\(|\bnew Function\(/.test(source);

describe("the built panel", () => {
  let restore = () => {};

  beforeAll(async () => {
    core.host.reset();
    restore = serveDist();
    await loadAsSuperset();
  });
  afterAll(() => restore());

  it("the container registers the chat when loaded as Superset 7.0.0rc2 loads it", () => {
    expect(core.host.registered.map((r) => r.chat.id)).toEqual(["agenticfabriq.mnemiq"]);
  });

  it("the registered panel renders from the bundle with the host's React 18", async () => {
    (globalThis as Record<string, unknown>).IS_REACT_ACT_ENVIRONMENT = true;
    const [{ panel: Panel }] = core.host.registered;
    const mount = document.createElement("div");
    document.body.appendChild(mount);
    const root = createRoot(mount);
    await React.act(async () => root.render(React.createElement(Panel)));
    expect(mount.querySelector("[data-mnemiq-panel]")?.textContent).toContain("mnemiq");
    expect(asked.react).toBeGreaterThan(0);
    await React.act(async () => root.unmount());
  });

  it("the bundle carries no React of its own", () => {
    expect(scripts().filter(carriesReact)).toEqual([]);
    // The control: a planted copy of React's internals is caught.
    expect(carriesReact(readFileSync(resolve(process.cwd(), "node_modules/react/cjs/react.production.min.js"), "utf8"))).toBe(true);
  });

  it("no emitted file evaluates strings as code", () => {
    expect(scripts().filter(evaluatesStrings)).toEqual([]);
    expect(evaluatesStrings("var f = new Function('return 1');")).toBe(true);
    expect(evaluatesStrings("x = eval(code);")).toBe(true);
  });

  it("every emitted file name carries a content hash, and no CSS is emitted", () => {
    expect(files().length).toBeGreaterThan(1);
    expect(files().filter((f) => !/\.[0-9a-f]{8,}\./.test(f))).toEqual([]);
    expect(files().filter((f) => /\.css$/.test(f) || f.endsWith(".map"))).toEqual([]);
  });

  it("the extension's packer packs the built panel", () => {
    const out = mkdtempSync(join(tmpdir(), "mnemiq-panel-"));
    execFileSync("python3", [join(ROOT, "scripts/build_superset_extension.py"), "--out", out, "--require-frontend"],
      { cwd: ROOT, stdio: "pipe" });
    const manifest = JSON.parse(execFileSync("python3", ["-c",
      "import sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); print(z.read('manifest.json').decode())",
      readdirSync(out).map((f) => join(out, f)).find((f) => f.endsWith(".supx"))!]).toString());
    expect(manifest.frontend.moduleFederationName).toBe("agenticfabriq_mnemiq");
    expect(files()).toContain(manifest.frontend.remoteEntry);
  });
});
