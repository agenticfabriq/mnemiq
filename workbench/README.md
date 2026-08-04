# mnemiq workbench

A chat workbench for the mnemiq engine. Answers arrive with the SQL that produced
them, the tables they read, and — when the data cannot support an answer — the
reason the engine declined and what to do about it.

## Run it

The workbench is served by the engine itself, so there is one process and no
separate UI server:

```
cd workbench && pnpm install && pnpm build
mnemiq serve --http          # then open http://127.0.0.1:8080
```

`pnpm build` writes the bundle to `src/mnemiq/server/static/`, inside the Python
package, so a wheel built afterwards carries the UI with it. Until you build,
`/` explains how — the API serves regardless.

## Develop

Run the engine and the Vite dev server side by side; Vite proxies `/v1` to the
engine, so the browser still sees a single origin:

```
mnemiq serve --http --port 8080     # terminal 1
pnpm dev                            # terminal 2, http://localhost:5173
```

Point the proxy elsewhere with `MNEMIQ_ENGINE_ORIGIN`.

```
pnpm test        # vitest
pnpm typecheck   # tsc --noEmit
```

## How it fits together

```
POST /v1/chat  ->  SSE, AG-UI event vocabulary
      lib/sse.ts        bytes    -> frames      split-frame safe, pure
      lib/transport.ts  frames   -> events
      lib/store.ts      events   -> messages    pure reducer
      lib/runtime.ts    messages -> assistant-ui
```

The engine's answer rides to the renderer as a `data` message part named
`mnemiq.turn`, so `components/` receives it as typed data rather than reaching
into a side channel.

`@assistant-ui/react` is pinned to an exact version on purpose: it is pre-1.0 and
moves fast. Upgrade deliberately, and run `pnpm test && pnpm typecheck` after.

## Conventions worth keeping

- **Never case-transform an identifier.** Column names, table names, mode names and
  version hashes render exactly as the engine spelled them — `CLAIM_COUNT` is a
  different string from `claim_count`. The `label` style is for words we choose;
  `meta` is for words the engine chose.
- **Never re-execute SQL to draw a table.** Results come from the `preview` the
  engine already returned; re-running would dodge the governed read path.
- **A refusal is not an error.** `deferred` and `failed` are different states with
  different colours and different next actions, because they call for different
  responses from whoever is reading.
