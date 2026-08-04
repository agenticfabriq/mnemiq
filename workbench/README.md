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

## Layout

History on the left, transcript in the middle, the tables in scope on the right
(hidable — the toggle sits in the header, and the choice is remembered).

History lives in this browser's `localStorage`, which is what makes it survive a
reload without inventing a server-side store. Two consequences worth knowing: it
is per browser rather than per principal, and it is **not** an audit record of
what the engine was asked — the engine's own answer log is.

## Streaming: stages, not tokens

The engine streams **progress**, not text. `/v1/chat` emits `STEP_STARTED` /
`STEP_FINISHED` frames as the answer moves through its phases, then delivers the
finished answer in a single `TEXT_MESSAGE_CONTENT`.

That split is deliberate and measured. Timing a real request against a live store:
SQL generation took 63% of the wall clock, correction 17%, **synthesis 11%** and
execution 0.07%. Token-streaming the synthesis would animate that last 11% and
reveal about a hundred characters — and a deep-mode run that defers never
synthesises at all, so it would show nothing. Stages cover the whole wait.

The mechanism is one `asyncio.Queue` per request. The engine is synchronous and
runs on a worker thread; it reports progress through a plain synchronous `emit`
callback (`mnemiq/progress.py`) whose body only schedules a thread-safe put. The
queue is then the single ordering point for everything the client sees, and the
completion itself is queued as a sentinel — so a step emitted before the answer
can never arrive after it. If token deltas are ever wanted, they become a second
producer on the same queue rather than a new mechanism.

Every step carries a `stepId` on both halves. Matching a finish to its start by
name would break the moment two steps of the same name overlap, which is exactly
what deep mode's candidates would do if they were ever run concurrently.

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
