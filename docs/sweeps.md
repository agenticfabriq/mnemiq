# Sweeps

`mnemiq sweep` finds the settings that suit a model on your data by measuring them. You give it a question set with
known answers, the model endpoints to tune, and the values to try for each setting. It runs each candidate set of
values as one **arm** beside the model's **start** (the profile it runs today, or the engine's defaults), grades every
answer by the data it returns, and proposes one [profile](profiles.md) per model, with the evidence behind it. A
person reads the evidence and decides whether to promote the profile. Nothing is promoted for you.

The words are the ones [settings.md](settings.md) uses: a **technique** is an idea about answering well, a
**setting** is one named value, a **profile** is a named set of setting values, a **mode** is `instant`, `thinking`
or `deep`, and an **arm** is one profile run in one mode: the start itself, or the start changed by one or more
settings.

## The flow

1. Write a question file: questions people ask of this data, each with the SQL that answers it, or marked as one the
   engine should decline ([the question file](#the-question-file)).
2. Write a config file naming that file, a deployment, the models and the values to try
   ([the config file](#the-config-file)).
3. Set each key variable the config names, for example `export LOCAL_LLM_KEY=...`. A server that takes no key still
   needs the variable to hold some value. No key is ever written in a file. `run` refuses to start, naming the
   variable, while one is not set.
4. Read the plan, which runs nothing:

   ```bash
   mnemiq sweep plan --config sweep.toml
   ```

   It lists the arms of each round, the arms the engine refuses (they are skipped), the questions it will ask, the
   hours and dollars when it can price them, the store copy each running arm makes, and the smallest gain the search
   could show. When that gain needs more questions than you have, it says so, and check mode is the run to make.
5. Run a smoke: each arm of the first round once, on its first few questions, to see that every arm runs.

   ```bash
   mnemiq sweep run --config sweep.toml --smoke 3
   ```

   It writes only to `<out>/smoke/`, which nothing else reads. After a smoke, `plan` prices the sweep from the tokens
   the smoke's questions took, for each model with a price.
6. Run the sweep. It runs every round in order and can take hours; stopped, it resumes where it stopped.

   ```bash
   mnemiq sweep run --config sweep.toml
   ```

7. Read the rounds, then propose:

   ```bash
   mnemiq sweep report --config sweep.toml
   mnemiq sweep propose --config sweep.toml
   ```

8. Read `<out>/proposal/<model>/evidence.json`, and promote `<out>/proposal/<model>/profiles.toml` if it holds up
   ([promoting a proposal](#promoting-a-proposal)).

## The commands

Each action takes `--config`, the sweep's config file.

| Command | What it does |
|---|---|
| `mnemiq sweep plan` | Says what the sweep will run, what it will cost and the smallest gain it could show. Runs nothing and calls no model. |
| `mnemiq sweep run` | Runs the rounds in order, resuming where an earlier run stopped. `--until ROUND` stops after that round. `--smoke N` runs each arm of the first round once on its first N questions to tune on, in `<out>/smoke/`, instead of the sweep. |
| `mnemiq sweep report` | Prints a table for each model and mode of every round that has run, or of one with `--round ROUND`. |
| `mnemiq sweep propose` | Applies the config's rule to the finished sweep and writes each model's proposed profile and its evidence. `--choose ARM` writes a person's pick instead: an arm's label as the report shows it, `start`, or in check mode `defaults`. One pick applies to every model, and must name an arm that ran and is valid in every run there. |

A command exits 0 when it did what it was asked. It exits 1 with the reason, and no traceback, when the config or
the sweep's files refuse it, and when `run` stopped because an endpoint stopped answering or a worker died. A `run`
that the budget stopped says so and exits 3, so a script can tell it from one that finished: raise the budget and run
again, and the sweep resumes where it stopped.

Under `repeats = "auto"`, `report` reads each arm at the number of runs the first round decided, which the sweep
records. Before that is decided, or when no start ran, it reads each arm at one run and says so.

## The question file

A JSON list of cases. Each has an `id` (unique), a `question`, and either `gold_sql`, the query whose result is the
right answer, or `"answerable": false` for a question the data cannot answer, which the engine should decline. A
decline there counts as Declined, never as Right, and an answer to it counts as Made up. With several deployments,
each case's `db_id` names the one it reads.

```json
[
  {"id": "open-orders", "question": "How many orders are still open?",
   "gold_sql": "SELECT COUNT(*) FROM orders WHERE status = 'open'"},
  {"id": "order-margin", "question": "What margin did each order make?", "answerable": false}
]
```

An answer is graded by the rows it returns, never by its SQL's spelling: any query that returns the gold rows is
right. The questions are what the sweep tunes to, so they should be the ones people ask, in their words, including some
the data cannot answer. How many: an arm must win at least a handful of questions the start gets wrong
([what a sweep can show](#what-a-sweep-can-show)), so a search needs a few dozen answerable questions to tune on that
the start misses. With fewer, `plan` says so, and check mode is the run to make.

## The config file

A TOML file. Paths in it are read from the file's own folder. Paths inside a deployment's env file are read as the
engine reads them, from the working directory.

```toml
modes = ["instant", "thinking"]

[sweep]
name = "orders"
out = "sweeps/orders"

[dataset]
questions = "questions.json"
env = "deployment.env"
split_seed = 20261107

[dataset.identity]
roles = ["analyst"]

[models.local]
llm_model = "qwen2.5-coder-32b"
llm_base_url = "http://127.0.0.1:8000/v1"
api_key_env = "LOCAL_LLM_KEY"

[models.hosted]
llm_model = "your-hosted-model"
llm_base_url = "https://llm.example.com/v1"
api_key_env = "HOSTED_LLM_KEY"
price_in_per_mtok = 0.25
price_out_per_mtok = 2.0

[space]
retrieval_k = [16, 36]
writer_rules = ["none"]
guided_sql = [true]
prose = [false]          # a mode's setting outside a mode's table is tried in every mode

[space.thinking]
corrector = [false]

[search]
budget_hours = 12.0
seconds_per_question = 20.0

[machine]
parallel = 4
```

Every value is checked before anything runs: a key the file does not know, a value of the wrong type (a count
written as `1.5` or `"2"`), or a value outside a setting's limits is refused with where and why.

### `[sweep]`

- `name`: the sweep's name, recorded in its evidence.
- `out`: the folder the sweep writes everything to. One sweep per folder.
- `max_errors` (default `0`): how many cases an arm may leave unmeasured and still be valid
  ([what each answer counts as](#what-each-answer-counts-as)).
- `repeats` (default `"auto"`): how many times each arm runs. Under `"auto"` the first round decides
  ([how many runs](#how-many-runs-each-arm-owes)); a number fixes it.

### `[dataset]`

- `questions`: the question file.
- `env`: one deployment's env file, as `mnemiq` itself reads it, which answers every question. Or `deployments`, a
  table from each `db_id` to its env file: `deployments = { sales = "sales.env", support = "support.env" }`. Name one or
  the other.
- `held_out` (default `0.3`): the share of the questions kept back to confirm the pick on. It is taken from the
  answerable questions and from the others separately, rounded down. It must be above 0 for a strategy that confirms;
  check mode holds none out.
- `split_seed`: decides which questions are held out. The split depends on the seed and each case's id only, so it is
  the same on every machine and when the file is reordered.
- `[dataset.identity]`: who the sweep asks as, `tenant` and `principal` (each `"local"` by default) and `roles` (at
  least one). See [identities](#identities-and-why-row-filters-are-refused).

The deployment's env file is the only place its settings come from: `MNEMIQ_*` variables in the shell running the
sweep are ignored, so the sweep measures the deployment its file names. That holds for `MNEMIQ_LOCAL_ONLY` too: set
it in the deployment's env file, and the engine's profile loader refuses any arm whose writer, judge or embedder would
leave the network. Its store must be built (`mnemiq enrich` and `mnemiq build`) before the sweep runs.

### `[models]`

One table per model, `[models.<name>]`. The name is the profile's name in the proposal unless `[propose]` gives
another, so it follows the profile name rule, `[a-z0-9][a-z0-9._-]*`.

- `llm_model`, `llm_base_url`: where the model is served, an `http` or `https` URL with no credential and no
  fragment (`#...`) in it.
- `api_key_env`: the environment variable holding the model's key. It must be set when the sweep runs and when it
  proposes: each refuses, naming it, when it is not.
- `llm_context_window`, `verify_model`, `verify_base_url`, `verify_api_key_env`: as in a profile, optional.
- `price_in_per_mtok`, `price_out_per_mtok`: dollars per million prompt and completion tokens. Without both, the
  sweep cannot price the model or check a dollar budget.

A model's endpoint is a fact about the model, stated here once. It is never a value to try, and it replaces whatever
the start profile says.

### `[judge]`

Optional: one judge for every model, replacing each model's own: `verify_model`, `verify_base_url` and
`verify_api_key_env`. A judge on another endpoint or model, in a mode that calls it, leaves dollars unknown, since a
row's tokens cannot be split between the writer and the judge.

### `[start]`

Where each model starts: `profile_file` and `profile` name a profiles file and one profile in it (its `default` when
`profile` is left out). A model's own `[start.<model>]` replaces it for that model: naming a file it replaces both,
naming only a profile it keeps `[start]`'s file. With neither, a model starts from the engine's defaults.

No profile setting the deployment's env file sets is part of an arm: each arm starts from the engine's field
defaults, overlaid by the start profile, and its profile file writes out every setting it holds a value for. What a
profile cannot change (the source, the store, the access policy and the rest) is the deployment's, the same for every
arm.

### `modes`

The modes the sweep runs, such as `["instant", "thinking"]`. It is a top-level key, so in the file it comes before the
first table. Each arm of the search runs in one mode, and the proposal is one profile holding each mode that ran.

### `[space]`

The values to try, each a list: `retrieval_k = [16, 36]`. A profile-wide setting goes here by its name; a mode's
setting goes in a table named for the mode, `[space.thinking]`, or here, where it is tried in every mode. Each value
becomes one arm in the screen. A value equal to the start's own is not an arm.

What may vary is what a profile may change ([settings.md](settings.md) lists them, with each one's technique). Refused:
a model's endpoint (`llm_model`, `llm_base_url`, `llm_context_window`, `verify_model`, `verify_base_url`), which
belongs to `[models]`; a deployment setting; a setting no profile can change; an empty list or a value listed twice;
and a value the setting's type or limits refuse. An arm is labelled by its changes, `retrieval_k=36`, a mode's setting
named with its mode, `thinking.corrector=false`, and several joined with ` + `.

Whether an arm can run is the engine's to say: each arm's profile file is loaded by the engine's own profile loader,
and one it refuses (a judge on a mode with one candidate, say) is skipped and listed by `plan` and `run`.

### `[search]`

- `strategy` (default `"screen-then-grid"`): `"screen-then-grid"`, `"grid"` (every combination of the space, with no
  screen first) or `"check"` ([check mode](#check-mode)).
- `screen_alpha` (default `0.25`): the screen's bar ([the rounds](#the-rounds)).
- `budget_hours`, `budget_dollars`: checked before each arm, never during one. A run stops before an arm the budget
  cannot cover, says so, and exits 3.
- `seconds_per_question`: an estimate of one question's time. `plan` reckons its hours from it, and the hour budget
  uses it until the sweep has measured its own pace.

`budget_hours`, `budget_dollars` and `seconds_per_question` change no answer, so a sweep can be resumed after
changing them. Changing `strategy` or `screen_alpha` needs a new sweep, in another out folder.

### `[propose]`

The rule, fixed before any answer is written:

- `alpha` (default `0.05`): the bar an arm must pass, after Holm's correction.
- `max_more_wrong`, `max_more_made_up` (default `0` each): how many more wrong answers, and made-up answers to
  questions the data cannot answer, an arm may give than the start.
- `profile`: the name of the proposed profile (default: the model's name).

### `[machine]`

- `parallel` (default `1`): how many arms run at once, each in a process of its own
  ([sizing it](#sizing-parallel-to-memory)).

### `[[control]]`

Optional, repeatable: another system's answers to the same questions, graded beside the arms
([controls](#controls-another-systems-answers)). Each has a `name` and `answers`, a JSON file.

## The rounds

A sweep compares every arm with its start on the same questions, one question at a time. Only the questions where the
two differ count: those the arm alone got right (`only`) and those the start alone got right (`other`). The exact
McNemar test asks how likely a split at least that uneven would be if either side were as likely to win each one.

1. **The screen** runs each value of the space as one change from the start, on the questions to tune on. It keeps a
   value when, in every run, its arm and the start are valid, the arm wins more of the differing questions than it
   loses, with a one-sided exact p below `screen_alpha`, and it adds no more wrong or made-up answers than `[propose]`
   allows. The bar is loose on purpose: the screen decides what is worth combining, not what is proposed.
2. **The grid** runs every combination of the values the screen kept that changes two or more settings.
3. **The rule.** An arm is a candidate in its mode when it is valid in every run, and in every run it wins more of the
   differing questions than it loses, with an exact two-sided p that passes Holm's correction at `alpha` over every
   arm compared with that start, screen and grid together, and it adds no more wrong or made-up answers than
   `[propose]` allows. Every arm compared is one more chance of a lucky win, so Holm raises the bar with their number.
4. **Timing.** The candidates with the most right answers on average, and the start, run once each, alone, on the
   first 20 questions to tune on. First the sweep waits, at most five minutes, for the machine's 1-minute load to fall
   under its core count. Before each question it reads the model server's count of running requests from the
   server's `/metrics` page, and it samples the machine's load while the run lasts. A reading is taken only when the
   run is valid, the server ran nothing else before any question, and the load stayed under the core count. An
   endpoint that publishes no such count is treated as hosted, and its times are indicative. A tie on right answers
   goes to the lower median time of a reading that was taken and is not indicative, when every tied arm has one, then
   to fewer prompt tokens.
5. **Confirm.** A profile holds every mode, so each mode's pick (or the start, where no arm won) is assembled into one
   profile. A profile-wide setting the picks disagree on keeps the start's value, and the evidence names it. That one
   profile runs against the start on the held-out questions, in every mode, once per run the arms owe. It passes when,
   in every mode and run, it is valid, gets at least as many of the differing questions right as the start, and adds
   no more wrong or made-up answers than `[propose]` allows. Then it is proposed. When it fails, the start is
   proposed: confirm runs once, with no second try, since trying another candidate would make the held-out questions
   part of the search. Wherever the rule falls back to the start, it proposes nothing instead when the start is not
   valid in every run, as check mode does: nothing could be compared with it.

With `strategy = "grid"`, the grid runs every combination of the space at once, its own start among them, with no
screen. Fine for a small space; the screen exists because combinations multiply.

### How many runs each arm owes

A model can answer the same question differently from one run to the next, so one run of an arm can win by chance.
Under `repeats = "auto"` the first round runs the start twice. When its two runs wrote the same SQL for every
question, every arm runs once; otherwise every arm runs twice, and an arm must win in both. Every arm runs once before
any arm runs twice. A number in `repeats` fixes the count for every arm.

### Room

After the start's first run, the sweep counts the answerable questions to tune on that the start got wrong. No arm can
gain more than that. When it is fewer than the smallest gain that could pass (the next section), no arm can win, so
the search stops for that model and mode, and the start is proposed there. The search goes on for the others.

With `parallel` above 1, other arms start beside the start's first run, before its room is known, and those already
running finish. They appear in the report and the evidence, but none is a candidate where the search stopped: each
says why, nothing of that mode is timed, and its pick in the confirm round is the start.

## What a sweep can show

With no question lost, an arm right alone on k questions has a two-sided p of 2 / 2^k. Holm's correction over m arms
puts the strictest bar at `alpha / m`. So the fewest more right answers, with none lost, that can pass at
`alpha = 0.05`, here with 500 answerable questions to tune on:

| Arms compared with the start | Fewest more right, with none lost |
|---|---|
| 1 | 6 |
| 5 | 8 |
| 10 | 9 |
| 20 | 10 |

More questions do not lower these numbers: they are set by the test and the number of arms. What more questions buy is
room, since the start must get at least that many wrong for any arm to win, and the power to show a win that also
loses some questions. An arm right alone on 10 questions and wrong alone on 2 has p = 0.039: it passes with one arm
compared, and not with five. `plan` prints the number for the sweep's own arms; each grid arm adds to them.

The held-out part is smaller still: it can show harm, and a gain only when it holds enough answerable questions. The
evidence says how many it holds and what gain it could show.

## Check mode

`strategy = "check"` does not search. It runs the start and the engine's defaults (an arm named `defaults` that
undoes whatever the start profile sets) on every question, with none held out, and compares them for harm. It proposes
the start when, in every run, the start's wrong and made-up answers exceed the defaults' by no more than `[propose]`
allows; the defaults otherwise; and nothing unless both arms are valid in every run. Use it when the question set is
too small to search, as `plan` says, or to check a profile you already run. `[space]` plays no part in it. Its start
must differ from the engine's defaults in each mode it runs: with no start profile, or one that sets nothing in a mode,
the two arms would be the same profile, and the config is refused.

## Controls: another system's answers

A `[[control]]` grades your current system's answers, or any other system's, beside the arms, so you can see how each
arm compares with it. Its `answers` file is a JSON list where each answer is `{"id": ..., "sql": ...}` or
`{"id": ..., "declined": true}`, each naming a case of the question file at most once.

The SQL is graded by the same grader as the arms, on the deployment's source, which is opened read-only whatever the
deployment allows. That read-only attach is the only barrier: a control's SQL is the operator's own file, and it
bypasses the engine's SQL checks, the decider among them. Put in a control only SQL you would run yourself.

A control is shown in every round's table, paired with each run of the start. It never decides what the screen keeps
or what is proposed, and it can be added to a sweep that has already run.

## What each answer counts as

Each answer, one row, counts as exactly one of:

- **right**: the gold rows (`Strict` counts the exact matches among them; the rest returned the right data in another
  shape, such as an extra column);
- **wrong**, or **made up** when the question has no answer and the arm gave one;
- **declined**;
- **failed**: the arm's own failure (error kind `answer`), such as SQL the source rejected at every repair. The question
  was measured, and the arm did not get it right. It is a column of its own, neither right nor wrong;
- **an error**: a model endpoint failed in a way that fails every question (error kind `endpoint`: a refused key, an
  unknown model, a rate limit, a server error, a connection that failed) or the source did (error kind `source`: the
  source did not answer, or the gold query failed while the answer was graded). The question was not measured. One
  question's own model error, such as a prompt too long for the window, is the arm's failure, not an error;
- **unjudged**: the answer went out while its judge could not be reached, so it is no measure of the arm as it was set;
- **ungraded**: a row with no outcome the report can read. No run writes one today; the column is there so one would
  show.

Errors, unjudged and ungraded answers are the questions an arm did not measure. An arm is valid when it holds one
answer for every question it owes, the questions it did not measure are at most `max_errors`, the files it pins have
not moved, and its answers on each deployment carry one profile version. Paired with the start, an answer is right or
it is not: failed, declined, wrong, an error and unjudged are all not right. So with `max_errors` above 0, a question
the start did not measure counts as one the start got wrong, and can read as a win for an arm.

An endpoint that fails three questions in a row is taken to have stopped answering. The arm stops and leaves those
three out, the arms running finish, no other starts, and `run` exits naming the endpoint (without its credentials).
The next run asks those questions again, after the arm's others. One that fails again counts as that question's own
error only once the endpoint has answered another question in that run. Until then it is held as before, so a run
made while the endpoint is still down stops again the same way, and no question is lost to it.

## The report

`mnemiq sweep report` prints, for each model and mode, the start, each arm and each control, with each run's values
in turn: Valid; Kept (the screen only); Right, Strict, Wrong, Made up, Declined, Failed, Errors, Ungraded, Unjudged;
the median seconds per question and to the first rows; the mean prompt tokens per question; and against the start,
each run's `only/other` and p. The screen's p is the one-sided p it keeps by; every other round's is two-sided.

## What the evidence holds

`propose` writes, for each model, `<out>/proposal/<model>/profiles.toml` (unless nothing is proposed) and
`evidence.json`. Neither holds a key: the profile names the variable that holds one. The evidence holds:

- **the sweep**: its name, its fingerprint (what its answers depend on), the engine build its arms ran on and the one
  that wrote the proposal, the split seed and each part's size, the identity, each model's endpoints without
  credentials, the rule (strategy, `alpha`, `screen_alpha`, the bounds, `max_errors`, the runs each arm owed and
  whether the first round decided them), and the sources it could not pin;
- **the verdict**: whether the start is valid in every run (`start_valid`), what was chosen and by whom (the rule or a
  person), the rule's own pick, whether it was confirmed, why in words, and any warnings;
- **each mode**: its pick and what decided it, its room (the start's misses, the gain needed, the arms compared), and
  every arm compared with the start, with its runs, its pairs, whether it passed Holm in each run and why it is not a
  candidate when it is not;
- **timing**: each reading's medians and checks (the questions checked, how often the server was busy, the machine's
  load against its cores), whether it was taken, and whether it is indicative;
- **confirm**: the assembled profile's changes and modes, the settings the picks disagreed on, whether it passed and
  in which mode it failed, and the held-out numbers beside the numbers it was picked on;
- **the profile**: its name, which arm's file it is, how it was measured (whole, or mode by mode when each mode ran an
  arm of its own), and its `version`, recomputed by the engine and checked against the version the measured arm's
  answers were stamped with;
- **modes the engine refused and modes the sweep did not search** (below), and in check mode the start and the
  defaults side by side.

`propose` writes both files only once every check has passed. It opens each deployment's source to read its
dialect, which a profile's version holds, so it needs the same access to the source as the run, and each key variable
the models name must be set, since the engine's loader reads them to recompute each version. Packaged by another engine
build than the one the arms ran on, it warns and goes on: the version check still refuses a profile that would resolve
to anything other than what was measured. A refusal for what was asked (a `--choose` that names no arm that ran and is
valid, a config edited since the sweep ran, a key variable not set, the out folder in use by a run or another
`propose`) leaves an earlier proposal as it was. A refusal that shows an earlier proposal no longer holds (the sweep
has not finished its rounds, its split is not the one it drew, an arm's file is not the profile its answers were
stamped with) removes it, so none is left looking current.

### A mode the engine refuses the start in

When the engine refuses a model's start in one mode, nothing runs there. That mode is left out of the confirm round,
and the proposed profile has no section for it: in that mode the profile keeps the engine's own per-mode settings, and
runs under the profile's whole-profile settings, which were never measured there. The evidence names the mode, the
engine's reason, and those whole-profile changes. The same holds for each of the engine's modes that `modes` leaves
out: the profile has no section for it, and the evidence (`unsearched_modes`) and its reason name it, with the
whole-profile changes never measured there.

## Identities, and why row filters are refused

Every arm asks as the identity in `[dataset.identity]`, through the deployment's access policy, as a person would.
The identity must be one whose answers can be graded:

- **No row filter and no mask.** The gold query runs as written, with no filter, while an arm's SQL carries the
  identity's row filters and masks. A right answer would return fewer rows, or masked ones, and be graded wrong. A
  sweep whose identity's grants hold either is refused before anything runs.
- **Every table the gold reads.** A question whose gold SQL reads a table the identity cannot see has no answer it can
  give that is graded right. The sweep refuses, listing those questions.

So sweep as a role that sees every table the questions read, unfiltered. The access policy itself is unchanged: the
profile you promote runs under each person's own grants.

## Sizing `parallel` to memory

Each running arm is a process of its own. It copies every store it reads into its run folder, opens the source and
builds a runtime over the copy, so `parallel` arms take about `parallel` times one arm's memory: the store copy (`plan`
prints its size), the store's index and the source's working set. Each run keeps its store copy on disk, so the out
folder grows by one copy per arm run.

Start at `parallel = 1`, watch one arm's memory during the smoke, and raise `parallel` while that many arms fit with
room to spare. The model server batches the arms' requests together, so more arms also load the server. That is why the
timing round runs alone, and why the times of the parallel rounds are not compared.

## Sweeping a live source: only from a snapshot

The sweep pins every file an answer depends on (the question file, the start profiles, each deployment's access
policy and store, and each source read from a file) and refuses to go on, or marks an arm invalid for good, when one
moves. A live database cannot be hashed, so a change to it during the sweep goes unseen, and arms run hours apart
would be graded against different data. `plan` lists such a source as not pinned.

So sweep a snapshot: a copy that nothing writes while the sweep runs. A SQLite or DuckDB file is pinned, so a change
to it is caught. A frozen copy in the database's own engine keeps the SQL dialect the profile will run against, which
the writer's prompt and the profile's version both depend on; it is not pinned, so keep everything else away from it
until the sweep ends. Build the snapshot deployment's store from the snapshot, and write the gold SQL in its dialect.

## Running, resuming and smoke

- **One run at a time per out folder.** A second `run` on the same folder is refused at once, and so is a `propose`
  while a run goes on, or beside another `propose`.
- **Resume.** A stopped run, whether by the budget, an endpoint, a worker, Ctrl-C or the machine, resumes where it
  stopped: each arm from the answers it wrote, and what the first round decided (the runs each arm owes, each start's
  room) read back. A resumed config may change only `budget_hours`, `budget_dollars`, `seconds_per_question` and its
  `[[control]]` tables. Anything else (the config, a deployment's settings, a pinned file) or another engine build is
  refused before any arm: put it back, or start a new sweep in another out folder. Each worker checks the build again
  as it starts an arm, so an edit or a pull while the sweep runs stops it with nothing of that arm asked, rather than
  writing rows under another build.
- **Smoke.** `--smoke N` runs each arm of the first round once on its first N questions to tune on, in
  `<out>/smoke/`, which nothing else reads, with no budget, room or repeats. Each smoke replaces that folder whole, and
  only a folder a smoke made.

The out folder holds the sweep's record, `sweep.json` (what it was started under and what its rounds decided); its
split, `split.json`; `retries.json`, the questions an endpoint left to ask again; one folder per run of each arm,
`<model>/<round>/<modes>/<arm>/run-<n>/`, with the arm's `profile.toml`, its answers in `rows.jsonl` (one graded row
per question and mode) and its store copies; `proposal/<model>/`; and `smoke/`.

## Promoting a proposal

The proposed `profiles.toml` holds one profile, its file's default, named `[propose] profile` or the model's name. It
writes out every setting it holds a value for, its endpoint and its key variable included; a setting left unset
inherits from the deployment, as in any profile. To promote it, point a deployment's `MNEMIQ_PROFILES_PATH` at it, or
copy its tables into your profiles file under the name you want, adding `roles` unless it is that file's default
([profiles.md](profiles.md)).

Then compare the `profile_version` the deployment's answers carry with the evidence's `version`. The version hashes
what a profile resolves to, never its name, so the same value means the deployment runs what was measured. Another
value means something that changes an answer differs: a deployment setting such as the question limit, a setting the
deployment fills in, the source's dialect, or the engine's prompts.
