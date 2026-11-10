# Starters

A **starter** is a [profile](profiles.md) the engine ships inside its package, one per model build, holding the
answer-time settings a [sweep](sweeps.md) found safe over the engine's defaults for that build. A deployment's profile
names one by its exact name, `starter = "<name>"` ([profiles.md](profiles.md#starters)), and a sweep can start from one
with `[start] starter` ([sweeps.md](sweeps.md#start)). The engine never picks one from the model's name.

A starter is where to begin, not the answer. It was measured on one question set, on one build, and your questions
decide: sweep from the starter on questions people ask of your data, in check mode against the engine's defaults or as
a search from it, before you rely on it.

## What "safe" means

A starter gave no more wrong answers and no more made-up answers than the engine's defaults for that model, in each
measured mode, by the sweep's own rule and under the reading its evidence records (below), with both `[propose]`
bounds at 0 (`max_more_wrong` and `max_more_made_up`). It is not a claim that it is the best profile for the build,
only that its sweeps found it no worse on those two counts.

How the starter's counts were compared with the defaults' is the reading the rule ran under, which the evidence
records under `rule` (`harm`, [sweeps.md](sweeps.md#check-mode)). Publishing accepts either:

- **Run by run** (no `harm` in the evidence). In each run, the starter gave no more wrong and no more made-up answers
  than the defaults gave in the run of the same number. A search's confirm round, on the held-out part, reads them
  this way, and so does check mode by default.
- **Worst against worst** (`harm: worst`, check mode only, with two or more runs of each arm). The starter's highest
  wrong count over its runs is not above the defaults' highest over theirs, and the same for made-up answers. That is
  not "no more wrong answers in every run": a starter can give more than the defaults in one run and still be
  published, when its highest count is not above theirs. Like the right-answers reading below, it grows more lenient
  as the defaults get more runs, and with one run of each it would be the run-by-run reading.

### Right answers

The two ways a starter is made check right answers differently, and its evidence says which one it was made by
(`safe_by`):

- **`safe_by: check`.** In each measured mode, the starter's lowest right count over its runs is not below the
  defaults' lowest right count over theirs, on every question the check ran. That is not "no fewer right answers in
  every run": a starter can fall short of the defaults in one run and still be published, when its lowest count is not
  below theirs. With one run of the defaults, any shortfall refuses it.
- **`safe_by: search_from_defaults`.** Publishing reads no right counts of its own. What applies is the search's
  confirm round: in every run, on the held-out part only, the pick is right on at least as many questions as the
  start, which is the engine's defaults, paired question by question.

### Made-up answers

A made-up answer is one given to a question that has none, so only questions without an answer can show one.
`made_up_measured: false` means the questions could not show made-up answers, as on a set where every question has an
answer. A starter is published on such evidence with that side named as not measured. Its safety on declines is then
yours to show, by a sweep of your own on questions that include some with no answer.

## How one is made

A starter is written by a command, never by hand, so no number in its evidence is typed by a person.

1. Search on the build's server ([sweeps.md](sweeps.md)), and propose.
2. Run check mode on the same build, started from the search's proposal (`[start] profile_file`, the search's
   `<out>/proposal/<model>/profiles.toml`), against the engine's defaults, with both `[propose]` bounds at 0, and
   propose.
3. Publish from the two:

   ```bash
   mnemiq sweep starter --config check.toml --model <model> --dataset "<the question set's name>" --search search.toml
   ```

A search from the engine's defaults whose pick was confirmed on the held-out part can be published on its own, without
check mode: `mnemiq sweep starter --config search.toml --model <model> --dataset "<the question set's name>"`.

A check on its own can be published too, when the profile to ship already exists, from an earlier search or written by
hand:

1. Run check mode on the build's server, started from that profile (`[start] profile_file`, with `[start] profile`
   when the file's default is not the one), against the engine's defaults, with both `[propose]` bounds at 0, and
   propose.
2. Publish from the check alone, without `--search`:

   ```bash
   mnemiq sweep starter --config check.toml --model <model> --dataset "<the question set's name>"
   ```

Its evidence says `safe_by: check` and holds no search (`search` is null): the check is all that found it safe. Check
mode holds no question out, so a profile tuned on the same questions is measured on the questions it was tuned on;
check it on questions its tuning never saw when you can.

The starter is named for its build: the part of `[models.<model>.build] model` after the `/`, lowercased, so
`Qwen/Qwen2.5-Coder-32B-Instruct-AWQ` gives `qwen2.5-coder-32b-instruct-awq`. `--name` may only repeat that name.
`--dataset` labels the questions in the starter's description and its evidence: 1 to 80 printable ASCII characters,
not all spaces, with no `/` or `\`, so the label is never a path.

Publishing refuses, before it writes anything:

- a `--dataset` label that is not 1 to 80 printable ASCII characters, not all spaces, without a slash;
- a sweep whose out folder a run or a proposal is using: publishing holds each sweep's lock until it ends, as
  `propose` does, and reads each sweep's record and proposal once;
- a key variable the model names that is not set;
- a sweep that has not finished its rounds, or that ran under another config than the one given;
- an engine with no checkout to write into: the starters live in the package's source, `src/mnemiq/starter_profiles/`;
- an engine other than the one the sweeps ran on, by source hash: publish from a checkout at the sweeps' commit;
- a proposal a person chose (`propose --choose`), one made with either `[propose]` bound above 0, or one whose
  evidence records another reading of the harm rule (`harm` under `rule`) than the config the sweep ran under, as an
  edit of `evidence.json` would leave it;
- a `--search` that names a sweep in check mode, and a `--search` beside a sweep that is not check mode;
- check mode that proposed the engine's defaults, a search that did not start from them, a search whose pick was not
  confirmed on the held-out part, and a search whose pick is the defaults themselves, every change it picked lost to a
  disagreement between the modes;
- a starter that gets fewer right answers than the defaults, as [right answers](#right-answers) says;
- a build that is not whole: the server, its version and its window not read from the server by each sweep, the
  model, the GPU or the quantization not stated in `[models.<model>.build]`, two sweeps that read different servers or
  state different builds, or an `llm_context_window` other than the window the server reports;
- a build that could publish more than a build's facts: a server other than one on the engine's list of servers a
  starter may be published from (`vllm`), a server version not shaped as a release (digits and dots, then optionally
  a pre-release such as `rc2`, `.postN`, `.devN`, and an accelerator tag such as `+cu124` or `+rocm6.2`, never a word
  of the server's own), or a stated GPU or quantization that is not printable ASCII without a slash. A sweep on another
  server still runs; only what may be published is held to this;
- a `--name` other than the build's;
- with `--search`, a search whose proposal is not the file the check's arms started from, by its hash as they pinned
  it;
- evidence its schema refuses, such as a value of another type than the schema's;
- the starters folder, while another publisher holds its lock. Publishing takes that lock only once every check above
  has passed, and the staged checks below, so a refusal leaves the starters folder as it was, with no lock file in it.

It then writes the starter and its evidence into a staging folder outside the package, and checks them there: the
evidence as the sweep reads it, and that a deployment whose profile starts from the staged starter resolves, on every
deployment the sweep read, to the version the proposal recorded, with the engine's own loader and version function.
Only then are the two files moved into the starters folder as `<name>.evidence.json` and `<name>.toml`, each under a
hidden name until its move, while publishing holds the folder's lock. Holding it, publishing first removes any hidden
copy that a publish killed mid-move left in the folder, and no other file. A starter the folder lists is never a
draft, and when a check fails nothing reaches the folder: a starter already shipped under that name is left as it was.

Publishing writes the evidence, then the starter, and each is bound to the other: the evidence records the sha256 of
the starter's file (`starter_sha256`). If publishing is interrupted between the two, the pair fails closed: the engine
refuses at boot a starter whose evidence is missing, is a link, or records another file. Restore both files with git,
from the checkout publishing requires (`git checkout -- src/mnemiq/starter_profiles/<name>.*`), or publish again. For
a new starter, which git has never held, an interrupted publish leaves only a lone evidence file, which nothing loads:
delete it, or publish again.

## What the evidence holds

Every starter ships with `<name>.evidence.json` beside it. The engine reads its `starter_sha256` whenever it loads the
starter, and a sweep reads all of it before it starts from the starter. It holds builds, the rule and aggregate
counts, never a question, a statement, a case id or a path, and its schema refuses any field it does not name.
Deployments are numbered in it, never named. The dataset label is the publisher's own words, and is public:

- `starter`: the starter's name.
- `starter_sha256`: the sha256 of the starter's file. The engine loads the starter only beside evidence that records
  it, so a pair published incompletely is refused rather than loaded.
- `published`: the local date it was written.
- `dataset`: the label `--dataset` gave the questions, how many there are, and how many have an answer.
- `made_up_measured`: whether the questions hold one with no answer, which a made-up answer needs
  ([made-up answers](#made-up-answers)).
- `model_build`: the model's repository id, its quantization and the GPU, as the config states them, and the server,
  its version and its window, as the server reported them: a server on the engine's list, and a version shaped as a
  release. The folder the server loaded the weights from is never recorded.
- `engine_build`: the engine's package version, the commit the sweeps ran on, and its source hash.
- `safe_by`: `check` or `search_from_defaults` ([right answers](#right-answers)).
- `rule`: the strategy, `alpha`, both bounds, `max_errors` and the runs each arm owed, and `harm: worst` when check
  mode read the bounds worst against worst ([what "safe" means](#what-safe-means)). A missing `harm` and
  `harm: per_run` both mean the run-by-run reading, and publishing writes neither: it leaves `harm` out, as evidence
  written before the reading existed does. `harm: worst` is refused with a strategy other than `check`.
- `modes_measured`: the modes the starter was measured in, each of which it holds a section for.
- `unmeasured_modes`: each other mode, with the starter's profile-wide settings that differ from the engine's
  defaults, which apply there unmeasured.
- `held_unset`: the profile settings the starter holds no value for, which it therefore cannot clear.
- `check`: in check mode, each measured mode's runs of the starter and of the defaults side by side, each run's counts
  of right, strict, wrong, made-up, declined and failed answers, errors and what was not graded or judged, with the
  medians and prompt tokens. Null for a starter made by a search alone.
- `search`: from the search, its split, how many arms each mode compared with the start, each mode's pick, whether the
  pick was confirmed, and the held-out runs of the start and the pick. Null when no search is part of the evidence.
- `versions`: the version the measured profile resolves to on each deployment the sweep read, under
  `deployment-1`, `deployment-2` and on, numbered in the order of the deployments' sorted names, or under `""` for a
  sweep of one.
- `measured_versions`: when each mode was measured with a profile of its own, the version each mode's rows were stamped
  with, by deployment, numbered as in `versions`. Null when one profile was measured in every mode.

## What a starter overrides, cannot clear, and never reaches

A starter holds every answer-time value the measured profile held a value for: that profile, less its endpoints, its
key variables, its roles and its description, with a section for each measured mode.

- **It overrides** those settings and sections under any deployment, whatever the deployment's env file sets. A
  profile that names the starter overrides it in turn with its own settings.
- **It cannot clear** a setting the measured profile held unset, such as `writer_temperature`, `llm_seed` or
  `guard_fanout`, since TOML has no null: a deployment that sets one keeps its own value. The evidence lists them in
  `held_unset`.
- **It never reaches** the deployment settings that also change an answer, `verify_fail_closed` and
  `max_question_chars`, or the sections of the modes that were not measured.
- **Its mode sections** merge setting by setting with the sections of the profiles above it, and with nothing below
  it: `MNEMIQ_MODE_SETTINGS` is refused beside a profiles file, so an env file's mode settings never meet a starter.

## Deep and other unmeasured modes

A starter holds a section only for the modes it was measured in (`modes_measured`). Deep is not measured, so in deep,
and in any other mode its sweeps did not run, the engine's own per-mode settings apply, under the starter's
profile-wide settings, which were never measured there. `unmeasured_modes` names those settings for each such mode.
The temperature deep sends each of its candidates is in [settings.md](settings.md#several-candidates), under
"Several candidates".

## Re-sweeps at each tagged version

A starter was measured on one engine, and an engine change can change what it does. A release is a tagged version,
and re-sweeping every shipped starter on the engine being tagged is a step of tagging, in this order:

1. Set the release's version.
2. On that commit, run check mode of each shipped starter against the engine's defaults, started from the starter by
   name (`[start] starter`), with `[sweep] repeats = 2`, both `[propose]` bounds at 0, the reading its evidence
   records (`harm` under `rule`), and the modes the starter's evidence measured (`modes_measured`). A re-sweep is check
   mode alone; a new or replacement starter is made as [how one is made](#how-one-is-made) says.
3. Publish with `mnemiq sweep starter` from a checkout at that commit.
4. Tag the commit that adds the republished files, with no `.py` file of the package changed since the sweep, so the
   tagged engine has the source hash the evidence names (`engine_build` hashes only `.py` files).

A starter that publishing refuses after its re-sweep is removed from the package in that release, and the release
notes say so.

Until the first tag, a sweep that starts from a shipped starter warns that the engine differs from the one the starter
was swept on ([the model server's build](sweeps.md#the-model-servers-build)). Read that warning as "swept on an earlier
engine".

## Shipped starters

None yet.
