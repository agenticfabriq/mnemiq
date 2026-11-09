# Profiles

No one set of answer-time settings suits every model and every dataset. A prompt that helps a frontier model can
hurt a small local one, and a schema form that suits one database can crowd another's prompt. So mnemiq ships only
its defaults, and a deployment names its own **profiles** in a file: each one is a model and the settings that suit
it on this deployment's data.

A request can pick a profile it is allowed, with no restart, and every answer records which version of which profile
produced it.

## The file

Point `MNEMIQ_PROFILES_PATH` at a TOML file. Without one, the deployment's own settings run as a single profile
named `default`.

```toml
default = "frontier"

[profiles.frontier]
description = "The hosted model, with the engine's defaults"

[profiles.qwen32b-bare-t0]
description = "Qwen2.5-Coder-32B on a local vLLM server: instant reads the bare schema, and the writer samples at 0"
roles = ["analyst"]
llm_base_url = "http://127.0.0.1:8000/v1"
llm_model = "qwen2.5-coder-32b"
api_key_env = "LOCAL_LLM_KEY"
writer_temperature = 0.0

[profiles.qwen32b-bare-t0.instant]
card_style = "bare"
```

- **`default`** names the profile that serves whoever names none. It takes no `roles`.
- **Every other profile lists `roles`**, the roles or groups that may pick it, read as the access policy reads an
  identity. `["*"]` means anyone.
- **A profile lists only what differs** from the deployment's settings. Anything it leaves out is inherited, with
  three exceptions when a profile moves an endpoint, since the old value belongs to the old server: moving the chat
  endpoint drops the deployment's `llm_context_window` unless the profile sets one, and moving the judge's endpoint
  (which follows the chat endpoint unless `verify_base_url` is set) drops the deployment's `verify_model` and judge
  key unless the profile sets them.
- **Keys are never written in the file.** `api_key_env` and `verify_api_key_env` name the environment variables that
  hold them. A profile that moves the chat endpoint must name its own key variable, or it is refused: the deployment's
  key is never carried to another host. A server that takes no key still needs the variable to hold some value.
- **Per-mode settings** go in a section named for the mode: `instant`, `thinking` or `deep`. They take the keys
  `MNEMIQ_MODE_SETTINGS` takes: `card_style`, `candidates`, `corrector`, `judge`, `min_agreement`, `budget_s`,
  `max_attempts`, `verify`, `plan_attempts` and `prose`. With a profiles file, state them there:
  `MNEMIQ_MODE_SETTINGS` alongside it is refused.

## A profile's parent: `extends` and `starter`

A profile can start from one parent instead of from the deployment's settings alone: another profile in the same
file, named with `extends`, or a starter profile the engine ships, named with `starter`.

```toml
default = "local"

[profiles.local]
llm_base_url = "http://127.0.0.1:8000/v1"
llm_model = "qwen2.5-coder-32b"
api_key_env = "LOCAL_LLM_KEY"
writer_temperature = 0.0

[profiles.local-bare]
description = "The local model, with instant reading the bare schema"
roles = ["analyst"]
extends = "local"

[profiles.local-bare.instant]
card_style = "bare"
```

- **The layers, in order:** the deployment's settings, then the starter if the chain's root names one, then each
  profile from the root down, ending with the profile itself. Each overrides the one before. A chain can be as long
  as you need (`a` extends `b`, which extends `c`), and the order of the profiles in the file does not matter.
- **At most one parent.** A profile names `extends` or `starter`, never both, so only a chain's root can name a
  starter. A profile with neither starts from the deployment alone, as it always has.
- **Never inherited:** `roles`, `description`, and being the `default`. A profile that extends the default is not the
  default, so it lists its own `roles`.
- **Mode sections merge setting by setting.** A parent's `instant` section that sets `card_style` and a child's that
  sets `prose` give the child both.
- **The endpoint rules apply through a parent**, each layer compared with the one before it, as a profile with no
  parent is compared with the deployment. A child that moves the chat endpoint off its parent's must name its own
  key variable, and drops the parent's `llm_context_window`, judge model and judge key unless it sets them. So a
  child that points back at the deployment's own URL, under a parent that moved away, is moving too and must name its
  own key: the deployment's key is never picked back up. Whether the profile ends with a chat endpoint and key is
  checked on the resolved result, so a starter, which never names one, is not refused for it. A parent in the file
  is itself a profile: it is loaded and checked on its own, so it passes every check at boot as any profile does,
  a chat endpoint and key included, and `roles` unless it is the default.
- **A child cannot unset what a parent sets.** TOML has no null, so leaving a setting out inherits it. To run without
  a parent's setting, extend from a profile that does not set it.
- **Versions hash the resolved result**, not the file ([Versions](#versions)). A change to a parent or to a starter
  gives every profile under it a new version, and so can an engine upgrade, since an upgrade can change a starter it
  ships. A profile with no parent keeps the version it had before parents existed.

### Starters

A starter is a profile the engine ships inside its package, one file per model build, holding the answer-time
settings measured for that build. A profile names one by its exact name, `starter = "<name>"`: the engine never
matches one from the model's name. A name it does not ship, whether typed wrong, shaped like a path, or removed by
an upgrade, is refused at boot, and the refusal lists the starters this engine ships.

- **A starter wins over the deployment's environment.** It is layered over the deployment, so with
  `MNEMIQ_RETRIEVAL_K=36` in the env file and a starter that sets `retrieval_k = 24`, the profile runs with 24. Set the
  value in the profile itself to override the starter.
- **What a starter cannot clear.** A setting the starter leaves unset, such as `writer_temperature`, `llm_seed` or
  `guard_fanout`, keeps the deployment's value. TOML has no null, so a starter has no way to unset one.
- **What a starter never reaches.** The deployment settings that still change an answer, `verify_fail_closed` and
  `max_question_chars`, and every mode the starter holds no section for.
- **What a starter never sets.** The model and its endpoint (`llm_model`, `llm_base_url`, `llm_context_window`,
  `verify_model`, `verify_base_url`), either key variable, `roles`, or a parent of its own. Those belong to the
  deployment and its file, so a starter that sets one is refused.
- **A starter's mode sections** merge setting by setting with the sections of the profiles above it, and with nothing
  below it: `MNEMIQ_MODE_SETTINGS` is refused beside a profiles file, so an env file's mode settings never meet a
  starter.
- **A starter ships bound to its evidence.** `<name>.evidence.json` beside it records the sha256 of the starter's file
  (`starter_sha256`), and a starter whose evidence is missing, is not JSON or records another file is refused at
  boot, as one published incompletely.

How a starter is made from sweeps, what is meant by its being safe, what its evidence holds, and which starters this
engine ships: [starters.md](starters.md).

## What a profile may change

Only how a question is answered: the model and its endpoint, the writer's prompt and temperature, the schema form,
retrieval, the verifier and the per-mode settings. [settings.md](settings.md) explains each one, with its default
and the technique it belongs to; the code's list is `PROFILE_SETTINGS` in
[`src/mnemiq/profiles.py`](../src/mnemiq/profiles.py).

Nothing that decides who may see what is a profile's to change: the access policy, row and column rules, local-only
mode, the data source and the store stay with the deployment. A setting no one has classified cannot be set from a
profile, since the list is an allow-list.

## Picking one

`POST /v1/ask` and `POST /v1/chat` take an optional `profile` field. A name the caller may not pick is refused the
same way as a name that does not exist, so a name cannot reveal that a profile exists. `/v1/ask` answers 422 and
lists the profiles the caller may pick; `/v1/chat` ends its stream with a `RUN_ERROR` that says the profile is not
offered, without the list.

## Versions

Every answer carries `profile`, `profile_version` and `engine_commit`, and the answer log records them. The version
is a hash of what the profile **resolves to**, not of the file: its settings (with any credentials stripped from
URLs), its modes, the deployment settings that still change an answer, and a hash of every prompt it renders. So:

- an unchanged file can get a new version when an engine upgrade changes a default or a prompt it inherits;
- the profile's name and description are not hashed, so renaming a profile keeps its version;
- the same file on another deployment, with another endpoint, model or source dialect, gets another version.

Card rendering and the SQL decider are not hashed, which is why each answer stamps the engine's commit beside the
version.

## Refused at boot

- A profiles file without an embedder in the deployment's own settings (`MNEMIQ_EMBED_BASE_URL` and
  `MNEMIQ_EMBED_API_KEY`, or the chat endpoint and key they fall back to). The embedder stays with the deployment,
  since the store's index was built with it, so a profile cannot supply it.
- `MNEMIQ_PROFILE`, a setting from before profile files: name the file in `MNEMIQ_PROFILES_PATH` instead.
- Per-mode settings that do not fit together, such as a judge or a minimum agreement on a mode with one candidate.
- `writer_temperature` on a model that takes no temperature, such as GPT-5, which would ignore it.
- `writer_reasoning` with `guided_sql`, which allows no text before the JSON.
- With a profiles file, both of these refusals name the layer each value came from: the profile, a parent, a
  starter, or the deployment.
- A profile whose chat endpoint has no key, or whose judge runs on another endpoint with no key of its own.
- A profile that would leave the network under `MNEMIQ_LOCAL_ONLY`.
- An unknown key, a profile name outside `[a-z0-9][a-z0-9._-]*`, or a `default` that names no profile.
- A profile that names two parents (`extends` and `starter`), extends a profile the file does not hold, or extends a
  chain that comes back to itself, which the refusal shows from the profile to the repeated name.
- A profile other than the default that lists no `roles`, even one that extends a profile that does.
- A `starter` this engine does not ship. The refusal lists the ones it does.
- A starter that sets the model, an endpoint, the context window, a key variable, `roles`, or a parent.
- A starter whose evidence is missing, is not JSON, or does not record the sha256 of the starter's file.

## Finding the right settings

The best profile for a model and a dataset is found by measuring, not by guessing: run the candidate settings over
questions with known answers, keep what moves the score by more than the run-to-run noise, and check the result on
questions the search did not use before relying on it. `mnemiq sweep` does exactly this and proposes a profile for each
model, with its evidence, for you to promote: see [sweeps.md](sweeps.md).
