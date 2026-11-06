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
- **A profile lists only what differs** from the deployment's settings. Anything it leaves out is inherited.
- **Keys are never written in the file.** `api_key_env` and `verify_api_key_env` name the environment variables that
  hold them. A profile that moves the chat endpoint must name its own key variable, or it is refused: the deployment's
  key is never carried to another host. A server that takes no key still needs the variable to hold some value.
- **Per-mode settings** go in a section named for the mode: `instant`, `thinking` or `deep`. They take the keys
  `MNEMIQ_MODE_SETTINGS` takes: `card_style`, `candidates`, `corrector`, `judge`, `min_agreement`, `budget_s`,
  `max_attempts`, `verify`, `plan_attempts` and `prose`. With a profiles file, state them there:
  `MNEMIQ_MODE_SETTINGS` alongside it is refused.

## What a profile may change

Only how a question is answered: the model and its endpoint, the writer's prompt and temperature, the schema form,
retrieval, the verifier and the per-mode settings. The full list is `PROFILE_SETTINGS` in
[`src/mnemiq/profiles.py`](../src/mnemiq/profiles.py), and each setting's meaning is in
[`.env.example`](../.env.example).

Nothing that decides who may see what is a profile's to change: the access policy, row and column rules, local-only
mode, the data source and the store stay with the deployment. A setting no one has classified cannot be set from a
profile, since the list is an allow-list.

## Picking one

`POST /v1/ask` and `POST /v1/chat` take an optional `profile` field. A name the caller may not pick is refused the
same way as a name that does not exist, so a name cannot reveal that a profile exists. The refusal lists the
profiles the caller may pick.

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

- `writer_temperature` on a model that takes no temperature, such as GPT-5, which would ignore it.
- A profile whose chat endpoint has no key, or whose judge runs on another endpoint with no key of its own.
- A profile that would leave the network under `MNEMIQ_LOCAL_ONLY`.
- An unknown key, a profile name outside `[a-z0-9][a-z0-9._-]*`, or a `default` that names no profile.

## Finding the right settings

The best profile for a model and a dataset is found by measuring, not by guessing: run the candidate settings over
questions with known answers, keep what moves the score by more than the run-to-run noise, and check the result on
questions the search did not use before relying on it.
