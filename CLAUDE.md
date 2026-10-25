# mnemiq: instructions for Claude Code

This repository is **public** (`agenticfabriq/mnemiq`, Apache-2.0). Commits, their history, pull
request text and CI logs are all public, and none of it can be taken back. [CONTRIBUTING.md](CONTRIBUTING.md)
covers setup and what makes a good pull request; this file covers what an agent working here must
not get wrong.

## Changes land through a pull request

Branch from `main`, push the branch, open a pull request, and merge on GitHub. `main` is protected:
a pull request is required (administrators included), and force-pushes and rebase merges are
blocked.

- **Required checks:** `unit`, `workbench`, `guard` and `mcp-2x`.
- **CodeQL (GitHub's default setup) runs on every pull request without being required.** Let it finish green too. An alert
  needs a decision, a fix or a dismissal with a reason, which no setting can make.

## Run the repo guard on every commit and before a branch's first push

`scripts/repo-guard.sh` checks for secret shapes and machine home paths, and on a maintainer's
machine for names on a private blocklist.

- `git config core.hooksPath .githooks` runs it before each commit, with whitespace and lint checks.
  Staged mode is the only one that checks names. It takes the staged paths, skipping renamed ones,
  and reads their working-tree copies. So commit each file as it stands (no partial staging, no
  edits between `git add` and the commit), and commit a rename apart from edits to that file.
- Before a branch's first push, run `scripts/repo-guard.sh --range origin/main..HEAD` from the repo
  root. It scans the lines each non-merge commit on the branch adds, so it catches a secret one
  commit adds and a later one removes; a merge commit's own conflict resolution is not scanned. With
  no arguments it checks only what is staged, which before a push is usually nothing.
- The blocklist is `REPO_GUARD_NAME_PATTERNS`, read from the environment when set and non-empty,
  else from this checkout's own `./.env` (never the main checkout's), so a worktree needs its own
  `.env` unless the variable is exported. A contributor sets it to `off`, as `.env.example` does,
  which turns the name check off. Only when neither gives a value is every commit blocked.
- On a maintainer's machine, check the guard reports `clean (names, secrets, home paths)`. If it
  prints `proprietary-name check off` instead, this checkout has no list (a worktree whose `.env`
  was copied from `.env.example`, say), and its commits are not checked for names.
- CI skips the name check, because a public log would reveal the list.

## What never goes in this repository

- Secrets, `.env` files, and machine paths such as `/Users/<name>/...`. Use `$HOME` or a `MNEMIQ_*`
  environment variable.
- Customer data, customer schemas, or the names of their tables and columns.
- Internal specs, plans and strategy. Docs here are written for a public audience.
- Questions or gold SQL from gated benchmarks.
- Details of a security issue before it is fixed. Report one as CONTRIBUTING.md says.
- Code copied from another company's codebase. The engine is clean-room: patterns, never code.
