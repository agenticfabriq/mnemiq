#!/usr/bin/env bash
# repo-guard: block commits that leak proprietary names, personal identity, or
# secrets into this repo. Pre-commit hook (--staged) and CI (--all).
#
# THREE CHECKS, TWO AUDIENCES, and the split is deliberate.
#
# `secret shapes` and `home paths` carry their patterns in this file, in the open.
# They need nothing configured, so they run everywhere -- including on a pull request
# from a fork, which is the only guard a fork can get: GitHub withholds repository
# secrets from fork-triggered workflows, so anything needing one either passes
# vacuously or fails on config for exactly the contributors it most needs to check.
#
# `proprietary names` is LOCAL ONLY, and the list stays out of CI on purpose. It names
# an employer, and CI leaks it two ways. A repository secret is readable by anyone with
# write access, since a workflow step can print it. And the check would announce the
# list by enforcing it: `BLOCKED [proprietary-name] ... <word>` in a public log tells
# every reader that word is on the maintainer's blocklist, which is the association the
# list exists to avoid. The check stops the MAINTAINER committing those names, and it
# is the maintainer's machine that needs it -- an outside contributor cannot leak a
# name they have never heard.
#
# So the list comes from REPO_GUARD_NAME_PATTERNS (environment, or the gitignored
# .env), the check fails closed without it under --staged, and --all skips it and says
# so. Install the hook or the local half never runs:
#
#     git config core.hooksPath .githooks
set -uo pipefail

MODE="${1:---staged}"

list_files() {
  if [ "$MODE" = "--all" ]; then
    git ls-files
  else
    git diff --cached --name-only --diff-filter=ACM
  fi
}

# Resolve the blocklist: environment first (CI), then the gitignored .env.
if [ -z "${REPO_GUARD_NAME_PATTERNS:-}" ] && [ -f .env ]; then
  REPO_GUARD_NAME_PATTERNS="$(grep '^REPO_GUARD_NAME_PATTERNS=' .env | head -1 | cut -d= -f2- \
    | sed -e "s/^['\"]//" -e "s/['\"]\$//")"
fi
# Whether the name check runs. MODE first, then an EXPLICIT value -- never a guess.
#
# Keying it to MODE rather than to the variable being unset is deliberate: an earlier
# shape skipped names whenever REPO_GUARD_NAME_PATTERNS happened to be empty, and that
# is a guarantee a repository secret left configured -- or re-added by a later
# maintainer -- silently revokes.
#
# Under --staged the three states are spelled out rather than inferred. A first attempt
# inferred them from whether `.env` existed, which is wrong in the documented direction:
# `.env.example` says "Copy to .env", so a contributor who follows the setup has one,
# and the guard called them a maintainer and rejected every commit they made.
#
#   off      the machine has no list and is not meant to have one -- what a contributor
#            gets by copying .env.example, and what the check skips on
#   a regex  the maintainer's list; the check runs
#   empty    neither was chosen. Fail closed: an unset value is the one case that cannot
#            be told from a maintainer whose config broke, and a commit that skipped the
#            check is worse than one that is blocked.
# Normalised before the compare. An exact match would read `OFF` or a trailing space
# as a REGEX, and since the name grep is case-insensitive that pattern then blocks
# every file containing "off" -- a silent, baffling false positive on the value a
# contributor is most likely to type.
NAMES_SETTING="$(printf '%s' "${REPO_GUARD_NAME_PATTERNS:-}" \
  | tr '[:upper:]' '[:lower:]' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
CHECK_NAMES=1
if [ "$MODE" = "--all" ]; then
  CHECK_NAMES=0
  echo "repo-guard: proprietary-name check SKIPPED -- it runs locally only, by design."
  echo "  Secret shapes and home paths are checked below; neither needs configuration."
elif [ "$NAMES_SETTING" = "off" ]; then
  CHECK_NAMES=0
  echo "repo-guard: proprietary-name check off -- no list on this machine."
  echo "  That list is the maintainer's and is not needed to contribute; secret shapes"
  echo "  and machine home paths are checked below and are what a pull request must pass."
elif [ -z "$NAMES_SETTING" ]; then
  echo "repo-guard: BLOCKED [config]: REPO_GUARD_NAME_PATTERNS is not set."
  echo "Set it in .env (gitignored) or the environment. Two valid values:"
  echo "  off        you are contributing and hold no list -- the check is skipped"
  echo "  <regex>    the maintainer's blocklist -- the check runs"
  echo "The guard fails closed on an empty one: it will not bless a commit it could"
  echo "not check. Nothing else in your .env needs to change."
  exit 1
fi
NAME_PATTERNS="${REPO_GUARD_NAME_PATTERNS:-}"

# Secret shapes. Never commit these.
SECRET_PATTERNS='(sk-[A-Za-z0-9]{16,}|AF_SECRET_KEY|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]{8,})'

# Hardcoded machine home paths leak the local username. Use $HOME / os.path.expanduser("~")
# or an env var (e.g. MNEMIQ_MINIDEV_DIR) instead. Never a literal /Users/<name> or /home/<name>.
HOME_PATH_PATTERNS='/(Users|home)/[A-Za-z0-9._-]+'

# The scanner itself: free of name tokens now, but its secret-shape patterns
# can match their own spelling. Never scan it.
EXCLUDE_REGEX='^scripts/repo-guard\.sh$'

fail=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  if printf '%s\n' "$f" | grep -qE "$EXCLUDE_REGEX"; then continue; fi
  [ -f "$f" ] || continue

  if [ "$CHECK_NAMES" -eq 1 ] && grep -InEi "$NAME_PATTERNS" "$f" >/dev/null 2>&1; then
    echo "BLOCKED [proprietary-name]: $f"
    # Safe to echo the match: this branch is unreachable under --all, so the text
    # reaches a terminal on the maintainer's machine and never a public log.
    grep -InEi "$NAME_PATTERNS" "$f" | head -3 || true
    fail=1
  fi
  if grep -InE "$SECRET_PATTERNS" "$f" >/dev/null 2>&1; then
    echo "BLOCKED [secret]: $f (matched a secret pattern)"
    fail=1
  fi
  if grep -InE "$HOME_PATH_PATTERNS" "$f" >/dev/null 2>&1; then
    echo "BLOCKED [home-path]: $f (hardcodes a machine home directory)"
    grep -InE "$HOME_PATH_PATTERNS" "$f" | head -3 || true
    fail=1
  fi
done < <(list_files)

if [ "$fail" -ne 0 ]; then
  echo
  echo "repo-guard: commit blocked. Remove the offending content (or route secrets/URLs through"
  echo "env vars and a gitignored .env)."
  exit 1
fi
if [ "$CHECK_NAMES" -eq 1 ]; then
  echo "repo-guard: clean (names, secrets, home paths)."
else
  echo "repo-guard: clean (secrets, home paths; names are checked locally)."
fi
