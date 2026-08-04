#!/usr/bin/env bash
# repo-guard: block commits that leak proprietary names, personal identity, or
# secrets into this public-track repo. Runs as a pre-commit hook (--staged) and
# in CI (--all).
#
# The proprietary-name blocklist is NOT in this file or this repo: it comes from
# REPO_GUARD_NAME_PATTERNS -- set in the environment (CI passes a repo secret)
# or in the gitignored .env. A public guard must not carry what it blocks.
# Without it the guard fails closed rather than blessing an unchecked commit.
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
if [ -z "${REPO_GUARD_NAME_PATTERNS:-}" ]; then
  echo "repo-guard: BLOCKED [config]: REPO_GUARD_NAME_PATTERNS is not set."
  echo "Set it in .env (gitignored) or the environment; in CI it comes from a repo secret."
  echo "The guard fails closed: it will not bless a commit it could not check."
  exit 1
fi
NAME_PATTERNS="$REPO_GUARD_NAME_PATTERNS"

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

  if grep -InEi "$NAME_PATTERNS" "$f" >/dev/null 2>&1; then
    echo "BLOCKED [proprietary-name]: $f"
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
echo "repo-guard: clean."
