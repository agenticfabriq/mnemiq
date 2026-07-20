#!/usr/bin/env bash
# repo-guard: block commits that leak prior-art proprietary names or secrets into this
# public repo. Runs as a pre-commit hook (--staged) and in CI (--all).
#
# If a match is a genuine false positive, narrow the pattern below or rename the token —
# do NOT weaken the guard casually. This file is excluded from its own scan.
set -uo pipefail

MODE="${1:---staged}"

list_files() {
  if [ "$MODE" = "--all" ]; then
    git ls-files
  else
    git diff --cached --name-only --diff-filter=ACM
  fi
}

# Files that legitimately contain the patterns (the scanner itself) — never scan them.
EXCLUDE_REGEX='^scripts/repo-guard\.sh$'

# Prior-art / employer proprietary names (case-insensitive). Never allowed in this repo.
NAME_PATTERNS='(\boracle\b|\boci\b|\bdfl\b|odpic|dbms_cloud|\badw\b|essbase|peoplesoft|oraclecloud)'

# Secret shapes. Never commit these.
SECRET_PATTERNS='(sk-[A-Za-z0-9]{16,}|AF_SECRET_KEY|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]{8,})'

# Hardcoded machine home paths leak the local username. Use $HOME / os.path.expanduser("~") /
# an env var instead (e.g. MNEMIQ_MINIDEV_DIR). Never a literal /Users/<name> or /home/<name>.
HOME_PATH_PATTERNS='/(Users|home)/[A-Za-z0-9._-]+'

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
    echo "BLOCKED [home-path]: $f (hardcoded home path leaks a username; use \$HOME / expanduser / an env var)"
    grep -InE "$HOME_PATH_PATTERNS" "$f" | head -3 || true
    fail=1
  fi
  case "$f" in
    *.example|*.sample|*.template) : ;;  # example/template files are meant to be committed
    .env|*/.env|.env.*) echo "BLOCKED [env-file]: $f (do not commit env files)"; fail=1 ;;
  esac
done < <(list_files)

if [ "$fail" -ne 0 ]; then
  echo ""
  echo "repo-guard: commit blocked. Remove the offending content (or route secrets/URLs through"
  echo "env vars and a gitignored .env). Adjust scripts/repo-guard.sh only for a true false positive."
  exit 1
fi
echo "repo-guard: clean."
