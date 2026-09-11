#!/usr/bin/env bash
# repo-guard: block commits that leak proprietary names, personal identity, or
# secrets into this repo. Pre-commit hook (--staged), CI over the tree (--all), and CI over
# the commits a push adds (--range <rev-range>), which is the only one that sees a secret
# added and removed within a branch (M97).
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
# .env), the check fails closed without it under --staged, and the two CI modes -- --all and
# --range -- skip it and say so. Install the hook or the local half never runs:
#
#     git config core.hooksPath .githooks
set -uo pipefail

MODE="${1:---staged}"
RANGE="${2:-}"
if [ "$MODE" = "--range" ] && [ -z "$RANGE" ]; then
  echo "repo-guard: --range needs a rev-range, e.g. --range origin/main..HEAD" >&2
  exit 2
fi

list_files() {
  if [ "$MODE" = "--all" ]; then
    git ls-files
  elif [ "$MODE" = "--range" ]; then
    :   # --range does not scan paths; see scan_range
  else
    git diff --cached --name-only --diff-filter=ACM
  fi
}

# `--range <rev-range>`: content INTRODUCED by the commits a push adds, which is the one thing
# neither other mode can see. `--all` resolves to `git ls-files` -- the working TREE -- so a branch
# whose commit N adds a secret and commit N+1 removes it scans clean, while the secret stays
# permanently readable in the pushed public history, recoverable from the commit object long after
# the tip looks fine (M97). `--staged` has the same blind spot from the other end: it sees each
# commit as it is made, and a `--no-verify` or an amend-and-force skips it entirely.
#
# Reads ADDED LINES from a patch rather than each commit's whole tree. `git grep` over every
# commit is O(commits x tree) and re-reads unchanged files thousands of times; a secret has to
# arrive as an added line at some commit, so one pass over the patches sees every introduction and
# nothing twice.
#
# The scanner excludes ITSELF by pathspec, for the reason the tree scan does: its patterns can
# match their own spelling. Measured on mnemiq's history -- 1,399 commits -- the only file that
# ever matched a secret shape was this one, 1,089 times, matching the literal `AF_SECRET_KEY` in
# its own pattern list.
scan_range() {
  git log -p -U0 --no-color --diff-filter=AMR --format='__COMMIT__ %H' "$RANGE" \
      -- . ":(exclude)$EXCLUDE_PATH"
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
# `--range` joins `--all` here, and for the identical reason rather than by analogy: both run in
# CI, whose log is public on a public repo, and enforcing the list there would LEAK it -- a
# flagged word in a public log tells every reader that word is on the blocklist. A fork's pull
# request could not receive the secret that carries it anyway.
if [ "$MODE" = "--all" ] || [ "$MODE" = "--range" ]; then
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
EXCLUDE_PATH='scripts/repo-guard.sh'
EXCLUDE_REGEX="^$(printf '%s' "$EXCLUDE_PATH" | sed 's/\./\\./g')\$"

fail=0

if [ "$MODE" = "--range" ]; then
  # POSITIVE CONTROL, before trusting a clean result. The matching below runs in `awk`, and the
  # secret pattern leans on interval expressions (`{16,}`) that not every awk implements -- BSD
  # awk here, mawk on the CI runner. An awk that ignores intervals reports every range clean and
  # says nothing, which is the failure mode this whole finding is about: a scanner that passes
  # because it cannot see. So prove the engine matches a known sample and REFUSE if it cannot,
  # rather than emit a green tick a reader would trust.
  control='+KEY = "sk-abcdefghijklmnopqrstuvwx"'
  if ! printf '%s\n' "$control" \
       | awk -v secret="$SECRET_PATTERNS" '/^\+/ { if (substr($0,2) ~ secret) found=1 }
                                           END { exit found ? 0 : 1 }'; then
    echo "repo-guard: BLOCKED [instrument]: this awk does not match the secret pattern" >&2
    echo "  (interval expressions such as {16,} are the usual cause). A scanner that cannot" >&2
    echo "  detect its own positive control must not report a range clean." >&2
    exit 2
  fi

  # ONE awk pass, not a shell loop. The first version ran two `grep`s per added line through a
  # `printf` pipe -- two process spawns per line -- and measured 45s for 50 commits and 293s for
  # 300, about a second each. A CI step that costs a second per commit is one somebody narrows
  # later for speed, which is how the tree-only scan got here. This is the same matching in one
  # process.
  #
  # The matched TEXT is never printed: this mode runs in CI, whose log is public on a public repo,
  # so the shape and its location are the finding and the value is not.
  # git's own failure must not read as "nothing found". Sending stderr to /dev/null and taking
  # the empty output made an unresolvable range -- a typo, a missing object, a shallow clone --
  # exit 0 and report the push clean. CI builds this range from `github.event.before`, which is
  # exactly where an unresolvable ref comes from.
  if ! range_raw="$(scan_range)"; then
    echo "repo-guard: BLOCKED [instrument]: git could not read the range '$RANGE'." >&2
    echo "  A range this scanner cannot walk is not a range with nothing in it." >&2
    exit 2
  fi
  range_out="$(printf '%s\n' "$range_raw" | awk -v secret="$SECRET_PATTERNS" -v home="$HOME_PATH_PATTERNS" '
    # 12, not 13. `__COMMIT__` is ten characters and the space is the eleventh, so 13 dropped
    # the first hex digit and every SHA this printed was one an operator could not look up.
    /^__COMMIT__ / { commit = substr($0, 12, 12); next }
    /^\+\+\+ b\// { file = substr($0, 7); next }
    /^\+/ {
      body = substr($0, 2)
      if (body ~ secret) print "BLOCKED [secret]: " file " introduced at " commit " (matched a secret pattern)"
      else if (body ~ home) print "BLOCKED [home-path]: " file " introduced at " commit
    }
  ' | sort -u)"
  if [ -n "$range_out" ]; then
    printf '%s\n' "$range_out"
    fail=1
  fi
fi

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
