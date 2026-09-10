#!/usr/bin/env bash
# Exercises .githooks/pre-commit. Run it after touching that file:
#
#     bash scripts/test-pre-commit-hook.sh
#
# The hook is the only gate in this repo that nothing else checks, and it shipped two broken
# paths because of it: an escape hatch that aborted on an unbound variable AFTER printing
# "clean", and a lint step that read the working tree instead of the index. Both were found by
# a reviewer, not by running them. The hook's own opening comment says a gate that checks less
# than you think teaches people its green means nothing -- that applies to itself.
#
# EXIT CODES are asserted, not just output. The unbound-variable bug printed the right words
# and then exited 1; a test that grepped for the words would have passed it.
#
# Runs against a TEMPORARY INDEX (GIT_INDEX_FILE), so staging probe files here cannot disturb
# the real one.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

HOOK=.githooks/pre-commit
pass=0; fail=0
tmpidx="$(mktemp)"
# Probe files live in the real working tree, so they go in the trap too. Cleaned only on the
# success path, an interrupted run left `_probe_lint.py` (containing `import sys`) untracked in
# src/, after which a manual `ruff check .` reports an F401 in a file nobody wrote.
PROBES=""   # filled by stage(), so it cannot drift from the call sites
# Single-quoted on purpose: $PROBES must be read when the trap FIRES, not baked in when it is
# installed -- stage() appends to it as the suite runs. Unquoted inside, so the accumulated
# paths word-split into separate arguments; they are fixed literals from the call sites.
# shellcheck disable=SC2064,SC2086
trap 'rm -f "$tmpidx" $PROBES' EXIT

check() {  # check <name> <want_rc> <want_substring|-> ; probe files already staged by caller
  local name="$1" want_rc="$2" want="$3" out rc
  # Guarded expansion: bash 3.2 (macOS) errors on an empty array under `set -u`. Same bug the
  # hook itself hit passing config to git -- worth writing down twice.
  out="$(env ${ENVV[@]+"${ENVV[@]}"} bash "$HOOK" 2>&1)"; rc=$?
  if [ "$rc" -ne "$want_rc" ]; then
    printf 'FAIL %-44s exit %s, wanted %s\n' "$name" "$rc" "$want_rc"; fail=$((fail+1)); return
  fi
  if [ "$want" != "-" ] && ! printf '%s' "$out" | grep -qF "$want"; then
    printf 'FAIL %-44s exit ok but missing %s\n' "$name" "$want"; fail=$((fail+1)); return
  fi
  printf 'ok   %-44s exit %s\n' "$name" "$rc"; pass=$((pass+1))
}

stage() {  # stage <path> with <content>, into the temp index only
  # Registered here rather than in a list kept alongside: a hand-maintained list is one added
  # probe away from leaking a file into src/ on an interrupted run, with the suite green either
  # way. The trap re-reads PROBES when it fires, so appending after it is installed is fine.
  PROBES="$PROBES $1"
  printf '%s' "$2" > "$1"
  GIT_INDEX_FILE="$tmpidx" git add -- "$1"
}
reset_index() {
  : > "$tmpidx"; GIT_INDEX_FILE="$tmpidx" git read-tree HEAD
}

export GIT_INDEX_FILE="$tmpidx"
reset_index
ENVV=()

# `repo-guard: clean.` is repo-guard's own line. The obvious assertion -- `clean (repo-guard` --
# is the HOOK's summary literal, printed whether or not the guard ran: deleting the invocation
# left every check green. That is the property this file exists to defend, so it is asserted on
# the guard's own output, and then again on its behaviour below.
check "clean index passes"                        0 "repo-guard: clean."

# Composed, not written literally: a trailing space in THIS file is whitespace damage the hook
# would refuse, and a test whose fixture cannot be committed is not a test.
SP='   '
stage src/mnemiq/_probe_ws.py "x = 1${SP}
"
check "staged trailing whitespace blocks"         1 "BLOCKED [whitespace]"
ENVV=(MNEMIQ_SKIP_WHITESPACE=1)
check "  ...and the skip lets it through"         0 "NOT CHECKED:"
ENVV=()
rm -f src/mnemiq/_probe_ws.py; reset_index

stage src/mnemiq/_probe_lint.py 'import sys
'
check "staged lint violation blocks"              1 "BLOCKED [lint]"
ENVV=(MNEMIQ_SKIP_LINT=1)
check "  ...and the skip lets it through"         0 "NOT CHECKED: lint"
ENVV=(MNEMIQ_SKIP_LINT=0)
check "  ...but a falsey switch does not skip"    1 "BLOCKED [lint]"
ENVV=()
rm -f src/mnemiq/_probe_lint.py; reset_index

# The index is what is committed, so a violation fixed only on disk must still block.
stage src/mnemiq/_probe_idx.py 'import sys
'
printf '"""clean now."""\n' > src/mnemiq/_probe_idx.py   # working tree fixed, NOT restaged
check "index is linted, not the working tree"     1 "BLOCKED [lint]"
rm -f src/mnemiq/_probe_idx.py; reset_index

# The claim both skip messages make is "(repo-guard still runs)". Asserted by BEHAVIOUR, not by
# a string: stage something repo-guard rejects, turn both skips on, and require the commit to be
# refused. A summary line can be printed by a hook that never called the guard; a refusal cannot.
#
# The secret shape is composed rather than written out, for the same reason the trailing-space
# fixture is: a literal one in this file is a secret shape in the repository, and repo-guard
# would refuse the commit that adds the test.
FAKE_KEY="sk-$(printf 'A%.0s' 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)"
stage src/mnemiq/_probe_guard.py "KEY = \"$FAKE_KEY\"
"
ENVV=(MNEMIQ_SKIP_WHITESPACE=1 MNEMIQ_SKIP_LINT=1)
check "repo-guard runs even with both skips on"   1 "BLOCKED [secret]"
ENVV=()
rm -f src/mnemiq/_probe_guard.py; reset_index

printf 'FAIL: %s   ok: %s\n' "$fail" "$pass"
[ "$fail" -eq 0 ]
