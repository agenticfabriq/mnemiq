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
tmpidx="$(mktemp)"; trap 'rm -f "$tmpidx"' EXIT

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
  printf '%s' "$2" > "$1"
  GIT_INDEX_FILE="$tmpidx" git add -- "$1"
}
reset_index() {
  : > "$tmpidx"; GIT_INDEX_FILE="$tmpidx" git read-tree HEAD
}

export GIT_INDEX_FILE="$tmpidx"
reset_index
ENVV=()

check "clean index passes"                        0 "clean (repo-guard"

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

printf 'FAIL: %s   ok: %s\n' "$fail" "$pass"
[ "$fail" -eq 0 ]
