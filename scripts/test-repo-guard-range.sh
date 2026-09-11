#!/usr/bin/env bash
# Exercises `scripts/repo-guard.sh --range`. Run it after touching that file:
#
#     bash scripts/test-repo-guard-range.sh
#
# `--range` exists because `--all` resolves to `git ls-files` -- the working TREE -- so a branch
# whose commit N adds a secret and commit N+1 removes it scans clean while the secret stays
# permanently readable in the pushed public history (M97). Every case below builds a throwaway
# repo and asserts an EXIT CODE, because the mode that matters here is the one that reports
# "clean": a scanner that passes for the wrong reason looks exactly like one that works.
set -uo pipefail

GUARD="$(cd "$(dirname "$0")" && pwd)/repo-guard.sh"

# ASSEMBLED, never written literally, so this file does not trip the very scan it tests --
# `scripts/test-pre-commit-hook.sh` builds its sample the same way and for the same reason. The
# alternative is adding this path to the guard's exclusion list, which buys a blind spot in a
# real file to avoid one in a fixture.
FAKE_KEY="sk-$(printf 'a%.0s' $(seq 1 24))"
FAKE_HOME="/$(printf 'U')sers/someone/data"
pass=0; fail=0

check() {  # check <label> <expected-exit> <actual-exit>
  if [ "$2" = "$3" ]; then pass=$((pass+1)); printf '  ok   %-52s exit %s\n' "$1" "$3"
  else fail=$((fail+1)); printf '  FAIL %-52s expected %s, got %s\n' "$1" "$2" "$3"; fi
}

# EVERY `cd` is guarded, and that is not defensive style. Unguarded, a failed `mktemp -d` or
# `git init` leaves the shell in the operator's own checkout, where `git config user.name`
# rewrites their identity and the case commits land on their branch -- including `leak.py`, which
# carries a working key shape. A test for a secret scanner must not be able to commit a secret.
scratch() {  # a repo with the guard in it, one clean commit, and $BASE set
  REPO="$(mktemp -d)" || exit 1
  git init -q "$REPO" || exit 1
  cd "$REPO" || exit 1
  git config user.email t@t; git config user.name t
  mkdir -p scripts && cp "$GUARD" scripts/repo-guard.sh
  git add scripts/repo-guard.sh && git commit -qm base
  echo hello > app.py && git add app.py && git commit -qm clean
  BASE="$(git rev-parse HEAD)"
}

# 1. THE FINDING ITSELF: added then removed, so the tree is clean and the history is not.
scratch
printf 'KEY = "%s"\n' "$FAKE_KEY" > leak.py && git add leak.py && git commit -qm oops
git rm -q leak.py && git commit -qm removed
bash scripts/repo-guard.sh --range "$BASE..HEAD" >/dev/null 2>&1
check "add-then-remove is caught by --range" 1 "$?"
REPO_GUARD_NAME_PATTERNS=off bash scripts/repo-guard.sh --all >/dev/null 2>&1
check "...and --all still cannot see it (the reason --range exists)" 0 "$?"
cd / || exit 1; rm -rf "$REPO"

# 2. A range with nothing in it must PASS, or the check is just "always red".
scratch
echo "more" >> app.py && git add app.py && git commit -qm ordinary
bash scripts/repo-guard.sh --range "$BASE..HEAD" >/dev/null 2>&1
check "an ordinary range passes" 0 "$?"
cd / || exit 1; rm -rf "$REPO"

# 3. A home path introduced and kept.
scratch
printf 'P = "%s"\n' "$FAKE_HOME" > cfg.py && git add cfg.py && git commit -qm paths
bash scripts/repo-guard.sh --range "$BASE..HEAD" >/dev/null 2>&1
check "a machine home path is caught" 1 "$?"
cd / || exit 1; rm -rf "$REPO"

# 4. The scanner must not flag ITSELF. One of its own secret patterns is a bare token rather than
#    a shape, so its source matches it -- on mnemiq's real history that token in this scanner was
#    1,089 of the only matches there were. Naming the token here would trip the guard on THIS
#    file, which is the same lesson one level down.
scratch
sed -i.bak 's/^set -uo pipefail/set -uo pipefail\n# touched/' scripts/repo-guard.sh && rm -f scripts/repo-guard.sh.bak
git add scripts/repo-guard.sh && git commit -qm "edit the guard"
bash scripts/repo-guard.sh --range "$BASE..HEAD" >/dev/null 2>&1
check "the scanner does not flag its own pattern list" 0 "$?"
cd / || exit 1; rm -rf "$REPO"

# 5. No range is a usage error, not a silent pass.
scratch
bash scripts/repo-guard.sh --range >/dev/null 2>&1
check "--range with no rev-range refuses" 2 "$?"

# 6. THE INSTRUMENT CHECK. An awk without interval expressions matches nothing and would report
#    every range clean; the guard must refuse instead. Simulated by blinding the pattern.
# The replacement must not itself be a key shape -- the first one was `sk-` plus twenty
# alphanumerics, which is precisely what the pattern it replaced matches, and the guard blocked
# this file for containing it.
sed 's/sk-\[A-Za-z0-9\]{16,}/__no_such_pattern__/' scripts/repo-guard.sh > scripts/blind.sh
bash scripts/blind.sh --range "$BASE..HEAD" >/dev/null 2>&1
check "a matcher that fails its own control refuses" 2 "$?"
cd / || exit 1; rm -rf "$REPO"

# 7. A range git cannot resolve must REFUSE, not report clean. `scan_range` sent stderr to
#    /dev/null and took the empty output as "nothing found", so a typo'd or missing ref exited 0 --
#    and CI builds the range from `github.event.before`, which is where an unresolvable ref comes
#    from.
scratch
bash scripts/repo-guard.sh --range "nosuchref..alsonosuch" >/dev/null 2>&1
check "an unresolvable range refuses rather than passing" 2 "$?"

# 8. The reported SHA must be one an operator can look up. It was read from the wrong offset and
#    every hash printed was missing its first hex digit.
printf 'K = "%s"\n' "$FAKE_KEY" > leak.py && git add leak.py && git commit -qm oops
real="$(git rev-parse HEAD)"
reported="$(bash scripts/repo-guard.sh --range "$BASE..HEAD" 2>&1 \
            | grep -oE 'introduced at [0-9a-f]{12}' | head -1 | awk '{print $3}')"
if [ "$reported" = "${real:0:12}" ]; then
  pass=$((pass+1)); printf '  ok   %-52s %s\n' "the reported commit is the real one" "$reported"
else
  fail=$((fail+1)); printf '  FAIL %-52s got %s, want %s\n' "the reported commit is the real one" "$reported" "${real:0:12}"
fi
cd / || exit 1; rm -rf "$REPO"

echo
if [ "$fail" -ne 0 ]; then echo "repo-guard --range: $fail failed, $pass passed"; exit 1; fi
echo "repo-guard --range: $pass cases behave as intended"
