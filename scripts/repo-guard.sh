#!/usr/bin/env bash
# repo-guard: block commits that leak proprietary names, personal identity, or
# secrets into this repo. Pre-commit hook (--staged, reading the index), CI over the tree
# (--all), and CI over the commits a push adds (--range <rev-range>, merge commits' own
# resolutions included), which is the only one that sees a secret added and removed within a
# branch (M97).
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
{ set +x; } 2>/dev/null  # never traced: a trace prints the private lists and identities this reads
set -uo pipefail

MODE="${1:---staged}"
RANGE="${2:-}"
if [ "$MODE" = "--range" ] && [ -z "$RANGE" ]; then
  echo "repo-guard: --range needs a rev-range, e.g. --range origin/main..HEAD" >&2
  exit 2
fi

list_files() {
  # NUL-delimited in both modes: git C-quotes a path holding a quote, a backslash, a tab, a newline
  # or, by default, a byte above 0x7f, and the quoted spelling names no file and no blob, so the
  # tree scan skipped such a path silently and the index scan would refuse it as an instrument
  # fault (the review gate on the fix, twice: `core.quotePath=false` covers the last case alone).
  if [ "$MODE" = "--all" ]; then
    git ls-files -z
  elif [ "$MODE" = "--range" ] || [ "$MODE" = "--identities" ]; then
    :   # neither scans paths: --range reads its commits (scan_range), --identities reads identities from stdin
  else
    # R and T too: a renamed file's staged content is scanned like any other's (without R, `git mv
    # b c` plus a blocked line appended to c showed as R096 and passed, where the same edit
    # without the rename was blocked: M221; the hook's whitespace check has always listed ACMR),
    # and so is a symlink replaced by a regular file, which git stages as a type change (the
    # review gate on the fix).
    git diff --cached --name-only -z --diff-filter=ACMRT
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
  # `--diff-merges=combined`: `git log -p` prints no patch for a merge commit by default, so a
  # line added only in a merge's conflict resolution was never scanned (M219), and CI runs this
  # mode on every push, the merge a pull request's merge button makes included. A combined diff
  # shows the lines present in NO parent, prefixed `++`, which the added-line rule below reads
  # through its first `+`; a line the second parent already has is prefixed `+ ` and is read
  # too, which rescans that parent's lines where its commits lie outside the range (a merge of
  # `origin/main` into a branch scanned as `origin/main..HEAD`) and flags one at the merge, a
  # verdict `--all` on main keeps true; a line the first parent already has is prefixed ` +`, is
  # not an added line, and is not read.
  # `--text`: a binary blob's patch is otherwise the one line `Binary files ... differ`, with no added
  # line to read, so a key shape behind a NUL byte reached the public history unscanned (Codex's
  # review of the fix); the patterns are ASCII and match in a binary blob as in a text one.
  # T too: a symlink replaced by a regular file is a type change, which `AMR` left out, so a key
  # shape such a file carried reached the history unread, and a later commit restoring the
  # symlink left `--all` a clean tree to look at (the second Codex review of the fix).
  git log -p -U0 --text --no-color --diff-filter=AMRT --diff-merges=combined --format='__COMMIT__ %H' "$RANGE" \
      -- . ":(exclude)$EXCLUDE_PATH"
}

# What the commit will hold: under --staged the INDEX blob, under --all the working tree. The
# staged scan read the working-tree copy of each staged path, so a blocked line staged and then
# removed from the file without staging the removal was committed unscanned, and a staged file
# deleted from the working copy was skipped (M220); the hook materialises the index only for
# ruff, after this script has run.
content_of() {
  # Stage 0 spelled out: `:<path>` is parsed as `:<n>:<rest>` where the path begins with a digit
  # and a colon, so `0:s.txt` read as stage 0 of `s.txt` (the review gate on the fix).
  if [ "$MODE" = "--staged" ]; then git show ":0:$1"; else cat "$1"; fi
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
if [ "$MODE" = "--identities" ]; then
  CHECK_NAMES=0   # it reads identities, never content
elif [ "$MODE" = "--all" ] || [ "$MODE" = "--range" ]; then
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

# Who a commit says made it. A checkout without a repository-local identity commits as the machine's global one, and the
# content checks above pass such a commit: on 2026-11-21 seven commits from a scratch clone reached main as the
# maintainer's private identity, and main's history had to be rewritten. So each commit's author and committer are read
# against a private identity list, resolved like the name list (environment, then .env) with the same three states,
# `off`, a regex, or unset. Unset blocks a commit (--staged), since a checkout with no identity configured is the case
# this exists for; in --range it is skipped and said, as CI on a fork's pull request receives no secret. --all scans a
# tree, which has no commits.
if [ -z "${REPO_GUARD_IDENTITY_PATTERNS:-}" ] && [ -f .env ]; then
  REPO_GUARD_IDENTITY_PATTERNS="$(grep '^REPO_GUARD_IDENTITY_PATTERNS=' .env | head -1 | cut -d= -f2- \
    | sed -e "s/^['\"]//" -e "s/['\"]\$//")"
fi
IDENTITY_SETTING="$(printf '%s' "${REPO_GUARD_IDENTITY_PATTERNS:-}" \
  | tr '[:upper:]' '[:lower:]' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
IDENTITY_PATTERNS="${REPO_GUARD_IDENTITY_PATTERNS:-}"
CHECK_IDENTITY=1
# grep exits 2 on a pattern it cannot compile, and a test that reads 2 as "no match" turns the check off while reporting
# clean. So an identity is matched only through identity_match, which tells the three apart, and the error is reported
# without the pattern or grep's own text, either of which would put the list in a public log.
identity_match() {  # 0 match, 1 no match, 2 the pattern is not a valid regex
  local rc
  printf '%s\n' "$1" | grep -Eiq -- "$IDENTITY_PATTERNS" 2>/dev/null
  rc=$?
  [ "$rc" -le 1 ] && return "$rc"
  return 2
}
# One commit's identities: its author, its committer, and every line of its message, since GitHub credits a co-author
# named there (Co-authored-by, read its own way, so not git's trailer parse, which sees only the last paragraph) and a
# listed identity written anywhere in a message is in public history too. A record whose sha is not 40 hex digits means
# the fields shifted (a NUL inside a value), so it is refused rather than read. Prints only the short SHA and the field.
check_identity_record() {  # sha author committer message
  local field value line
  if ! [[ "$1" =~ ^[0-9a-f]{40}$ ]]; then
    echo "repo-guard: BLOCKED [instrument]: an identity record's first field is not a commit SHA (fields shifted?)." >&2
    exit 2
  fi
  for field in author committer; do
    if [ "$field" = author ]; then value="$2"; else value="$3"; fi
    identity_match "$value"
    case $? in
      0) echo "BLOCKED [identity]: commit ${1:0:8}'s $field is on the private identity list"; fail=1 ;;
      2) identity_pattern_invalid ;;
    esac
  done
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    identity_match "$line"
    case $? in
      0) echo "BLOCKED [identity]: commit ${1:0:8}'s message names an identity on the private identity list"; fail=1 ;;
      2) identity_pattern_invalid ;;
    esac
  done <<EOF_MESSAGE
$4
EOF_MESSAGE
}
identity_pattern_invalid() {
  echo "repo-guard: BLOCKED [config]: REPO_GUARD_IDENTITY_PATTERNS is not a valid extended regex; the identity check"
  echo "  cannot run, so nothing passes it. (The pattern is not printed: it is private.)"
  exit 2
}
if [ "$MODE" = "--all" ]; then
  CHECK_IDENTITY=0
elif [ "$IDENTITY_SETTING" = "off" ]; then
  CHECK_IDENTITY=0
  echo "repo-guard: identity check off -- no identity list on this machine."
elif [ -z "$IDENTITY_SETTING" ]; then
  if [ "$MODE" = "--staged" ]; then
    echo "repo-guard: BLOCKED [config]: REPO_GUARD_IDENTITY_PATTERNS is not set."
    echo "Set it in .env (gitignored) or the environment, as REPO_GUARD_NAME_PATTERNS is:"
    echo "  off        you are contributing and hold no list -- the check is skipped"
    echo "  <regex>    the identities that must never author or commit here -- the check runs"
    exit 1
  fi
  # CI sets REPO_GUARD_IDENTITY_REQUIRED on a same-repo push or pull request, where the secret is always given: empty
  # there means it was deleted or renamed, and a green "skipped" would hide that the check stopped (the e2e lane).
  if [ "${REPO_GUARD_IDENTITY_REQUIRED:-}" = "1" ]; then
    echo "repo-guard: BLOCKED [config]: REPO_GUARD_IDENTITY_PATTERNS is not set, and this run requires it (a push or"
    echo "  pull request from this repository receives the secret; is it deleted or renamed?)."
    exit 1
  fi
  CHECK_IDENTITY=0
  echo "repo-guard: identity check skipped -- no identity list here (in CI: a pull request from a fork, which receives"
  echo "  no secret, or a bot's)."
fi

# Secret shapes. Never commit these.
SECRET_PATTERNS='(sk-[A-Za-z0-9]{16,}|AF_SECRET_KEY|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|xox[baprs]-[A-Za-z0-9-]{8,})'

# Hardcoded machine home paths leak the local username. Use $HOME / os.path.expanduser("~")
# or an env var (e.g. MNEMIQ_MINIDEV_DIR) instead. Never a literal /Users/<name> or /home/<name>.
HOME_PATH_PATTERNS='/(Users|home)/[A-Za-z0-9._-]+'

# The scanner itself: free of name tokens now, but its secret-shape patterns
# can match their own spelling. Never scan it.
EXCLUDE_PATH='scripts/repo-guard.sh'

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
    /^__COMMIT__ / { commit = substr($0, 12, 12); header = 0; next }
    # A file header runs from its `diff --` line to its first hunk line (`@@`, or `@@@` combined),
    # and only there is `+++ b/` the file name: read by spelling alone, a content line that began
    # `++ b/` (or `+ b/` in a combined diff) was taken for a header, so the key it carried named a
    # file and was never matched (the Codex review of the fix).
    /^diff --/ { header = 1; next }
    header && /^\+\+\+ b\// { file = substr($0, 7); next }
    header && /^@@/ { header = 0; next }
    header { next }
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

while IFS= read -r -d '' f; do
  [ -z "$f" ] && continue
  # The whole path, not a line of it: a path holding a newline whose second line spelled the
  # excluded one was skipped by a line-wise match (the review gate on the fix).
  [ "$f" = "$EXCLUDE_PATH" ] && continue
  if [ "$MODE" = "--staged" ]; then
    # A gitlink (a submodule added or moved) is a commit in another repository, not a blob here:
    # nothing to scan, and read as a blob it was refused as an instrument fault (the review gate
    # on the fix).
    # `:(literal)`: a bare path here is a pathspec, so a file named `*` took the mode of the first
    # entry the glob matched (the review gate on the fix).
    # `head -c 6`, not `cut -c1-6`: `cut` works line by line, and a gitlink whose path holds a newline
    # gave `160000` followed by the path's second line, which the case did not match, so a healthy
    # submodule was refused as an instrument fault (Codex's review of the fix).
    case "$(git ls-files -s -z -- ":(literal)$f" | head -c 6)" in 160000) continue ;; esac
    # A path the index lists is a blob the index holds; a read that fails is the instrument
    # failing, never a clean file (a path git quoted would fail here, which is why the list
    # above is NUL-delimited).
    if ! git cat-file -e ":0:$f" 2>/dev/null; then
      echo "repo-guard: BLOCKED [instrument]: the index lists '$f' but no blob is read at that path." >&2
      exit 2
    fi
  else
    [ -f "$f" ] || continue
  fi

  # `-a`, not `-I`: a blob with a NUL byte is read as text, so a key shape behind one is matched where
  # `-I` took the whole blob for binary and reported nothing (Codex's review of the fix).
  if [ "$CHECK_NAMES" -eq 1 ] && content_of "$f" | grep -anEi "$NAME_PATTERNS" >/dev/null 2>&1; then
    echo "BLOCKED [proprietary-name]: $f"
    # Safe to echo the match: this branch is unreachable under --all, so the text
    # reaches a terminal on the maintainer's machine and never a public log.
    content_of "$f" | grep -anEi "$NAME_PATTERNS" | head -3 || true
    fail=1
  fi
  if content_of "$f" | grep -anE "$SECRET_PATTERNS" >/dev/null 2>&1; then
    echo "BLOCKED [secret]: $f (matched a secret pattern)"
    fail=1
  fi
  if content_of "$f" | grep -anE "$HOME_PATH_PATTERNS" >/dev/null 2>&1; then
    echo "BLOCKED [home-path]: $f (hardcodes a machine home directory)"
    content_of "$f" | grep -anE "$HOME_PATH_PATTERNS" | head -3 || true
    fail=1
  fi
done < <(list_files)

# The identity checks. Under --staged the identity the commit will carry, which is local output and may name it. Under
# --range each commit's, naming only the commit and the field: a CI log is public, and printing the identity or the
# pattern would publish what the list exists to keep out.
if [ "$CHECK_IDENTITY" -eq 1 ]; then
  identity_match ""
  [ $? -eq 2 ] && identity_pattern_invalid
fi
if [ "$CHECK_IDENTITY" -eq 1 ] && [ "$MODE" = "--staged" ]; then
  for who in AUTHOR COMMITTER; do
    ident="$(git var "GIT_${who}_IDENT" 2>/dev/null | sed -E 's/ [0-9]+ [+-][0-9]{4}$//')"
    identity_match "$ident"
    case $? in
      0) echo "BLOCKED [identity]: this commit's $(printf '%s' "$who" | tr '[:upper:]' '[:lower:]') is on the private identity list: $ident"
         echo "  Set this checkout's public identity (git config user.name and user.email), then commit again."
         fail=1 ;;
      2) identity_pattern_invalid ;;
    esac
  done
fi
if [ "$CHECK_IDENTITY" -eq 1 ] && [ "$MODE" = "--range" ]; then
  if ! expected="$(git rev-list --count "$RANGE" 2>/dev/null)"; then
    echo "repo-guard: BLOCKED [instrument]: git could not read the range '$RANGE' for identities." >&2
    exit 2
  fi
  seen=0
  # Each field NUL-terminated and read without a command substitution, which would drop the NULs: a name may hold a
  # tab, and tab-split fields shifted so an exact pattern matched neither (Codex's review of #290).
  while IFS= read -r -d '' sha && IFS= read -r -d '' author && IFS= read -r -d '' committer \
        && IFS= read -r -d '' message; do
    sha="${sha#$'\n'}"
    seen=$((seen + 1))
    check_identity_record "$sha" "$author" "$committer" "$message"
  done < <(git log -z --format='%H%x00%an <%ae>%x00%cn <%ce>%x00%B' \
             "$RANGE" 2>/dev/null)
  # A git log that failed inside the process substitution reads as no commits; every commit in the range must be read.
  if [ "$seen" -ne "$expected" ]; then
    echo "repo-guard: BLOCKED [instrument]: read the identities of $seen of $expected commit(s) in '$RANGE'." >&2
    exit 2
  fi
  echo "repo-guard: identity: $seen commit(s) checked."
fi
# `--identities`: a pull request's commit identities on stdin, as NUL-terminated records of four fields (sha, author,
# committer, the whole message), each optionally followed by a newline, as scripts/repo-guard-identities.py writes them. identity-guard.yml runs it under pull_request_target, so a fork's pull request
# is checked too: that job runs this repository's code with the secret and reads the identities from GitHub's API, never
# the fork's code. REPO_GUARD_IDENTITY_EXPECTED is the pull request's own commit count; fewer identities read fails.
if [ "$MODE" = "--identities" ] && [ "$CHECK_IDENTITY" -eq 1 ]; then
  if ! [[ "${REPO_GUARD_IDENTITY_EXPECTED:-}" =~ ^[0-9]+$ ]]; then
    echo "repo-guard: BLOCKED [config]: --identities needs REPO_GUARD_IDENTITY_EXPECTED, the pull request's commit count." >&2
    exit 2
  fi
  seen=0
  while IFS= read -r -d '' sha && IFS= read -r -d '' author && IFS= read -r -d '' committer \
        && IFS= read -r -d '' message; do
    sha="${sha#$'\n'}"
    seen=$((seen + 1))
    check_identity_record "$sha" "$author" "$committer" "$message"
  done
  expected="$REPO_GUARD_IDENTITY_EXPECTED"
  if [ "$seen" -ne "$expected" ]; then
    echo "repo-guard: BLOCKED [instrument]: read the identities of $seen of $expected commit(s) the pull request has." >&2
    exit 2
  fi
  echo "repo-guard: identity: $seen commit(s) checked."
fi

if [ "$MODE" = "--identities" ]; then
  if [ "$fail" -ne 0 ]; then
    echo "repo-guard: blocked: a commit's author or committer is on the private identity list."
    exit 1
  fi
  echo "repo-guard: identity clean."
  exit 0
fi
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
