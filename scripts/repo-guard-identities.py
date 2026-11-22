"""Turn a pull request's commits, as JSON lines, into the NUL-terminated records `repo-guard.sh --identities` reads.

    gh api ... --jq '.commits[] | {sha, a: ..., c: ..., m: .commit.message}' | python3 -I scripts/repo-guard-identities.py

Each input line is one commit: `sha`, `a` (author), `c` (committer), `m` (the whole message). Each output record is
sha, author, committer and message, each NUL-terminated. A value that could shift those fields is refused, the whole
input with it (exit 1, nothing written): a NUL anywhere, a newline in a name, or a sha that is not 40 hex digits. JSON
can carry a NUL (`\\u0000`), and one inside a name split it into two fields, moving every later field by one so a listed
identity landed in the sha slot, which is never matched (the e2e lane's review of #290). Prints no value: the commits
belong to a pull request, and this runs in a public log.
"""
from __future__ import annotations

import json
import re
import sys

SHA = re.compile(r"[0-9a-f]{40}")


def main() -> int:
    out = []
    for n, line in enumerate(sys.stdin, 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            sha, author, committer, message = r["sha"], r["a"], r["c"], r.get("m") or ""
        except (ValueError, KeyError, TypeError):
            print(f"repo-guard-identities: input line {n} is not a commit record", file=sys.stderr)
            return 1
        if not (isinstance(sha, str) and SHA.fullmatch(sha)):
            print(f"repo-guard-identities: input line {n} has no 40-hex sha", file=sys.stderr)
            return 1
        names = (author, committer)
        if not all(isinstance(v, str) for v in (*names, message)) or any("\0" in v for v in (*names, message)) \
                or any("\n" in v or "\r" in v for v in names):
            print(f"repo-guard-identities: input line {n} holds a value that could shift fields", file=sys.stderr)
            return 1
        out.append(f"{sha}\0{author}\0{committer}\0{message}\0".encode())
    sys.stdout.buffer.write(b"".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
