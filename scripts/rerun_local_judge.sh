#!/usr/bin/env bash
# Re-run the LOCAL judge sweep under RECORDED conditions (inventory item 14).
#
# Two things this exists to prevent, both of which have already happened once:
#
#   1. A SILENT NO-OP. `_CachingJudge` returns a cached score without calling the model, and its
#      key is sha1(model \0 question \0 sql). If the endpoint serves the same model string as the
#      stale cache, every key hits and the "re-run" replays the old scores at full speed. This
#      moves the old cache aside first and then asserts the new one was actually written.
#   2. AN UNRECORDED JUDGE. The July figures cannot be reproduced because nothing recorded which
#      local model produced them -- the cache tag carries a bare model string, no version. This
#      captures what /v1/models reports, beside the run, before scoring anything.
#
# Usage:  scripts/rerun_local_judge.sh <run.jsonl> [base_url]
set -euo pipefail

RUN="${1:?usage: rerun_local_judge.sh <run.jsonl> [base_url]}"
BASE="${2:-http://localhost:8000/v1}"
[ -f "$RUN" ] || { echo "no such run: $RUN" >&2; exit 2; }

echo "== endpoint =="
MODELS_JSON="$(curl -sS --max-time 10 "${BASE%/}/models")" || { echo "endpoint unreachable: $BASE" >&2; exit 3; }
SERVED="$(printf '%s' "$MODELS_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
echo "  serving: $SERVED"

TAG="$(python3 -c 'import sys; print("".join(c if c.isalnum() else "_" for c in sys.argv[1]))' "$SERVED")"
CACHE="${RUN}.judgecache.${TAG}.json"
PROV="${RUN}.judgeprov.${TAG}.json"

# ARCHIVE FIRST, then write the new provenance. Both files carry a stable per-(run, model) name,
# so writing the new provenance before this block would truncate the previous sweep's -- and then
# archive the file that has just been written, filing the NEW judge's identity as if it described
# the OLD scores. Order is the whole guarantee here.
# Archive the cache and its provenance TOGETHER, under one timestamp. They are linked by name,
# so moving only the cache leaves the archived scores with no provenance and leaves a live
# provenance file describing a sweep whose scores are no longer on disk -- and the next run then
# truncates it. Losing the judge's identity is the second failure this script exists to prevent,
# so it must not be lost by the preventing.
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [ -f "$CACHE" ]; then
  mv "$CACHE" "${CACHE}.superseded.${STAMP}"
  echo "  archived stale cache -> $(basename "${CACHE}.superseded.${STAMP}")"
  echo "  (had $(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${CACHE}.superseded.${STAMP}") entries)"
  if [ -f "$PROV" ]; then
    mv "$PROV" "${PROV}.superseded.${STAMP}"
    echo "  archived its provenance -> $(basename "${PROV}.superseded.${STAMP}")"
  else
    echo "  NOTE: that cache had no provenance file -- it predates this script"
  fi
fi

# Record the judge's identity beside the run. Named after the CACHE TAG, not a timestamp, so the
# provenance and the scores it describes are linked by NAME -- a timestamped file links to nothing,
# and if scoring aborts, "the most recent provenance" then describes a run that produced no scores
# while the surviving cache belongs to an earlier one. That is the July failure mode this script
# exists to close, so it must not be reintroduced by the closing.
python3 - "$PROV" "$SERVED" "$BASE" "$MODELS_JSON" <<'PY'
import json, subprocess, sys, datetime
path, served, base, models = sys.argv[1:5]
rev = subprocess.run(["git","rev-parse","--short","HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git","status","--porcelain"], capture_output=True, text=True).stdout.strip())
json.dump({"served_model": served, "base_url": base, "engine_rev": rev + ("-dirty" if dirty else ""),
           "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
           "scoring_complete": False,   # flipped only after the sweep returns; see below
           "models_endpoint": json.loads(models)}, open(path, "w"), indent=2)
print(f"  provenance -> {path}")
PY

echo "== scoring =="
MNEMIQ_VERIFY_BASE_URL="$BASE" MNEMIQ_VERIFY_MODEL="$SERVED" \
MNEMIQ_VERIFY_API_KEY="${MNEMIQ_VERIFY_API_KEY:-EMPTY}" \
  .venv/bin/python scripts/run_verify_replay.py "$RUN" --judge

echo "== check the run was not a cache replay =="
.venv/bin/python - "$CACHE" "$RUN" "$PROV" <<'PY'
import json, sys, pathlib
# the venv, not system python3: this imports mnemiq, and the scoring step above already uses it
from mnemiq.eval.verify_replay import _ANSWERABLE
cache_p, run_p, prov_p = (pathlib.Path(a) for a in sys.argv[1:4])
if not cache_p.exists():
    print("  FAIL: no cache written -- the judge never scored"); raise SystemExit(1)
n_cache = len(json.load(cache_p.open()))
# Compared against ANSWERABLE records, not every line: the judge scores only those, and the cache
# dedupes on (model, question, sql). Comparing against the raw line count always looks short by
# the deferred cases and reads as if a complete run had skipped hundreds of rows.
rows = [json.loads(l) for l in run_p.open() if l.strip()]
n_answerable = sum(1 for r in rows if r["outcome"] in _ANSWERABLE)
# On mini-dev every record is answerable, so these two coincide; the distinction is kept because
# the script takes a run path and other corpora defer.
print(f"  cache entries {n_cache} over {n_answerable} answerable records ({len(rows)} rows total)")
# n_answerable is printed for the operator; the ASSERTION below uses the distinct-pair count.
# The cache dedupes on (model, question, sql), so the exact expected size is the number of
# DISTINCT (question, sql) pairs among answerable records -- not the record count, which
# double-counts any repeated pair. Comparing against it is what makes this a completeness check:
# testing only for "not empty" would stamp a sweep that scored ONE case as complete.
expected = len({(r["question"], r.get("sql") or "") for r in rows if r["outcome"] in _ANSWERABLE})
# Defensive, and unreachable under today's control flow: `set -e` aborts on a failed sweep, and
# a sweep that returns has scored every answerable record. It exists because the alternative --
# testing only for a non-empty cache -- would certify a one-case sweep as complete the moment
# that stops being true, and `scoring_complete` is the one field a later reader trusts.
if n_cache < expected:
    print(f"  FAIL: {n_cache} scored of {expected} distinct answerable (question, sql) pairs "
          f"-- partial sweep, provenance left marked incomplete")
    raise SystemExit(1)
prov = json.load(prov_p.open())
prov["scoring_complete"] = True
prov["cache_file"] = cache_p.name
prov["cache_entries"] = n_cache
json.dump(prov, prov_p.open("w"), indent=2)
print(f"  OK: fresh cache under the served model's tag; {prov_p.name} marked complete")
PY
