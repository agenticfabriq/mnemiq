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
# Best-effort. Captured as RAW TEXT and parsed defensively below -- `curl -sS` without `-f` exits 0
# on a 404 and hands back the error page, and a --max-time abort leaves a truncated body, so
# parsing here would abort the script under `set -e` AFTER the archive has moved the cache aside.
VERSION_BODY="$(curl -sS --max-time 10 -w '\n%{http_code}' "${BASE%/}/../version" 2>/dev/null || echo '')"
VERSION_CODE="$(printf '%s' "$VERSION_BODY" | tail -n1)"
VERSION_JSON="$(printf '%s' "$VERSION_BODY" | sed '$d')"

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
python3 - "$PROV" "$SERVED" "$BASE" "$MODELS_JSON" "$VERSION_JSON" "$VERSION_CODE" <<'PY'
import json, os, re, subprocess, sys, datetime
path, served, base, models, version_raw, version_code = sys.argv[1:7]
models_obj = json.loads(models)

# A NAME IS NOT A VERSION. `qwen` is what the July judge recorded, and it is why 22/186 cannot be
# reproduced -- the same weights could be re-served under it forever, or different ones tomorrow.
# So this looks for something IMMUTABLE and, finding none, says so in the record rather than
# letting a filled-in `served_model` read as provenance.
def immutable_id():
    # Operator-supplied, and SCOPED to a model: `MNEMIQ_JUDGE_MODEL_VERSION=<served_model>=<version>`.
    # An unscoped value exported once in a shell profile would mark every later sweep, against any
    # model on any endpoint, as versioned -- stamping the exact unreproducibility this script exists
    # to close. A value naming a different model than the one being served is ignored, loudly.
    env = os.environ.get("MNEMIQ_JUDGE_MODEL_VERSION", "").strip()
    if env:
        scope, sep, value = env.partition("=")
        if not sep:
            print(f"  IGNORING MNEMIQ_JUDGE_MODEL_VERSION={env!r}: expected <served_model>=<version>")
        elif scope != served:
            print(f"  IGNORING MNEMIQ_JUDGE_MODEL_VERSION: scoped to {scope!r}, serving {served!r}")
        elif not value.strip():
            # `export X="$SERVED=$SHA"` with SHA unset gives `qwen=`, which is truthy and scoped.
            # Returning "" from here sets `model_version_recorded: true` on an empty version while
            # the console prints the unversioned warning -- record and operator told opposite things.
            print(f"  IGNORING MNEMIQ_JUDGE_MODEL_VERSION={env!r}: version half is empty")
        else:
            return value, f"MNEMIQ_JUDGE_MODEL_VERSION, scoped to {scope!r}"
    # Only a snapshot PATH counts: it ties the hex to the weights on disk. A bare 40-hex anywhere in
    # the payload is not evidence -- a permission id, a lora directory or a server-generated id would
    # all match, and be recorded as the model's version under a label naming a source it never had.
    # Scoped to the entry actually being served. Searching the whole payload would take a LORA
    # adapter's snapshot sha -- a second `data` entry with `parent: "qwen"` -- and record it as the
    # base model's version, which the label would then vouch for.
    entry = next((e for e in models_obj.get("data", []) if e.get("id") == served), None)
    if entry is None:
        return None, None
    m = re.search(r"snapshots/([0-9a-f]{40})", json.dumps(entry))
    return (m.group(1), f"HF snapshot sha on the /v1/models entry for {served!r}") if m else (None, None)

value, source = immutable_id()

# Never fatal, never silently wrong: a body that will not parse is kept verbatim so a reader can
# see what the endpoint actually said.
# A JSON error body parses. `{"detail":"Not Found"}` is structurally a version response, so without
# the status a reader cannot tell it from `{"version": "0.9.0.1"}` -- `curl -sS` without `-f` exits
# 0 on a 404 and hands back the error page.
parse_ok = False
try:
    parsed = json.loads(version_raw) if version_raw.strip() else None
    parse_ok = parsed is not None
except (ValueError, TypeError):
    parsed = {"unparsed_response": version_raw[:500]}
# `ok` needs BOTH: `%{http_code}` reports the STATUS LINE, not transfer completion, so a --max-time
# abort mid-body still yields 200 with a truncated payload. Status alone would certify that as
# captured -- the same "said yes on the wrong evidence" shape as the rest of this block.
server_version = {"http_status": version_code or None, "body": parsed,
                  "ok": version_code == "200" and parse_ok}
rev = subprocess.run(["git","rev-parse","--short","HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git","status","--porcelain"], capture_output=True, text=True).stdout.strip())
json.dump({"served_model": served, "base_url": base, "engine_rev": rev + ("-dirty" if dirty else ""),
           "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
           "model_version": value, "model_version_source": source,
           "model_version_recorded": value is not None,
           "scoring_complete": False,   # flipped only after the sweep returns; see below
           "server_version": server_version,
           "models_endpoint": models_obj}, open(path, "w"), indent=2)
print(f"  provenance -> {path}")
if value:
    print(f"  model version: {value}  ({source})")
else:
    print("  WARNING: no immutable model version available. `served_model` is a NAME, and a name is")
    print("           what made the July figures unreproducible. The record says so")
    print("           (`model_version_recorded: false`); set MNEMIQ_JUDGE_MODEL_VERSION to fix it.")
PY

echo "== scoring =="
MNEMIQ_VERIFY_BASE_URL="$BASE" MNEMIQ_VERIFY_MODEL="$SERVED" \
MNEMIQ_VERIFY_API_KEY="${MNEMIQ_VERIFY_API_KEY:-EMPTY}" \
  .venv/bin/python scripts/run_verify_replay.py "$RUN" --judge

echo "== check the run was not a cache replay =="
JUDGE_MODELS_URL="${BASE%/}/models" .venv/bin/python - "$CACHE" "$RUN" "$PROV" <<'PY'
import json, os, sys, pathlib
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

# A PARTIAL outage is not detectable from the scores -- a mid-sweep death leaves real scores
# followed by fallbacks that no value test can separate from genuine ones. The post-sweep probe is
# WEAK evidence, and contamination is never ruled out by EITHER result -- a 429 burst or per-request
# timeout leaves the judge writing its constant into the cache while `/v1/models` keeps answering.
# So the exit on FALSE is precautionary, not inferential: it is not that a dead endpoint proves the
# scores are bad, but that certifying a run which also carries a known-bad signal is worse than
# declining one that might have been fine. `scoring_complete` is the field a later reader trusts,
# and the cost of withholding it is a re-run. `LLMClient.complete` turns
# every API error into `ModelUnavailable` -- timeout, rate limit, bad gateway -- and the judge
# swallows all of them to its constant, so a 429 burst or per-request timeout leaves `/v1/models`
# answering seconds later with contaminated scores already in the cache. True here is not evidence
# of clean scores.
import urllib.error, urllib.request
try:
    with urllib.request.urlopen(os.environ["JUDGE_MODELS_URL"], timeout=10) as r:
        healthy_after = r.status == 200
except (urllib.error.URLError, OSError, KeyError, ValueError):
    healthy_after = False
print(f"  endpoint still answering after the sweep: {healthy_after}")
if not healthy_after:
    print("  FAIL: it is not. That may mean it died mid-sweep -- leaving this judge's error")
    print("        constant in the cache, which no value test separates from real scores -- or")
    print("        simply that it was taken down after a clean run. Neither can be ruled out")
    print("        from here, so the sweep is not certified. Re-probe and re-run to settle it.")
    raise SystemExit(1)
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

# A FULL CACHE IS NOT A SUCCESSFUL SWEEP. `SemanticJudge.score` catches every exception and returns
# 1.0 -- deliberately, so the product degrades to answering rather than crashing. In a MEASUREMENT
# that fail-open is indistinguishable from a judge that approved everything: a dead endpoint yields
# one 1.0 per case, a cache of exactly the right size, and a sweep reporting "0 wrong caught at
# every threshold", which reads as a finding rather than as an outage. Refuse to certify it.
scores = list(json.load(cache_p.open()).values())
distinct = len(set(scores))
at_one = sum(1 for v in scores if v == 1.0)
print(f"  score distribution: {distinct} distinct value(s), {at_one}/{len(scores)} at exactly 1.0")
# Keyed on distinct == 1, NOT on the fallback's value. Hardcoding 1.0 here duplicates a literal
# from `SemanticJudge.score` with nothing linking them: change that fallback to 0.0 and an
# all-0.0 outage would pass, certified, reporting every wrong answer caught. Any single-valued
# sweep over hundreds of cases is a red flag whatever the value is.
if distinct == 1:
    print(f"  FAIL: every score is the same value ({scores[0]!r}). This judge returns a constant on")
    print("        error, so a dead endpoint produces exactly this shape. Not certifying the sweep.")
    raise SystemExit(1)
prov = json.load(prov_p.open())
prov["scoring_complete"] = True
prov["cache_file"] = cache_p.name
prov["cache_entries"] = n_cache
prov["score_distinct_values"] = distinct
# NOT "at_fallback": `SemanticJudge` clamps a parsed confidence with min(1.0, ...), so a healthy
# judge replying 1.0 is indistinguishable BY VALUE from its error path. This counts what it says
# and no more -- claiming these were fallbacks would assert a provenance the number cannot carry.
prov["scores_at_exactly_1_0"] = at_one
prov["endpoint_healthy_after_sweep"] = healthy_after
json.dump(prov, prov_p.open("w"), indent=2)
print(f"  OK: fresh cache under the served model's tag; {prov_p.name} marked complete")
PY
