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

# Record the judge's identity beside the run, BEFORE scoring -- this is the gap that made the
# July numbers unreproducible, and it is item 15's rule applied to the judge.
PROV="${RUN}.judgeprov.$(date -u +%Y%m%dT%H%M%SZ).json"
python3 - "$PROV" "$SERVED" "$BASE" "$MODELS_JSON" <<'PY'
import json, subprocess, sys, datetime
path, served, base, models = sys.argv[1:5]
rev = subprocess.run(["git","rev-parse","--short","HEAD"], capture_output=True, text=True).stdout.strip()
dirty = bool(subprocess.run(["git","status","--porcelain"], capture_output=True, text=True).stdout.strip())
json.dump({"served_model": served, "base_url": base, "engine_rev": rev + ("-dirty" if dirty else ""),
           "captured_at": datetime.datetime.now(datetime.UTC).isoformat(),
           "models_endpoint": json.loads(models)}, open(path, "w"), indent=2)
print(f"  provenance -> {path}")
PY

TAG="$(python3 -c 'import sys; print("".join(c if c.isalnum() else "_" for c in sys.argv[1]))' "$SERVED")"
CACHE="${RUN}.judgecache.${TAG}.json"
if [ -f "$CACHE" ]; then
  STALE="${CACHE}.superseded.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$CACHE" "$STALE"
  echo "  moved stale cache aside -> $(basename "$STALE")"
  echo "  (had $(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$STALE") entries)"
fi

echo "== scoring =="
MNEMIQ_VERIFY_BASE_URL="$BASE" MNEMIQ_VERIFY_MODEL="$SERVED" \
MNEMIQ_VERIFY_API_KEY="${MNEMIQ_VERIFY_API_KEY:-EMPTY}" \
  .venv/bin/python scripts/run_verify_replay.py "$RUN" --judge

echo "== check the run was not a cache replay =="
python3 - "$CACHE" "$RUN" <<'PY'
import json, sys, pathlib
cache_p, run_p = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
if not cache_p.exists():
    print("  FAIL: no cache written -- the judge never scored"); raise SystemExit(1)
n_cache = len(json.load(cache_p.open()))
n_run = sum(1 for line in run_p.open() if line.strip())
print(f"  cache entries {n_cache} over {n_run} run rows")
if n_cache == 0:
    print("  FAIL: empty cache"); raise SystemExit(1)
print("  OK: a fresh cache was written under the served model's own tag")
PY
