#!/usr/bin/env bash
# Source this to run mnemiq's pipeline with GENERATION on a local vLLM model while
# everything else stays on the hosted endpoint. Usage:
#
#   set -a; source .env; set +a          # Settings() reads os.environ only, no env_file
#   source scripts/slm-pipeline-env.sh http://HOST:8003/v1 arctic
#
# THE ORDER MATTERS. `embed_base_url` and `embed_api_key` default to the CHAT values,
# so repointing chat at vLLM drags embeddings along with it -- to a server that has no
# embedding model, which fails retrieval rather than reporting a misconfiguration.
# Pin embeddings (and enrichment) to the hosted endpoint FIRST, then move chat.
#
# Enrichment is pinned for a second reason: this arm exists to vary the GENERATOR and
# nothing else. Let the local model enrich as well and two variables move at once.
# BLOCKER this script earned: it is mode 755 with a shebang, so `./slm-pipeline-env.sh`
# runs it in a CHILD shell -- it prints all three confirmation lines, exits 0, and the
# parent's MNEMIQ_LLM_* stay on the hosted model. The operator then runs the "local SLM"
# arm against the hosted endpoint holding a receipt that says otherwise. Refuse that.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "slm-pipeline-env.sh must be SOURCED, not executed:" >&2
  echo "  source ${BASH_SOURCE[0]} <base-url> <model-name>" >&2
  echo "Executing it changes nothing in your shell." >&2
  exit 64
fi

# `set -u` is deliberately NOT used: this file is sourced, and leaving nounset on would
# persist into the operator's interactive shell for the rest of the session.

_base="${1:?usage: source slm-pipeline-env.sh <base-url> <model-name>}"
_model="${2:?usage: source slm-pipeline-env.sh <base-url> <model-name>}"

: "${MNEMIQ_LLM_BASE_URL:?source .env first}"

# 1. hold embeddings where they already work
export MNEMIQ_EMBED_BASE_URL="${MNEMIQ_EMBED_BASE_URL:-$MNEMIQ_LLM_BASE_URL}"
export MNEMIQ_EMBED_API_KEY="${MNEMIQ_EMBED_API_KEY:-$MNEMIQ_LLM_API_KEY}"

# 2. hold the semantic layer constant
export MNEMIQ_ENRICH_BASE_URL="${MNEMIQ_ENRICH_BASE_URL:-$MNEMIQ_LLM_BASE_URL}"
export MNEMIQ_ENRICH_API_KEY="${MNEMIQ_ENRICH_API_KEY:-$MNEMIQ_LLM_API_KEY}"
export MNEMIQ_ENRICH_MODEL="${MNEMIQ_ENRICH_MODEL:-$MNEMIQ_LLM_MODEL}"

# 3. the judge follows chat too -- `verify_endpoint()` defaults by the same `or
#    llm_base_url` rule as `embed_endpoint()`, so under MNEMIQ_VERIFY=1 it would ride
#    onto the local model and move a second variable.
export MNEMIQ_VERIFY_BASE_URL="${MNEMIQ_VERIFY_BASE_URL:-$MNEMIQ_LLM_BASE_URL}"
export MNEMIQ_VERIFY_API_KEY="${MNEMIQ_VERIFY_API_KEY:-$MNEMIQ_LLM_API_KEY}"
# The judge's MODEL resolves separately (`verify_model or llm_model`), so pinning only
# url+key would ask the hosted endpoint for the local served-model-name.
export MNEMIQ_VERIFY_MODEL="${MNEMIQ_VERIFY_MODEL:-$MNEMIQ_LLM_MODEL}"

# 4. only now move generation
export MNEMIQ_LLM_BASE_URL="$_base"
export MNEMIQ_LLM_MODEL="$_model"
export MNEMIQ_LLM_API_KEY="${MNEMIQ_LOCAL_API_KEY:-local}"   # vLLM ignores it; Settings requires one

echo "generation -> $MNEMIQ_LLM_MODEL @ $MNEMIQ_LLM_BASE_URL"
echo "embeddings -> hosted (unchanged)"
echo "enrichment -> hosted (unchanged)"
echo "verifier   -> hosted (unchanged)"
unset _base _model
