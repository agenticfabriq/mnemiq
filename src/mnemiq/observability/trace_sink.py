"""Emit one governed trace per answer to Verity.

Verity's five trace tables were empty and its CAPTURE/OBSERVE screens starved: the ingest path
validates, dedupes, enqueues and grades, and had never received a row because nothing sent one.
This is the sender.

**The payload is a disclosure decision, not plumbing.** An answer carries result rows, prose that
restates values, SQL embedding literals from the question, the question itself, and the identity of
whoever asked. So the request is built FROM THE TIER TABLE below rather than from the event: a field
nobody classified cannot ship, because "the event carries it" is what code does when nobody decided.
Adding a field to `IdentityContext` therefore ships it OFF until someone puts it in a tier.

**The boundary is enforced here, before the request leaves.** Not by Verity discarding on receipt --
filtering at the receiver means the bytes already crossed the wire, already sat in a log, already
existed on a second machine. "We do not store it" and "we do not send it" are different claims and
only the second survives a security review.

Two placements are load-bearing and both were learned the hard way:

* the answer goes in an **`answer_text` artifact**, never the top-level `answer` field. That is what
  every Verity grader reads (`artifact_payload(trace, "answer_text")`) AND what Verity redacts for
  identities without reviewer/admin/operator. Gradeable and protected are the same place (D132).
* `semantic_refs` names what the PACKET selected, not what the snapshot held. `certified_refs` is
  snapshot-scoped and retrieval takes a subset per question, so shipping the snapshot's list would
  report what was *available* as what was *used* -- and Verity's D131 reader would republish that as
  fact.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from mnemiq.enrichment.verity_auth import access_token

logger = logging.getLogger(__name__)


@dataclass
class AnswerEvent:
    """One answered question, as the runtime saw it.

    Everything the emitter may send is reachable from here -- and every field of it appears in
    exactly one tier in `_build_trace`, which is what makes the tier table binding rather than
    advisory.
    """

    source_id: str
    question: str
    identity: Any
    answer: Any
    elapsed_ms: float
    packet: Any = None  # the retrieval packet, for what the answer actually USED
    events: list[dict] = field(default_factory=list)


class TraceSink:
    """A sink for whole answers, beside (not instead of) the metrics sink.

    Two jobs, two seams: `ObservabilitySink` is a counter that wants six scalars fast; this batches,
    crosses a network to another product, tolerates latency and may retry. Fusing them would make
    every future trace field justify itself to a counter.
    """

    def record_answer(self, event: AnswerEvent) -> None:  # pragma: no cover - protocol
        raise NotImplementedError


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


class VerityTraceSink(TraceSink):
    """Post one trace per answer, fail-soft, with the tier table as the only source of payload."""

    def __init__(self, settings: Any) -> None:
        self._settings = settings
        # M18's lesson: a fail-soft that produces nothing must still be countable. M54 is the next
        # turn of it -- countable is not enough when three unrelated failures share the count.
        #
        # `dropped` stays the TOTAL so an operator asking "is anything failing" watches one number
        # and does not have to sum three. The breakdown answers the second question, "which", and
        # the three want opposite responses:
        #
        #   unbuildable  this engine could not build the payload   -> our BUG, never self-heals
        #   rejected     the receiver understood us and said no    -> our CONFIG, permanent
        #   unavailable  the receiver is broken or absent          -> their OUTAGE, self-heals
        #
        # Measured live: a 401 from Verity's DEFAULT auth mode incremented the same counter a dead
        # socket does, so a deployment emitting zero traces forever looked exactly like one whose
        # receiver was briefly down. The cause was in the per-event log and absent from the
        # aggregate, which is the channel anyone actually alerts on.
        self.dropped = 0
        self.dropped_unbuildable = 0
        self.dropped_rejected = 0
        self.dropped_unavailable = 0

    # -- tier gates ------------------------------------------------------------------------------
    def _send_text(self) -> bool:
        """Question text, SQL, answer prose, deferral message -- all carry literals."""
        return bool(getattr(self._settings, "verity_trace_send_text", False))

    def _send_identity_detail(self) -> bool:
        """email / roles / groups / attributes. None is needed for audit; all are free riders, and
        `attributes` is an open dict a DEPLOYMENT fills, so what travels is whatever some future
        integrator stapled to an identity."""
        return bool(getattr(self._settings, "verity_trace_send_identity_detail", False))

    def _send_rows(self) -> bool:
        """Result rows. The deepest tier: this is the customer's data, not a description of it."""
        return bool(getattr(self._settings, "verity_trace_send_rows", False))

    # -- payload ---------------------------------------------------------------------------------
    def _semantic_refs(self, event: AnswerEvent) -> list[dict]:
        """The certified versions THIS ANSWER used: the packet's selection ∩ the snapshot's refs."""
        packet = event.packet
        snapshot_refs = getattr(event, "certified_refs", None) or []
        if packet is None or not snapshot_refs:
            return []
        selected: set[tuple[str, str]] = set()
        for object_type, attr in (
            ("definition", "definitions"),
            ("metric", "metrics"),
            ("dimension", "dimensions"),
        ):
            for obj in getattr(packet, attr, []) or []:
                object_id = getattr(obj, "id", None)
                if object_id:
                    selected.add((object_type, str(object_id)))
        return [
            {
                "object_type": ref.object_type,
                "object_id": ref.object_id,
                "version_hash": ref.version_hash,
            }
            for ref in snapshot_refs
            if (ref.object_type, ref.object_id) in selected
        ]

    def _build_trace(self, event: AnswerEvent) -> dict:
        answer = event.answer
        identity = event.identity
        trace = getattr(answer, "trace", None)

        # ---- ALWAYS: no field here restates a value from the customer's database ---------------
        record: dict[str, Any] = {
            "trace_id": _sha256(f"{identity.tenant_id}|{event.source_id}|{id(answer)}")[7:39],
            "tenant_id": identity.tenant_id,
            "source_id": event.source_id,
            "source_system": "mnemiq",
            "agent": {"system": "mnemiq", "name": event.source_id, "version": _version()},
            "identity": {
                # A person-identifier, and labelled as one. It earns the always tier by the test
                # applied to semantic_refs: an audit store that cannot say WHO asked answers
                # nothing. `roles` is NOT here -- grant_fingerprint already carries the
                # authorization boundary, so roles duplicate a fact already present while
                # disclosing group membership on their own.
                "principal_id": identity.principal_id,
                "roles": [],
                "groups": [],
            },
            "question": {
                # A correlation key, NOT a privacy measure: questions are low-entropy and
                # enumerable, so a hash confirms a guess. Labelled rather than salted, because a
                # weak mitigation invites the misplaced trust an honest label does not.
                "text": "",
                "hash": _sha256(event.question),
            },
            "resolved_intent": {
                "mode": getattr(answer, "mode", None),
                "deferred": bool(getattr(answer, "deferred", False)),
                "failed": bool(getattr(answer, "failed", False)),
                # The ENUM, never the message: the message is free text carrying schema and values,
                # and now the provider's own error string too.
                "reason_code": _reason_code(answer),
                "enrichment_version": getattr(trace, "enrichment_version", None),
                "candidates_executed": getattr(answer, "candidates_executed", None),
                "judge_engaged": getattr(answer, "judge_engaged", None),
                "agreement": getattr(answer, "agreement", None),
                "cached": getattr(answer, "cached", None),
            },
            # Stage names and durations. Metadata by construction, and what Live Traces renders.
            "events": [_to_trace_event(e, i) for i, e in enumerate(event.events)],
            "artifacts": [],
            # ---- ALWAYS: what the answer READ, and whether that is the whole story ----------
            # M56. Deliberately NOT in the text tier: lineage beside the executed SQL would reach
            # a text-enabled deployment and nothing else, and the deployments most likely to keep
            # text off are the ones most likely to need an audit trail. Object ids are not the
            # question's literals or the answer's prose -- they are the audit question itself.
            "lineage": {
                "tables": list(getattr(trace, "tables_used", []) or []),
                "completeness": getattr(trace, "lineage_completeness", "unknown") or "unknown",
                "unresolved": list(getattr(trace, "lineage_unresolved", []) or []),
                "reasons": list(getattr(trace, "lineage_reasons", []) or []),
            },
            "semantic_refs": self._semantic_refs(event),
            # ---- ALWAYS: what the access decision did to WHICH OBJECT ----------------------
            # The field existed and was always `[]`, next to a `grant_fingerprint` whose own
            # comment calls itself "a hash of the policy, not the policy" -- so the store could
            # tell two answers apart under different grants and could not say what either grant
            # DID. That is the M56 question one layer on, and this is its answer.
            #
            # SHAPED TO THE STORE, which is a typed contract and not a free-form bag:
            # `trace_schema::PolicyDecisionV1` requires `policy_id` and `effect` as strings and
            # takes `metadata` as free JSON. A first version of this emitted
            # `{object, rows, columns}` and `null` for the not-evaluated case; the field is
            # `Vec<PolicyDecisionV1>` with `#[serde(default)]`, which covers a MISSING key and not
            # an explicit null, so that version rejected the whole trace on the governed path --
            # storing nothing where the old empty list at least stored the rest.
            #
            # `policy_id` is a constant because mnemiq does not know one: the decision comes from
            # the provider, and `Narrowing` deliberately carries no policy identity. Inventing an
            # id here would put a fabricated identifier into an audit record.
            "policy_decisions": _policy_decisions(trace, answer),
            "collector_metadata": {
                "elapsed_ms": round(event.elapsed_ms, 3),
                # A hash of the policy, not the policy. Two answers to one question under different
                # grants are different events and must not be indistinguishable in the store.
                "grant_fingerprint": getattr(answer, "grant_fingerprint", None),
                # `policy_decisions` cannot carry this: it is a typed Vec, so `[]` is the only
                # empty it can express and "evaluated, narrowed nothing" would be identical to
                # "never evaluated". The marker lives here, in the one part of the record that is
                # free-form JSON on the store side.
                "access_evaluated": _decision(trace, answer) is not None,
            },
            "captured_at": _now(),
            "idempotency_key": "",
        }
        record["idempotency_key"] = f"{record['trace_id']}-v1"

        # ---- OPT-IN: text that restates or embeds values --------------------------------------
        if self._send_text():
            record["question"]["text"] = event.question
            record["executed_sql"] = getattr(trace, "target_sql", None) or None
            answer_text = str(getattr(answer, "answer", "") or "")
            if answer_text:
                record["artifacts"].append({
                    "artifact_id": f"{record['trace_id']}-answer",
                    "kind": "answer_text",
                    "sha256": _sha256(answer_text),
                    "redaction_state": "redacted",
                    "payload": {"answer": answer_text},
                })

        if self._send_identity_detail():
            record["identity"].update({
                "email": getattr(identity, "email", None),
                "roles": list(getattr(identity, "roles", []) or []),
                "groups": list(getattr(identity, "groups", []) or []),
            })
            attributes = dict(getattr(identity, "attributes", {}) or {})
            if attributes:
                record["collector_metadata"]["identity_attributes"] = attributes

        # ---- DEEPEST OPT-IN: the customer's rows ----------------------------------------------
        if self._send_rows():
            preview = getattr(answer, "preview", None)
            if preview is not None:
                record["artifacts"].append({
                    "artifact_id": f"{record['trace_id']}-rows",
                    "kind": "result_sample",
                    "sha256": _sha256(json.dumps(preview.rows, default=str, sort_keys=True)),
                    "redaction_state": "redacted",
                    "payload": {
                        "columns": list(preview.columns),
                        "rows": preview.rows,
                        "row_count": preview.row_count,
                        "truncated": preview.truncated,
                    },
                })
        return record

    # -- transport -------------------------------------------------------------------------------
    def _drop(self, cause: str, exc: BaseException) -> None:
        """Count a drop against its cause AND the total, and name the cause in the log.

        One place, because the two increments must never disagree about what happened -- the
        counter and the breakdown drifting is the same defect one level down from the one this
        exists to fix.
        """
        self.dropped += 1
        setattr(self, f"dropped_{cause}", getattr(self, f"dropped_{cause}") + 1)
        logger.warning("verity trace dropped (%s; %d total); answer unaffected: %s",
                       cause, self.dropped, exc)

    def record_answer(self, event: AnswerEvent) -> None:
        url = getattr(self._settings, "verity_traces_url", None)
        if not url:
            return
        try:
            body = json.dumps({"traces": [self._build_trace(event)]}).encode()
        except Exception as exc:  # a payload we cannot build is a drop, not a raised answer
            self._drop("unbuildable", exc)
            return
        token = access_token(self._settings)
        headers = {"content-type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10):
                return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # Never raises into Runtime.ask: a Verity outage cannot stop mnemiq answering. But M18
            # is the cautionary tale -- a fail-soft that silently produced nothing looked exactly
            # like "no certified records", so the drop is counted and named.
            #
            # `HTTPError` subclasses `URLError` and carries `.code`, which is the whole difference
            # between "you are wrong" and "they are down". A 4xx is permanent and actionable; a 5xx
            # is the receiver being broken, which is an outage that heals without anyone acting, so
            # it groups with a refused socket rather than with a rejection.
            code = getattr(exc, "code", None)
            cause = "rejected" if isinstance(code, int) and 400 <= code < 500 else "unavailable"
            self._drop(cause, exc)


# mnemiq's `Stage` and Verity's `TraceEventKindV1` are two vocabularies for the same idea, and they
# do not line up. Verity's `kind` is a CLOSED enum, so this is a real mapping rather than a rename --
# and it is LOSSY at exactly one point worth naming: mnemiq's VERIFY is a distinct phase (the
# verifier is a product claim, not an implementation detail) and Verity has no variant for it, so it
# arrives as a model call. The true stage always rides in `payload.stage`, so nothing is lost even
# where `kind` approximates; a `Verify` variant on Verity's side would let `kind` stop approximating.
_STAGE_TO_KIND = {
    "retrieve": "semantic_lookup",
    "plan": "sql_compile",
    "candidate": "model_call",
    "execute": "sql_execute",
    "verify": "model_call",  # lossy, see above
    "synthesize": "answer",
}


def _decision(trace, answer):
    """What the access decision narrowed, from whichever object still has it.

    The trace is built AFTER execution -- it needs timing and result shape -- so a verifier
    deferral or a synthesis failure returns an answer with no trace, and reading the trace alone
    recorded "governance was never evaluated" for queries that were governed and had already run.
    The answer carries the same fact from the moment the decider produces it, so it is the more
    complete source; the trace is preferred only because a successful answer has both and they
    agree by construction.
    """
    for holder in (trace, answer):
        found = getattr(holder, "narrowed", None)
        if found is not None:
            return found
    return None


def _policy_decisions(trace, answer):
    """The access effects, shaped to `trace_schema::PolicyDecisionV1`.

    A function rather than a comprehension inline in the record so it can be tested by CALLING it.
    The first version was asserted by regex-matching the sink's source and eval-ing the expression,
    which broke the moment the expression changed and tested the text rather than the value.
    """
    return [
        {"policy_id": "mnemiq.access",
         "effect": ("row_filter+column_mask" if (n.rows and n.columns)
                    else "row_filter" if n.rows else "column_mask"),
         "metadata": {"object": n.object}}
        for n in (_decision(trace, answer) or [])
    ]


def _to_trace_event(raw: dict, index: int) -> dict:
    stage = str(raw.get("stage") or "")
    kind = _STAGE_TO_KIND.get(stage, "tool_call")
    if raw.get("ok") is False:
        kind = "error"
    return {
        "event_id": f"{index}-{stage or 'stage'}",
        "kind": kind,
        "occurred_at": str(raw.get("at") or _now()),
        "duration_ms": int(raw["ms"]) if isinstance(raw.get("ms"), (int, float)) else None,
        # The stage mnemiq actually ran, kept whatever `kind` had to approximate to.
        "payload": {"stage": stage, "ok": raw.get("ok")},
    }


def _reason_code(answer: Any) -> str | None:
    reason = getattr(answer, "reason_code", None)
    if reason is None:
        return None
    return getattr(reason, "value", None) or str(reason)


def _version() -> str:
    try:
        from mnemiq import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
