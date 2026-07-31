"""M3 -- the deployment write switch must be a governed refusal, not a database error.

`settings.write_enabled` had exactly two consumers, both `read_only=not write_enabled` on the
adapter. No decider ever saw it. `Runtime.write`'s docstring promised "two locks" and the second was
the read-only attach *raising*, caught and returned as "the source rejected the write: {exc}" -- the
comment beside it calls that "the backstop", with nothing in front of it.

The outcome was fail-closed, so this was never a leak. The defect is that the control WAS the
database, in the one place where the switch is deployment-level rather than grant-level -- the exact
principle the project defines itself against.
"""

from mnemiq.authz.grants import GrantSet
from mnemiq.sql.decide_write import decide_write
from mnemiq.sql.verdict import ApprovedWrite, Refusal, RefusalCode

_VISIBLE = {"claim": {"id", "amount"}}
_SQL = "UPDATE claim SET amount = 0 WHERE id = 1"


def _grants() -> GrantSet:
    return GrantSet(frozenset({"claim"}), writable=frozenset({"claim"}))


class _ExplodingAdapter:
    """Proves the decider never reaches the source. Any call here is the bug."""

    def execute(self, sql):
        raise AssertionError(f"the decider consulted the source after refusing: {sql!r}")


def test_a_deployment_with_writes_disabled_refuses_before_touching_the_source():
    verdict = decide_write(
        _SQL, _VISIBLE, _grants(), adapter=_ExplodingAdapter(), writes_enabled=False
    )

    assert isinstance(verdict, Refusal)
    assert verdict.code == RefusalCode.WRITES_DISABLED


def test_the_refusal_does_not_claim_the_identity_lacks_a_grant():
    # D43's distinction: saying "governance said no" when governance was never consulted.
    verdict = decide_write(
        _SQL, _VISIBLE, _grants(), adapter=_ExplodingAdapter(), writes_enabled=False
    )

    assert verdict.code != RefusalCode.UNAUTHORIZED_WRITE
    message = verdict.message.lower()
    assert "deployment" in message or "disabled" in message, verdict.message
    assert "may not write" not in message, "that wording belongs to a grant denial"


def test_omitting_the_switch_fails_closed():
    # A security parameter whose default is permissive means forgetting it fails OPEN.
    verdict = decide_write(_SQL, _VISIBLE, _grants(), adapter=_ExplodingAdapter())

    assert isinstance(verdict, Refusal)
    assert verdict.code == RefusalCode.WRITES_DISABLED


def test_writes_enabled_is_unchanged():
    class _OkAdapter:
        def execute(self, sql):
            return []

    verdict = decide_write(
        _SQL, _VISIBLE, _grants(), adapter=_OkAdapter(), writes_enabled=True
    )

    assert isinstance(verdict, ApprovedWrite), getattr(verdict, "message", verdict)


def test_runtime_threads_the_setting_into_the_decider():
    # The switch reaching the decider is the whole finding; a correct decider nobody informs is
    # the same shape one level up.
    import mnemiq.runtime as rt_mod
    from mnemiq.contract import IdentityContext, Snapshot

    seen = {}

    def fake_decide_write(sql, visible, grants, **kwargs):
        seen["writes_enabled"] = kwargs.get("writes_enabled")
        return Refusal(code=RefusalCode.WRITES_DISABLED, message="stub", subject="claim")

    class _Settings:
        write_enabled = False
        source_id = "acme"

    class _Authz:
        def grants_for(self, identity):
            return _grants()

    runtime = rt_mod.Runtime.__new__(rt_mod.Runtime)
    runtime.snapshot = Snapshot(version="v", source_id="s", created_at="t", columns=[])
    runtime.authz = _Authz()
    runtime.adapter = None
    runtime.settings = _Settings()
    runtime.sink = None

    original = rt_mod.decide_write
    rt_mod.decide_write = fake_decide_write
    try:
        runtime.write(_SQL, IdentityContext(tenant_id="t", principal_id="u", roles=["analyst"]))
    finally:
        rt_mod.decide_write = original

    assert seen["writes_enabled"] is False, "the deployment switch must reach the decider"
