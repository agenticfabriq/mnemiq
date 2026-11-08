"""Who is asking: verifying a caller's token and turning it into an `IdentityContext`.

Authentication only. What that identity may see is still decided by the access policy in `mnemiq.authz`:
the token says who, and the policy says what.
"""
