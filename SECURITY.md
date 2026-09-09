# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** Use GitHub's private
advisory form:

**[Report a vulnerability](https://github.com/agenticfabriq/mnemiq/security/advisories/new)**

That opens a channel visible only to you and the maintainers, and it is the route we
watch. If you cannot use it, open a public issue saying only that you have a security
report and giving no details — we will open a private channel and come to you.

We aim to acknowledge a report within three working days and to tell you what we intend
to do about it within ten. If a fix is warranted we will credit you in the release notes
unless you ask us not to.

## What is in scope

mnemiq sits between a question and a database that has an access policy, so the bugs
that matter most here are the ones that let a caller reach data the policy denies. In
rough order of how seriously we take them:

- **An identity reading what its grants exclude** — through retrieval, a generated
  query, a view, a function, a federated source, or the write path's read side.
- **Disclosure through a side channel** — an error message, a refusal reason, a trace or
  audit record, a value-grounding refusal, or a lineage report that names an object the
  caller was not entitled to know about. A refusal that reveals what it refused is a
  disclosure.
- **A statement executing that the decider should have refused** — DDL or DML reaching a
  read-only deployment, a row filter or column mask not applied, a repair loop producing
  an approved statement the original would have been refused for.
- **Credential or configuration exposure** — a DSN, key or wallet path reaching a
  response, a log, or a published artifact.

Both halves of an access control count. The engine scoping what a model is *shown* and
the decider re-checking what a query *names* are separate mechanisms, and a bypass of
either is a finding even when the other happens to catch it.

## What is out of scope

- Findings that require the deployer to have configured something the documentation
  tells them not to — a superuser or `BYPASSRLS` connection, `MNEMIQ_WRITE_ENABLED` on a
  source with no policy, or an unset `MNEMIQ_AUTHZ_PATH`. These are deployment
  preconditions, documented as such, and we would rather hear about places the docs fail
  to say so.
- The model generating incorrect SQL. That is an accuracy problem and belongs in a
  public issue; the engine is designed to refuse rather than trust generated SQL, so a
  wrong query is only a security finding if it *executed* when it should not have.
- Denial of service through expensive queries against your own database.
- Anything in `demo/`, `evals/` or the benchmark harnesses, which are not deployment
  surface.

## Supported versions

mnemiq is pre-1.0 and moves quickly. Fixes land on `main`; there is no backport branch
yet. If you depend on a released version and need that to change, say so in your report.
