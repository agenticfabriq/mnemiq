"""On Oracle, nothing of a person's request reaches Superset until Superset's session for them is shown to run their
statements as the schema the snapshot was read from.

Superset mode on Oracle answers from one schema (MNEMIQ_SUPERSET_SCHEMA), and the snapshot it serves was read from that
schema on one Oracle service, as the snapshot's `superset:oracle` job records them. Every name mnemiq writes and every
check it makes of a statement is made for that schema, while Oracle reads a statement's unqualified names in the
session's current schema, and Superset opens each person's session as its database connection says: with one login for
everyone, or, where the database impersonates its users, as the person. So mnemiq sends a person nothing until
Superset's session for them is shown to log in as that schema's owner, keep that schema as its current schema, and
reach the service the snapshot was read from.

`binding_refusal` asks the extension's `session_binding`, as the person, through the adapter their request runs on
(`extension.session_binding`): the tool opens the session Superset opens for the person's statements and answers only
whether it is bound, never what it read. The service and the schema are asked as the job records them, as Oracle
stores them, so no name is mapped between spellings. The tool finds the database through Superset's own filter of the
databases the caller may see and refuses one whose backend is not Oracle, so no separate description of the database
is asked.

`Runtime.bound` asks it once per request, and keeps nothing between requests.
"""

from __future__ import annotations

from mnemiq.superset.client import SupersetRefusal
from mnemiq.superset.extension import session_binding

# What a person is told when their connection was not shown to be the snapshot's. Why goes to the log only: it names
# the deployment's service and schema.
NOT_BOUND = ("Superset's connection to Oracle was not shown to run your statements as the schema this deployment answers "
             "from, so nothing is answered for you until an operator has Superset log in to Oracle as that schema's "
             "owner; mnemiq's log records why.")


class ConnectionNotBound(SupersetRefusal):
    """A request refused because Superset's connection for the person asking was not shown to run their statements as
    the snapshot's schema, on its service. The message, also `message`, is mnemiq's sentence for the person; `detail`
    says why, and goes to the log only."""

    def __init__(self, detail: str) -> None:
        super().__init__(NOT_BOUND, detail=detail)
        self.message = NOT_BOUND


def binding_refusal(adapter, *, database_id: int, service: str, schema: str) -> str | None:
    """None when the extension's `session_binding`, asked as the person whose token `adapter` carries, answers that the
    session Superset opens for them on database `database_id` logs in as `schema`, keeps it as its current schema and
    reached `service` (each as Oracle stores it); else why not, in words for the log.

    Any answer but bound or not (an extension older than the tool, a database not Oracle or not shown to the person, a
    session the tool could not read, an answer mnemiq cannot read) is not bound. Superset not answering
    (`SupersetUnavailable`) and a refusal of the person raise as themselves."""
    found = session_binding(adapter.client, database_id=database_id, service=service, schema=schema)
    if found is True:
        return None
    if found is False:
        return (f"Superset's session for this person on database {database_id} does not log in as {schema!r}, keep "
                f"it as its current schema and reach service {service!r}, as the snapshot was read")
    return (f"Superset's mnemiq extension did not say whether its session for this person on database {database_id} "
            "is the snapshot's (its answer is logged): an extension older than 0.3.0, a database Superset does not "
            "show the person or that is not Oracle, or a session it could not read")
