"""On Oracle, nothing of a person's request reaches Superset until Superset's session for them is shown to run their
statements as the schema the snapshot was read from.

Superset mode on Oracle answers from one schema (MNEMIQ_SUPERSET_SCHEMA), and the snapshot it serves was read from that
schema on one Oracle service, as the snapshot's `superset:oracle` job records them. Every name mnemiq writes and every
check it makes of a statement is made for that schema, while Oracle reads a statement's unqualified names in the
session's current schema, and Superset opens each person's session as its database connection says: with one login for
everyone, or, where the database impersonates its users, as the person. So mnemiq sends a person nothing until
Superset's session for them is shown to log in as that schema's owner, keep that schema as its current schema, and
reach the service the snapshot was read from.

`Runtime.bound` asks the extension's `session_binding`, as the person, through the adapter their request runs on
(`extension.session_binding`), once per request: the tool opens the session Superset opens for the person's statements
and answers only whether it is bound, never what it read. The service and the schema are asked as the job records them,
as Oracle stores them, so no name is mapped between spellings. The tool finds the database through Superset's own
filter of the databases the caller may see and refuses one whose backend is not Oracle, so no separate description of
the database is asked.

Only "not bound" refuses the person as not bound (`ConnectionNotBound`, 403). What else ends the request says what it
is: the tool could not read the session (`BindingUnread`, Superset not answering, 503: try again); the database is not
Oracle (a deployment misconfigured, 503, naming the setting); Superset does not show the person the database
(Superset's refusal, 403); an answer mnemiq cannot read (the tool's error, 503); an extension without the tool (an older
extension, 503). In every case the person is held: nothing else of the request reaches Superset.
"""

from __future__ import annotations

from mnemiq.superset.client import SupersetRefusal, SupersetUnavailable

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



# What a person is told when Superset could not say whether their session is bound: it may pass, so try again.
BINDING_UNREAD = ("Superset could not say whether its connection to Oracle runs your statements as the schema this "
                  "deployment answers from, so nothing was answered; try again, and if it persists, an operator can see "
                  "why in mnemiq's log.")
# What a person is told when the database Superset mode answers from is not an Oracle database in Superset.
NOT_ORACLE = ("Superset is not set up as mnemiq needs: the database MNEMIQ_SUPERSET_DATABASE_ID names is not an Oracle "
              "database in Superset, and this deployment answers from Oracle; an operator sets MNEMIQ_SUPERSET_DATABASE_ID "
              "to the id Superset gives its Oracle database.")
# What a person is told when the extension's answer cannot be read.
UNREADABLE = ("Superset's mnemiq extension gave an answer this mnemiq cannot read about the connection Superset runs your "
              "statements on, so nothing was answered; an operator checks that the extension and mnemiq are the same "
              "release.")


class BindingUnread(SupersetUnavailable):
    """Superset could not say whether the person's session is bound: the tool could not read it, or failed for its own
    reason. Superset not answering, read as such: the request is refused, and the next one asks again. `detail` goes to
    the log only."""

    def __init__(self, detail: str) -> None:
        super().__init__(BINDING_UNREAD, detail=detail)
