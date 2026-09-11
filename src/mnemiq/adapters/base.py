from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pyarrow as pa


class SourceAdapter(Protocol):
    # The SQL dialect this adapter actually executes. The engine transpiles its plan
    # (written in duckdb) to this before running -- so a SQLite source receives SQLite,
    # not un-transpiled DuckDB (which fails on YEAR/LISTAGG/etc.).
    dialect: str

    def introspect(self) -> list[str]: ...

    def list_columns(self) -> list[tuple[str, str, str]]: ...

    def foreign_keys(self) -> list[tuple[str, str, str, str, str]]: ...

    # (view_name, definition_sql, definition_dialect). The dialect is the SOURCE's, not this
    # adapter's execution dialect -- a DuckDB adapter attached to Postgres fetches Postgres SQL.
    def view_definitions(self) -> list[tuple[str, str, str]]: ...

    # Names this source defines ITSELF, lowercase, builtins excluded. Optional: an adapter that
    # cannot answer omits it, and the caller records "never asked" rather than an empty answer.
    # A failure RAISES, for the reason `view_definitions` does -- swallowing it turns "could not
    # look" into "there are none", which is the distinction FunctionInventory exists to keep.
    def user_functions(self) -> list[str]: ...

    # The other half of the same catalogue: names this source considers its OWN builtins.
    # Declared beside `user_functions` because the two are one question -- their INTERSECTION
    # is what the decider needs, since a call can be reached under a name the query never
    # spells only if that name is a builtin. Declaring it cannot ENFORCE the pairing, and the
    # objection recorded under the flag below applies here too: no adapter but DuckDB's has
    # either method. It is here to be read, not to bind, and what it says is what omitting it
    # costs -- every source holding one macro refuses every query against it, under a code
    # that is not repairable. Reached through `hasattr` at the call site. A failure RAISES.
    def builtin_functions(self) -> list[str]: ...

    # Whether `user_functions()` also answers for a VIEW BODY on this source. Read through
    # `getattr(adapter, ..., False)`, so an adapter that says nothing is taken not to cover them
    # -- forgetting yields the conservative answer. Deliberately NOT declared as a member here:
    # most adapters omit it, and a required member they do not have would make none of them
    # structurally match this Protocol.

    def execute(self, sql: str) -> list[tuple]: ...

    def execute_arrow(self, sql: str, timeout_s: float | None = None) -> pa.Table: ...
