"""What every sweep test stands on: the scripted model, a tiny real deployment, and a config file around them.

`tiny_deployment` is a real one: a SQLite source of three tables, a store enriched and indexed by the engine's own
`enrich` and `build` commands, an access policy, an env file and graded cases. Only the model is scripted.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sqlite3
import tempfile
import threading
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import duckdb
import pytest

import mnemiq
from mnemiq.config import DEFAULT_RETRIEVAL_K, Settings
from mnemiq.contract import EvaluationCase
from mnemiq.profiles import PROFILE_SETTINGS
from mnemiq.sweep.run import engine_build
from mnemiq.sweep.starters import starter_text as _starter_text
from tests.starter_fixtures import ROOT_AND_CHILD, STARTER, starter_folder, write
from tests.starter_fixtures import bind as bind  # for the sweep tests, which import their helpers from here
from tests.starter_fixtures import edit as edit

from .stub_model import _WRITER, STUB_WINDOW, StubModel, _messages, _proposal

ROLE = "sweep-test"
KEY_ENV = "SWEEP_TEST_KEY"
STUB_KEY = "stub-key"
# Nothing listens on the discard port: a config that names it loads, and anything that calls it fails at once.
UNREACHABLE = "http://127.0.0.1:9/v1"

_AUTHORS = [(1, "Ada Lindqvist", "Sweden"), (2, "Tomas Okafor", "Canada"), (3, "Mei Tanaka", "Japan"),
            (4, "Rosa Ferreira", "Canada")]
_BOOKS = [(1, 1, "The Quiet Harbour", 1938, "fiction"), (2, 1, "Salt and Iron", 1952, "history"),
          (3, 2, "Northern Lights Field Guide", 1971, "science"), (4, 2, "The Long Portage", 1946, "fiction"),
          (5, 3, "Tides of Kobe", 1989, "history"), (6, 3, "Paper Lanterns", 2004, "fiction"),
          (7, 4, "Counting Stars", 1999, "science"), (8, 4, "A Map of Small Rivers", 2015, "fiction")]
_LOANS = [(1, 1, "2026-01-04", 1), (2, 1, "2026-02-11", 1), (3, 2, "2026-01-19", 1), (4, 3, "2026-03-02", 0),
          (5, 4, "2026-03-15", 1), (6, 5, "2026-04-01", 0), (7, 6, "2026-04-20", 1), (8, 6, "2026-05-05", 1),
          (9, 6, "2026-06-12", 0), (10, 7, "2026-06-30", 1), (11, 8, "2026-07-08", 1), (12, 8, "2026-08-21", 0),
          (13, 2, "2026-09-03", 1), (14, 5, "2026-09-27", 1), (15, 1, "2026-10-14", 0)]

# Synthetic questions over the tables above. `wrong_sql` runs and returns other rows; `variants` return the gold
# rows, and `wrong_variants` the wrong rows, under another spelling.
CASES: tuple[dict, ...] = (
    {"id": "book-count", "question": "How many books does the library hold?",
     "gold_sql": "SELECT COUNT(*) AS books FROM books",
     "variants": ["SELECT COUNT(book_id) AS books FROM books"],
     "wrong_sql": "SELECT COUNT(*) AS books FROM authors",
     "wrong_variants": ["SELECT COUNT(author_id) AS books FROM authors"]},
    {"id": "canadian-authors", "question": "How many authors come from Canada?",
     "gold_sql": "SELECT COUNT(*) AS authors FROM authors WHERE country = 'Canada'",
     "variants": ["SELECT COUNT(author_id) AS authors FROM authors WHERE country = 'Canada'"],
     "wrong_sql": "SELECT COUNT(*) AS authors FROM authors",
     "wrong_variants": ["SELECT COUNT(author_id) AS authors FROM authors"]},
    {"id": "before-1950", "question": "Which books were published before 1950?",
     "gold_sql": "SELECT title FROM books WHERE published_year < 1950",
     "variants": ["SELECT title FROM books WHERE published_year <= 1949"],
     "wrong_sql": "SELECT title FROM books WHERE published_year > 1950",
     "wrong_variants": ["SELECT title FROM books WHERE published_year >= 1951"]},
    {"id": "loans-by-genre", "question": "How many loans has each genre had?",
     "gold_sql": "SELECT b.genre, COUNT(l.loan_id) AS loans FROM loans l JOIN books b ON b.book_id = l.book_id "
                 "GROUP BY b.genre",
     "variants": ["SELECT books.genre, COUNT(*) AS loans FROM loans JOIN books ON books.book_id = loans.book_id "
                  "GROUP BY books.genre"],
     "wrong_sql": "SELECT genre, COUNT(*) AS loans FROM books GROUP BY genre",
     "wrong_variants": ["SELECT genre, COUNT(book_id) AS loans FROM books GROUP BY genre"]},
    {"id": "open-loans", "question": "How many loans have not been returned?",
     "gold_sql": "SELECT COUNT(*) AS open_loans FROM loans WHERE returned = 0",
     "variants": ["SELECT COUNT(loan_id) AS open_loans FROM loans WHERE returned = 0"],
     "wrong_sql": "SELECT COUNT(*) AS open_loans FROM loans",
     "wrong_variants": ["SELECT COUNT(loan_id) AS open_loans FROM loans"]},
    {"id": "newest-book", "question": "What is the title of the most recently published book?",
     "gold_sql": "SELECT title FROM books ORDER BY published_year DESC LIMIT 1",
     "variants": ["SELECT title FROM books WHERE published_year = (SELECT MAX(published_year) FROM books)"],
     "wrong_sql": "SELECT title FROM books ORDER BY published_year ASC LIMIT 1",
     "wrong_variants": ["SELECT title FROM books WHERE published_year = (SELECT MIN(published_year) FROM books)"]},
    {"id": "book-prices", "question": "What does each book cost?", "gold_sql": None, "answerable": False,
     "wrong_sql": "SELECT title, published_year AS cost FROM books",
     "wrong_variants": ["SELECT books.title, books.published_year AS cost FROM books"]},
    {"id": "top-members", "question": "Which members borrowed the most books?", "gold_sql": None, "answerable": False,
     "wrong_sql": "SELECT book_id, COUNT(*) AS loans FROM loans GROUP BY book_id",
     "wrong_variants": ["SELECT book_id, COUNT(loan_id) AS loans FROM loans GROUP BY book_id"]},
)
_CASE_FIELDS = ("id", "question", "gold_sql", "answerable", "db_id")


@dataclass(frozen=True)
class TinyDeployment:
    root: Path
    source: Path  # the SQLite file
    sources: Path  # the source manifest naming it
    store: Path
    authz: Path
    env: Path
    questions: Path
    cases: tuple[dict, ...]


class _GivenOnly(Settings):
    """Settings from the values given and nothing else: no `MNEMIQ_*` variable of the process running the tests."""

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                   file_secret_settings):
        return (init_settings,)


def _graded(case: dict) -> dict:
    """A case as the question file holds it: the `EvaluationCase` fields, without what only the stub reads."""
    return {name: case[name] for name in _CASE_FIELDS if name in case} | {"answerable": case.get("answerable", True)}


def _write_source(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE authors (author_id INTEGER PRIMARY KEY, name TEXT, country TEXT)")
    con.execute("CREATE TABLE books (book_id INTEGER PRIMARY KEY, author_id INTEGER REFERENCES authors(author_id), "
                "title TEXT, published_year INTEGER, genre TEXT)")
    con.execute("CREATE TABLE loans (loan_id INTEGER PRIMARY KEY, book_id INTEGER REFERENCES books(book_id), "
                "loaned_on TEXT, returned INTEGER)")
    con.executemany("INSERT INTO authors VALUES (?, ?, ?)", _AUTHORS)
    con.executemany("INSERT INTO books VALUES (?, ?, ?, ?, ?)", _BOOKS)
    con.executemany("INSERT INTO loans VALUES (?, ?, ?, ?)", _LOANS)
    con.commit()
    con.close()


def _env_values(d: TinyDeployment, url: str | None) -> dict[str, str]:
    values = {"sources_path": str(d.sources), "store_path": str(d.store), "authz_path": str(d.authz)}
    if url:
        values |= {"llm_base_url": url, "llm_api_key": STUB_KEY, "llm_model": "stub-model",
                   "embed_base_url": url, "embed_api_key": STUB_KEY, "embed_model": "stub-embed"}
    else:
        # An embedder that nothing answers: the engine's profile loader requires one, and nothing here calls it.
        values |= {"embed_base_url": UNREACHABLE, "embed_api_key": STUB_KEY}
    return values


def _write_env(d: TinyDeployment, url: str | None, extra: dict[str, str] | None = None) -> None:
    lines = [f"MNEMIQ_{name.upper()}='{value}'\n" for name, value in _env_values(d, url).items()]
    lines += [f"{name}='{value}'\n" for name, value in (extra or {}).items()]
    d.env.write_text("".join(lines))


def tiny_files(tmp_path: Path) -> TinyDeployment:
    """The deployment's files with no store and no model endpoint: what `load_config` reads, with no server running."""
    root = tmp_path / "deployment"
    root.mkdir(exist_ok=True)
    d = TinyDeployment(root=root, source=root / "library.sqlite", sources=root / "sources.json",
                       store=root / "store.duckdb", authz=root / "authz.json", env=root / "deployment.env",
                       questions=root / "questions.json", cases=CASES)
    if not d.source.exists():
        _write_source(d.source)
    if not d.sources.exists():
        d.sources.write_text(json.dumps([{"id": "library", "kind": "sqlite", "target": str(d.source),
                                          "catalog": "src", "schema": "main"}]))
    if not d.authz.exists():
        d.authz.write_text(json.dumps({"roles": {ROLE: ["authors", "books", "loans"]}}))
    if not d.questions.exists():
        d.questions.write_text(json.dumps([_graded(case) for case in CASES], indent=2))
    if not d.env.exists():
        _write_env(d, None)
    return d


def _write_authz(d: TinyDeployment, *, withhold: str | None, row_filter: bool, mask: bool) -> None:
    tables = [table for table in ("authors", "books", "loans") if table != withhold]
    role: list | dict = tables
    if row_filter or mask:
        role = {"read": tables, **({"row_filters": {"loans": "returned = 1"}} if row_filter else {}),
                **({"pii_mask": ["contact"]} if mask else {})}
    d.authz.write_text(json.dumps({"roles": {ROLE: role}}))


def more_cases(count: int) -> tuple[dict, ...]:
    """`count` synthetic questions over the tiny tables, beside `CASES`. Three in four ask how many books were published
    after a year before the first book was, and their wrong answer counts the books published before it, which is none.
    The rest ask what the library paid for a book, which no table holds, and their wrong answer makes one up."""
    written: list[dict] = []
    for i in range(count):
        if i % 4 == 3:
            made_up = f"SELECT title, published_year AS paid FROM books WHERE book_id = {i % len(_BOOKS) + 1}"
            written.append({"id": f"paid-{i}", "question": f"What did the library pay for book number {i}?",
                            "gold_sql": None, "answerable": False, "wrong_sql": made_up})
            continue
        year = 1900 + i
        if year >= min(published for _, _, _, published, _ in _BOOKS):
            raise ValueError(f"{count} synthetic cases reach {year}, where the wrong answer would count some books")
        written.append({"id": f"after-{year}", "question": f"How many books were published after {year}?",
                        "gold_sql": f"SELECT COUNT(*) AS books FROM books WHERE published_year > {year}",
                        "wrong_sql": f"SELECT COUNT(*) AS books FROM books WHERE published_year < {year}"})
    return tuple(written)


def _with_cases(d: TinyDeployment, count: int) -> TinyDeployment:
    """The deployment with a question file of `count` cases: `CASES`, then `more_cases` up to `count`."""
    if count < len(CASES):
        raise ValueError(f"cases= counts the tiny deployment's own {len(CASES)} cases, so it is at least that")
    written = CASES + more_cases(count - len(CASES))
    path = d.root / f"questions-{count}.json"
    if not path.exists():
        path.write_text(json.dumps([_graded(case) for case in written], indent=2))
    return replace(d, questions=path, cases=written)


def tiny_deployment(tmp_path: Path, stub: StubModel, *, env_extra: dict[str, str] | None = None,
                    withhold: str | None = None, row_filter: bool = False, mask: bool = False,
                    cases: int | None = None) -> TinyDeployment:
    """The tiny deployment, its store built once per `tmp_path`: a second call returns what the first built, so a
    rebuilt store never moves a fingerprint. Its env file points at the first stub given, for chat and embeddings, and
    every stub given learns its cases. `env_extra` adds lines to the env file, as `{"MNEMIQ_NAME": "value"}`, and is
    taken only by the call that builds the deployment. `withhold` takes one table out of the sweep's role, `row_filter`
    gives the role a row filter, and `mask` a PII mask; the store depends on none of them. `cases=N` gives it a question
    file of N cases (`CASES`, then `more_cases`), which the store does not depend on either."""
    from mnemiq.cli import _cmd_build, _cmd_enrich

    d = tiny_files(tmp_path)
    if withhold is not None or row_filter or mask:
        _write_authz(d, withhold=withhold, row_filter=row_filter, mask=mask)
    if env_extra and d.store.exists():
        raise ValueError("env_extra is written when the deployment is built: pass it to the first call for this "
                         "tmp_path")
    if not d.store.exists():
        # A stub of its own builds the store, so the one given counts and fails only for what the test asks. The
        # vectors are a function of the text alone, so a question embedded by the given stub matches this index.
        with StubModel() as builder:
            settings = _GivenOnly(**_env_values(d, builder.url), llm_window_check=False)
            if _cmd_enrich(settings) != 0 or _cmd_build(settings) != 0 or builder.errors:
                raise RuntimeError(f"the tiny deployment's store did not build ({builder.errors!r}); see the "
                                   "captured output")
        _write_env(d, stub.url, env_extra)
    if cases is not None:
        d = _with_cases(d, cases)
    stub.learn(d.cases)
    return d


def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (str, Path)):
        return json.dumps(str(value))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items() if v is not None) + "}"
    raise TypeError(f"no TOML for {value!r}")


def dump_toml(doc: dict) -> str:
    """Each top-level key on a line of its own, its tables inline. Enough for a sweep config."""
    return "".join(f"{_toml_key(key)} = {_toml_value(value)}\n" for key, value in doc.items() if value is not None)


def write_config(tmp_path: Path, *, deployment: TinyDeployment | None = None, model: StubModel | None = None,
                 questions: Path | None = None, arms_in_screen: int | None = None, **sections) -> Path:
    """A sweep config over the tiny deployment, written to `tmp_path / "sweep.toml"`.

    One model, `m`, keyed by `SWEEP_TEST_KEY`, at `model`'s URL (or one nothing answers), in instant mode, starting
    from the engine's defaults. With `model` and no `deployment` it builds the tiny deployment on that stub; with
    neither, it writes the deployment's files without a store, which loads without any server. Each other keyword
    is a config section: `sweep` and `dataset` merge into the defaults (`dataset=tiny_dataset(...)` swaps the
    questions), any other replaces its section. `arms_in_screen=N` writes a space whose screen holds N arms beside
    the start.
    """
    if arms_in_screen is not None:
        if "space" in sections:
            raise ValueError("arms_in_screen writes the space: pass one or the other")
        sections["space"] = {"retrieval_k": [DEFAULT_RETRIEVAL_K + 4 * i for i in range(1, arms_in_screen + 1)]}
    if deployment is None:
        deployment = tiny_deployment(tmp_path, model) if model is not None else tiny_files(tmp_path)
    if model is not None:
        model.learn(deployment.cases)
    doc: dict = {
        "sweep": {"name": "tiny", "out": str(tmp_path / "sweep-out")},
        "dataset": {"questions": str(questions or deployment.questions), "env": str(deployment.env), "split_seed": 7,
                    "identity": {"tenant": "local", "principal": "sweep", "roles": [ROLE]}},
        "models": one_model(model),
        "modes": ["instant"],
    }
    for name, value in sections.items():
        if name in ("sweep", "dataset") and isinstance(value, dict):
            merged = {**doc[name], **value}
            if name == "dataset" and "deployments" in value and "env" not in value:
                merged.pop("env")
            doc[name] = merged
        else:
            doc[name] = value
    path = tmp_path / "sweep.toml"
    path.write_text(dump_toml(doc))
    return path


def start_that_differs(tmp_path: Path) -> dict:
    """A `[start]` section naming a profile that sets retrieval_k to 12, away from the engine's default. Check mode
    compares a start with the engine's defaults, so its start must differ from them."""
    path = tmp_path / "start-profiles.toml"
    path.write_text('default = "start"\n[profiles.start]\nretrieval_k = 12\n')
    return {"profile_file": str(path), "profile": "start"}


def start_refused_in_instant(tmp_path: Path) -> dict:
    """A `[start]` section naming a profile that sets min_agreement on instant's one candidate, which the engine refuses
    in instant and accepts in thinking."""
    path = tmp_path / "start-refused-in-instant.toml"
    path.write_text('default = "tuned"\n[profiles.tuned]\n[profiles.tuned.instant]\nmin_agreement = 0.6\n')
    return {"profile_file": str(path)}


# --- a starter, its evidence, and a model's build ---------------------------------------------------------------------

STARTER_NAME = "fixture-2.5-build"
# Model `m` as `write_config` writes it under `[models]`, at an address nothing answers.
ONE_MODEL = {"llm_model": "stub-model", "llm_base_url": UNREACHABLE, "api_key_env": KEY_ENV}
# What a config states of the build the fixture starter was swept on: what no server reports.
CONFIG_BUILD = {"model": "fixture/model-2.5-awq", "gpu": "GeForce RTX 4090", "quantization": "awq"}
# The engine this process runs, which `starter_with_evidence` records as the one its starter was swept on.
SHA12 = engine_build()["source_sha"][:12]
COMMIT = engine_build()["commit"] or "no commit"
# A database key shaped like a machine path, which the evidence schema refuses. Spelled in two parts, as the repo
# guard's own tests spell one, so the source never holds a home path.
PATH_SHAPED_KEY = "/" + "Users/x"


def one_model(stub: StubModel | None = None, **fields) -> dict:
    """The `[models]` section `write_config` writes: model `m`, at `stub`'s URL when given, with `fields` changed.
    `build=` is its build table, `[models.m.build]`."""
    entry = ONE_MODEL | ({"llm_base_url": stub.url} if stub is not None else {})
    return {"m": entry | fields}


def _counts(right: int, wrong: int, made_up: int, declined: int) -> dict:
    """One run of an arm in one mode, as starter evidence records it: 26 questions, every one measured."""
    return {"valid": True, "cases": 26, "right": right, "strict": right, "wrong": wrong, "made_up": made_up,
            "declined": declined, "failed": 0, "errors": 0, "ungraded": 0, "unjudged": 0, "median_ms": 1000.0,
            "median_rows_ms": 800.0, "prompt_tokens": 900.0}


def _evidence(name: str) -> dict:
    """Evidence the schema accepts for the starter `name`: checked in instant mode against the engine's defaults, on
    the build `CONFIG_BUILD` states, served as a stub serves by default (vLLM 0.9.0.1, a window of `STUB_WINDOW`), on
    the engine running now."""
    return {
        "starter": name,
        "published": "2026-11-07",
        "dataset": {"name": "fixture-questions", "questions": 26, "answerable": 17},
        "made_up_measured": True,
        "model_build": {"model": "fixture/model-2.5-awq", "quantization": "awq", "server": "vllm",
                        "server_version": "0.9.0.1", "gpu": "GeForce RTX 4090", "context_window": STUB_WINDOW},
        "engine_build": {"version": importlib.metadata.version("mnemiq"), **engine_build()},
        "safe_by": "check",
        "rule": {"strategy": "check", "alpha": 0.05, "max_more_wrong": 0, "max_more_made_up": 0, "max_errors": 0,
                 "repeats": 2},
        "modes_measured": ["instant"],
        "unmeasured_modes": {"thinking": ["retrieval_k", "writer_temperature"],
                             "deep": ["retrieval_k", "writer_temperature"]},
        "held_unset": ["guard_fanout", "llm_seed"],
        "check": {"instant": [{"run": run, "start": _counts(12, 2, 0, 12), "defaults": _counts(11, 3, 1, 11)}
                              for run in (1, 2)]},
        "search": None,
        "versions": {"": "0123456789ab"},
        "measured_versions": None,
    }


# Each worker a sweep starts is a spawned process that imports the engine afresh, so the starters folder a test patched
# in its own process is lost there. Imported as each worker starts, from `PYTHONPATH`, this patches it there too.
_STARTERS_IN_WORKERS = """
import os

if os.environ.get("SWEEP_TEST_STARTERS"):
    from pathlib import Path

    import mnemiq.profiles

    _folder = Path(os.environ["SWEEP_TEST_STARTERS"])
    mnemiq.profiles.starters_folder = lambda: _folder
"""


def starter_with_evidence(tmp_path: Path, monkeypatch, *, name: str = STARTER_NAME, settings: str = STARTER,
                          **evidence) -> Path:
    """A starters folder of the test's own (`starter_folder`), holding the starter `name` with `settings` and its
    evidence, which the schema accepts. Each keyword in `evidence` replaces that field of it. Returns the folder.

    The sweep's workers read the same folder, through a `sitecustomize` on `PYTHONPATH` that each spawned process
    imports as it starts."""
    folder = starter_folder(tmp_path, monkeypatch, {name: settings})
    bound = {"starter_sha256": hashlib.sha256((folder / f"{name}.toml").read_bytes()).hexdigest()}
    write(folder / f"{name}.evidence.json", json.dumps(_evidence(name) | bound | evidence, indent=2))
    hook = tmp_path / "starters-in-workers"
    hook.mkdir(exist_ok=True)
    write(hook / "sitecustomize.py", _STARTERS_IN_WORKERS)
    paths = [str(hook), str(Path(mnemiq.__file__).parents[1]), *filter(None, [os.environ.get("PYTHONPATH")])]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    monkeypatch.setenv("SWEEP_TEST_STARTERS", str(folder))
    return folder


def valid_evidence(tmp_path: Path, monkeypatch) -> dict:
    """The evidence `starter_with_evidence` writes, as a dict."""
    folder = starter_with_evidence(tmp_path, monkeypatch)
    return json.loads((folder / f"{STARTER_NAME}.evidence.json").read_text())


def start_via(tmp_path: Path, via: str) -> dict:
    """A `[start]` section that reaches the fixture starter: by its name, or through a profile file whose default (the
    root of `ROOT_AND_CHILD`) names it."""
    if via == "starter":
        return {"starter": STARTER_NAME}
    if via == "profile_file":
        return {"profile_file": str(write(tmp_path / "start-from-starter.toml", ROOT_AND_CHILD))}
    raise ValueError(f"no such way to start: {via}")


def altered(evidence: dict, path: tuple) -> dict:
    """A copy of `evidence` with one change its schema must refuse, at `path`: the keys down to a table, then what is
    planted in it. `"extra"` is a key the schema does not name, `"fast"` a mode that is not one, and `PATH_SHAPED_KEY` a
    database key shaped like a path; a table on the way that is null is made. A `None` in `path` sets the field named
    by the keys before it to the value after it: `("model_build", "model", None, "/srv/qwen")`, and
    `("published", None)`, with no value, writes the date in another form. `("rule", "max_more_wrong")` sets that bound
    to 1."""
    changed = copy.deepcopy(evidence)
    if None in path:
        cut = path.index(None)
        *where, field = path[:cut]
        holder = changed
        for key in where:
            holder = holder[key]
        holder[field] = path[cut + 1] if len(path) > cut + 1 else "07/11/2026"
        return changed
    if path == ("rule", "max_more_wrong"):
        changed["rule"]["max_more_wrong"] = 1
        return changed
    *where, planted = path
    holder = changed
    for key in where:
        if isinstance(holder, dict) and holder.get(key) is None:
            holder[key] = {}
        holder = holder[key]
    if planted == "extra":
        holder["extra"] = 1
    elif planted == "fast":
        holder["fast"] = copy.deepcopy(next(iter(holder.values())))
    elif planted == PATH_SHAPED_KEY:
        holder[PATH_SHAPED_KEY] = "0123456789ab"
    else:
        raise ValueError(f"no such change: {path}")
    return changed


def broken_questions(tmp_path: Path, bad: str) -> Path:
    """The tiny cases with one fault that makes them ungradable, written beside the deployment."""
    cases = [_graded(case) for case in CASES]
    if bad == "duplicate_id":
        cases[1]["id"] = cases[0]["id"]
    elif bad == "answerable_without_gold":
        cases[0]["gold_sql"] = None
    elif bad == "unknown_db_id":
        cases[0]["db_id"] = "nowhere"
    else:
        raise ValueError(f"no such fault: {bad}")
    path = tmp_path / f"questions-{bad}.json"
    path.write_text(json.dumps(cases))
    return path


def _rewrite_config(path: Path, change) -> None:
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    change(doc)
    path.write_text(dump_toml(doc))


def alter(cfg, change: str) -> None:
    """Change one thing a loaded config depends on, on disk."""
    from mnemiq.store.bootstrap import init_store
    from mnemiq.sweep.config import deployment_settings

    settings = deployment_settings(cfg, None)
    if change == "store":
        con = init_store(settings.store_path)
        con.execute("CREATE TABLE sweep_test_marker (x INTEGER)")
        con.close()
    elif change == "authz":
        policy = json.loads(Path(settings.authz_path).read_text())
        policy["roles"]["another-role"] = ["books"]
        Path(settings.authz_path).write_text(json.dumps(policy))
    elif change == "env_setting":
        with open(cfg.dataset.env, "a") as fh:
            fh.write("MNEMIQ_MAX_QUESTION_CHARS=2000\n")
    elif change == "start_profile":
        if cfg.start.profile_file:
            with open(cfg.start.profile_file, "a") as fh:
                fh.write("writer_rules = \"none\"\n")
        else:
            profile = cfg.path.parent / "start-profiles.toml"
            profile.write_text('default = "start"\n[profiles.start]\nretrieval_k = 12\n')
            _rewrite_config(cfg.path, lambda doc: doc.update(start={"profile_file": str(profile), "profile": "start"}))
    elif change == "questions":
        path = Path(cfg.dataset.questions)
        cases = json.loads(path.read_text())
        cases[0]["question"] += " Count every copy."
        path.write_text(json.dumps(cases))
    elif change == "budget":
        _rewrite_config(cfg.path, lambda doc: doc.setdefault("search", {}).update(budget_hours=12.5,
                                                                                    budget_dollars=40.0))
    elif change == "seconds_per_question":
        _rewrite_config(cfg.path, lambda doc: doc.setdefault("search", {}).update(seconds_per_question=45.0))
    elif change == "control":
        answers = cfg.path.parent / "other-answers.json"
        answers.write_text(json.dumps([{"id": CASES[0]["id"], "sql": CASES[0]["gold_sql"]}]))
        _rewrite_config(cfg.path, lambda doc: doc.setdefault("control", []).append(
            {"name": "another-system", "answers": str(answers)}))
    else:
        raise ValueError(f"no such change: {change}")


def rewrite(cfg, **sections) -> Path:
    """The config's file with each section's keys changed to those given, as a person edits it before resuming; its
    path."""
    def change(doc: dict) -> None:
        for name, value in sections.items():
            if isinstance(value, dict):
                doc.setdefault(name, {}).update(value)
            else:
                doc[name] = value

    _rewrite_config(cfg.path, change)
    return cfg.path


def out(cfg) -> Path:
    """The sweep's out folder."""
    return Path(cfg.sweep.out)


def sweep_json(cfg) -> dict:
    """The sweep's own record, `sweep.json`."""
    return json.loads((out(cfg) / "sweep.json").read_text())


def started_sweep(tmp_path: Path, stub: StubModel, **sections):
    """A sweep over the tiny deployment on `stub` whose first round has run, loaded. Each keyword is a section."""
    from mnemiq.sweep.config import load_config
    from mnemiq.sweep.run import run

    cfg = load_config(write_config(tmp_path, model=stub, **sections))
    run(cfg, until="screen")
    return cfg


def write_rows(cfg, arm, *, first: int, torn_last: bool = False) -> None:
    """What a sweep stopped part way through `arm`'s first run leaves: the sweep's own files, and the arm's rows for the
    first `first` cases to tune on, the last of them cut short when `torn_last`, as by a process stopped mid-line."""
    from mnemiq.sweep.run import _begin, arm_folder, run_arm

    _begin(cfg)
    folder = arm_folder(cfg, arm, 1)
    run_arm(cfg, arm, tuning(cfg)[:first], folder)
    if torn_last:
        rows = folder / "rows.jsonl"
        text = rows.read_text().rstrip("\n")
        last = text.rsplit("\n", 1)[-1]
        rows.write_text(text[:len(text) - len(last) // 2])


def config_with(*, held_out: float = 0.3, split_seed: int = 7, strategy: str = "screen-then-grid"):
    """A config for the parts that read no file: validated as `load_config` validates it, without opening any file it
    names."""
    from mnemiq.sweep.config import SweepConfig

    return SweepConfig.model_validate({
        "sweep": {"name": "pure", "out": "sweep-out"},
        "dataset": {"questions": "questions.json", "env": "deployment.env", "held_out": held_out,
                    "split_seed": split_seed, "identity": {"roles": [ROLE]}},
        "models": {"m": {"llm_model": "stub-model", "llm_base_url": UNREACHABLE, "api_key_env": KEY_ENV}},
        "modes": ["instant"],
        "search": {"strategy": strategy},
    })


def cases(*, answerable: int, unanswerable: int) -> list[EvaluationCase]:
    """Cases with nothing behind them but their ids and whether each has an answer."""
    return ([EvaluationCase(id=f"a{i:03d}", question=f"answerable {i}", gold_sql="SELECT 1") for i in range(answerable)]
            + [EvaluationCase(id=f"u{i:03d}", question=f"unanswerable {i}", answerable=False)
               for i in range(unanswerable)])


def answerable(some: list[EvaluationCase]) -> int:
    return sum(case.answerable for case in some)


def tiny_dataset(tmp_path: Path, *, cases: int, answerable: int, stub: StubModel | None = None) -> dict[str, str]:
    """A question file of `cases` synthetic questions over the tiny tables, `answerable` of them with gold SQL and the
    rest to decline. Returned as the `[dataset]` keys that name it, for `write_config(dataset=...)`.

    A stub declines a question it does not know. Given `stub`, it learns these, each answerable one with a wrong
    answer that returns another count."""
    written = [{"id": f"after-{1900 + i}", "question": f"How many books were published after {1900 + i}?",
                "gold_sql": f"SELECT COUNT(*) AS books FROM books WHERE published_year > {1900 + i}"}
               for i in range(answerable)]
    written += [{"id": f"price-{i}", "question": f"What did the library pay for book {i}?", "gold_sql": None,
                 "answerable": False} for i in range(cases - answerable)]
    path = tmp_path / f"questions-{cases}-{answerable}.json"
    path.write_text(json.dumps(written, indent=2))
    if stub is not None:
        stub.learn(case | ({"wrong_sql": case["gold_sql"].replace(">", "<")} if case["gold_sql"] else {})
                   for case in written)
    return {"questions": str(path)}


def live_postgres_env(tmp_path: Path) -> TinyDeployment:
    """The tiny deployment's files with its env file naming a Postgres source instead of the SQLite one. Nothing listens
    at the address, so anything that connects fails at once."""
    d = tiny_files(tmp_path)
    env = d.root / "live.env"
    values = {"source_id": "library", "pg_dsn": "postgresql://sweep@127.0.0.1:9/library", "store_path": str(d.store),
              "authz_path": str(d.authz), "embed_base_url": UNREACHABLE, "embed_api_key": STUB_KEY}
    env.write_text("".join(f"MNEMIQ_{name.upper()}='{value}'\n" for name, value in values.items()))
    return replace(d, env=env)


def arm_with(cfg, *, model: str = "m", mode: str = "instant", **changes):
    """The arm that changes `changes` from the start of `model` in `mode`; a mode setting is named bare, as
    `candidates=3`."""
    from mnemiq.sweep.arms import arm_of

    named = {name if name in PROFILE_SETTINGS else f"{mode}.{name}": value for name, value in changes.items()}
    assert model in cfg.models and mode in cfg.modes
    return arm_of(model, mode, named)


def start_of(cfg, mode: str, *, model: str = "m"):
    """The screen's start arm for `model` in `mode`."""
    from mnemiq.sweep.arms import screen_arms

    return next(a for a in screen_arms(cfg) if a.model == model and a.mode == mode and not a.changes)


def two_deployments(tmp_path: Path, stub: StubModel) -> dict:
    """The tiny deployment read as two, `lib-a` and `lib-b`, with the tiny cases split between them by db_id: two env
    files naming the same store and source, `lib-b`'s with a longer question limit, which its profile's version holds.
    Returned as the `[dataset]` keys that name them, for `write_config(dataset=...)`."""
    d = tiny_deployment(tmp_path, stub)
    envs = {name: d.root / f"{name}.env" for name in ("lib-a", "lib-b")}
    envs["lib-a"].write_text(d.env.read_text())
    envs["lib-b"].write_text(d.env.read_text() + "MNEMIQ_MAX_QUESTION_CHARS='2000'\n")
    questions = d.root / "questions-two-deployments.json"
    questions.write_text(json.dumps([_graded(case) | {"db_id": ("lib-a", "lib-b")[number % 2]}
                                     for number, case in enumerate(CASES)]))
    return {"questions": str(questions), "deployments": {name: str(env) for name, env in envs.items()}}


def one_arm(tmp_path: Path, stub: StubModel, *, env_extra: dict[str, str] | None = None,
            api_key_env: str | None = None, **sections):
    """A loaded config over the tiny deployment on `stub`, and the screen's start arm in the config's first mode.
    `env_extra` goes to `tiny_deployment`, `api_key_env` names model `m`'s key variable, and each other keyword is a
    config section."""
    from mnemiq.sweep.config import load_config

    d = tiny_deployment(tmp_path, stub, env_extra=env_extra)
    if api_key_env is not None:
        sections["models"] = {"m": {"llm_model": "stub-model", "llm_base_url": stub.url, "api_key_env": api_key_env}}
    cfg = load_config(write_config(tmp_path, deployment=d, model=stub, **sections))
    return cfg, start_of(cfg, cfg.modes[0])


def judged_by(stub: StubModel) -> dict:
    """A `[judge]` section naming `stub` as every model's judge."""
    return {"verify_model": "judge-model", "verify_base_url": stub.url, "verify_api_key_env": KEY_ENV}


def tuning(cfg) -> list[EvaluationCase]:
    """The cases the search runs on."""
    from mnemiq.sweep.config import load_questions
    from mnemiq.sweep.split import split

    return split(cfg, load_questions(cfg))["tuning"]


def read_rows(cfg, arm, repeat: int = 1) -> list[dict]:
    """The rows of one run of `arm`, as written."""
    from mnemiq.sweep.run import arm_folder

    path = arm_folder(cfg, arm, repeat) / "rows.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def row(case_id: str, outcome: str | None, *, mode: str = "instant", made_up: bool = False,
        error_kind: str | None = None, verify_layer: str | None = None, version: str = "v1", ms: float = 1000.0,
        rows_ms: float | None = 800.0, prompt_tokens: int = 900) -> dict:
    """One row as `run_arm` writes it, for a test that reads rows without running an arm. An error row's model error
    is a server that stopped answering when its kind is the endpoint's."""
    answered = outcome in ("correct", "correct_facts", "wrong")
    return {"id": case_id, "mode": mode, "outcome": outcome, "made_up": made_up,
            "sql": "SELECT COUNT(*) AS books FROM books" if answered else "", "ms": ms,
            "rows_ms": rows_ms if answered else None, "calls": 2, "prompt_tokens": prompt_tokens,
            "completion_tokens": 40, "profile_version": version,
            "engine_build": {"commit": None, "source_sha": "0" * 64}, "error_kind": error_kind,
            "model_error": "status_503" if error_kind == "endpoint" else None, "verify_layer": verify_layer}


def write_run(cfg, arm, rows: list[dict], repeat: int = 1) -> Path:
    """One run of `arm` as `run_arm` leaves it: its profile file and its rows, in its folder, which is returned."""
    from mnemiq.sweep.arms import profile_text
    from mnemiq.sweep.run import arm_folder

    folder = arm_folder(cfg, arm, repeat)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "profile.toml").write_text(profile_text(cfg, arm))
    (folder / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return folder


def answers_file(tmp_path: Path, *, gold_for: int = 0, wrong_for: int = 0, declined: int = 0) -> Path:
    """Another system's answers to the tiny cases: its gold SQL for the first `gold_for` answerable cases, SQL that
    returns other rows for the next `wrong_for`, and a decline for the first `declined` cases with no answer."""
    answerable = [case for case in CASES if case.get("answerable", True)]
    unanswerable = [case for case in CASES if not case.get("answerable", True)]
    answers = [{"id": case["id"], "sql": case["gold_sql"]} for case in answerable[:gold_for]]
    answers += [{"id": case["id"], "sql": case["wrong_sql"]} for case in answerable[gold_for:gold_for + wrong_for]]
    answers += [{"id": case["id"], "declined": True} for case in unanswerable[:declined]]
    path = tmp_path / "other-system-answers.json"
    path.write_text(json.dumps(answers, indent=2))
    return path


def out_files(cfg) -> list[Path]:
    """Every file under the sweep's out folder."""
    return [path for path in Path(cfg.sweep.out).rglob("*") if path.is_file()]


def touch_store(cfg, data: bytes) -> None:
    """Append `data` to the deployment's own store, as anything else writing to it would change it."""
    from mnemiq.sweep.config import deployment_settings

    with open(deployment_settings(cfg, None).store_path, "ab") as fh:
        fh.write(data)


def cancel_gold_once(monkeypatch, case_id: str = CASES[0]["id"]) -> None:
    """The source interrupts one case's gold query the first time it runs, as a statement bound does under load. The
    answer's own SQL reaches the source with the guard's row limit added, so only the grader's run of the gold matches;
    were the answer's run cancelled instead, it would be repaired, and the case would end with no error."""
    from mnemiq.adapters.duckdb import DuckDBAdapter

    gold = next(case["gold_sql"] for case in CASES if case["id"] == case_id)
    run = DuckDBAdapter.execute_arrow
    cancelled: list[str] = []

    def execute_arrow(self, sql, timeout_s=None):
        if sql == gold and not cancelled:
            cancelled.append(sql)
            raise duckdb.InterruptException("INTERRUPT Error: Interrupted!")
        return run(self, sql, timeout_s)

    monkeypatch.setattr(DuckDBAdapter, "execute_arrow", execute_arrow)


def source_down(monkeypatch) -> None:
    """Every statement run on the source fails as an attached source that cannot be read does: DuckDB's own word for a
    source that could not be asked, which the engine reads as the source not answering."""
    from mnemiq.adapters.duckdb import DuckDBAdapter

    def execute_arrow(self, sql, timeout_s=None):
        raise duckdb.OperationalError("IO Error: the attached database cannot be read")

    monkeypatch.setattr(DuckDBAdapter, "execute_arrow", execute_arrow)


def copy_package(tmp_path: Path) -> Path:
    """A copy of the installed `mnemiq` package's Python source."""
    import mnemiq

    source = Path(mnemiq.__file__).resolve().parent
    target = tmp_path / "mnemiq"
    for path in source.rglob("*.py"):
        copied = target / path.relative_to(source)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copied)
    return target


@pytest.fixture(autouse=True)
def _model_key(monkeypatch):
    """The key variable the default config names holds a key, as a user's shell would; a test that needs it unset
    removes it. A stray `MNEMIQ_PROFILE` in the shell running the tests would make the engine refuse every profile."""
    monkeypatch.setenv(KEY_ENV, STUB_KEY)
    monkeypatch.delenv("MNEMIQ_PROFILE", raising=False)


def fail_on_stub_errors(stub: StubModel) -> None:
    """Fail the test when a rule, predicate or callback raised inside `stub` or a stub it started. The stub answered
    each with a 500, which a test asserting on scripted failures would otherwise read as one of them."""
    if errors := stub.all_errors():
        pytest.fail(f"a stub's rule, predicate or callback raised {len(errors)} time(s); the first: {errors[0]!r}",
                    pytrace=False)


@pytest.fixture
def stub_model():
    """A running scripted model. Call it, `stub_model(**options)`, to start another; all stop at teardown, and the test
    fails there if any of them kept an error."""
    stub = StubModel()
    try:
        yield stub
    finally:
        stub.stop()
    fail_on_stub_errors(stub)


# --- a sweep's rows written as `run_arm` writes them, so the proposal's rule is tested without a model ---------------


def label_changes(label: str) -> dict:
    """An arm's label, each `setting=value` joined by ` + `, as its changes: a value read as the config types it."""
    changes = {}
    for part in label.split(" + "):
        name, _, value = part.partition("=")
        try:
            changes[name] = json.loads(value)
        except ValueError:
            changes[name] = value
    return changes


_STAMPED: dict[tuple[str, str], str] = {}


def stamped_version(cfg, arm) -> str:
    """The version a run of `arm` stamps on its rows: the engine's own profile loader and version function over the file
    the run writes, on the deployment's baseline, with the dialect of the tiny deployment's DuckDB adapter."""
    from mnemiq.profiles import load_profiles, profile_version
    from mnemiq.sweep.arms import baseline, profile_text
    from mnemiq.sweep.config import deployment_settings

    text = profile_text(cfg, arm)
    key = (str(cfg.dataset.env), text)
    if key not in _STAMPED:
        with tempfile.TemporaryDirectory(prefix="sweep-test-version-") as folder:
            path = Path(folder) / "profiles.toml"
            path.write_text(text)
            settings = baseline(deployment_settings(cfg, None)).model_copy(update={"profiles_path": str(path)})
            profiles, default = load_profiles(settings)
            _STAMPED[key] = profile_version(profiles[default].settings, "duckdb")
    return _STAMPED[key]


def graded_rows(arm, cases: list[EvaluationCase], *, right: int, version: str, spec: dict | None = None) -> list[dict]:
    """One run of `arm` on `cases`, in each of its modes. Without `spec` it is a start that gets the first `right`
    answerable cases right and declines the rest. A `spec` departs from that start by `only` (right where the start is
    not), `other` (not right where the start is), `more_wrong` and `more_made_up` (answers beyond the start's), every
    row taking `ms` and `prompt` (its prompt tokens). Every other answer is a decline."""
    spec = spec or {}
    only, other = spec.get("only", 0), spec.get("other", 0)
    wrong, made_up = spec.get("more_wrong", 0), spec.get("more_made_up", 0)
    answerable = [case.id for case in cases if case.answerable]
    unanswerable = [case.id for case in cases if not case.answerable]
    assert other <= right and right + only + wrong <= len(answerable) and made_up <= len(unanswerable)
    timed = {"ms": spec.get("ms", 1000.0), "prompt_tokens": spec.get("prompt", 900)}

    def outcome(i: int) -> str:
        if i < other or i >= right + only + wrong:
            return "deferred_wrongly"
        return "correct" if i < right + only else "wrong"

    rows = []
    for mode in arm.modes:
        rows += [row(case, outcome(i), mode=mode, version=version, **timed) for i, case in enumerate(answerable)]
        rows += [row(case, "wrong", mode=mode, made_up=True, version=version, **timed) if i < made_up
                 else row(case, "deferred_correctly", mode=mode, version=version, **timed)
                 for i, case in enumerate(unanswerable)]
    return rows


class ScriptedRounds:
    """The rounds a sweep runs after its search, as `after_search` drives them, with no worker and no model: each run it
    is asked for is written as `run_arm` writes it. A timed run copies the arm's own rows from the round that tuned it,
    and calls its check before each case, as the worker running it would. A confirm run answers the held-out cases: the
    start gets none right, and the assembled arm all of them, less what `held_out["assembled"]` asks of it."""

    def __init__(self, cfg, state: dict, held_out: dict) -> None:
        from mnemiq.sweep.run import _parts

        self.cfg, self.state, self.held_out = cfg, state, held_out
        self.parts = _parts(cfg)
        self.halted = False
        self.drained: list[tuple[int | None, list]] = []  # each drain's parallel override and its jobs

    def job(self, arm, repeat: int, cases=None, before=None):
        from mnemiq.sweep.run import _Job, arm_folder

        return _Job(arm, repeat, tuple(self.parts["tuning"] if cases is None else cases),
                    arm_folder(self.cfg, arm, repeat), before)

    def drain(self, pending, advance=None, *, parallel=None) -> None:
        jobs = list(pending)
        self.drained.append((parallel, jobs))
        for job in jobs:
            if not job.left():
                continue
            rows = self._answers(job)
            if job.before is not None:
                for answer in rows:
                    job.before(answer["id"], answer["mode"])
            write_run(self.cfg, job.arm, rows, job.repeat)

    def _answers(self, job) -> list[dict]:
        if job.arm.round == "timing":
            ids = {case.id for case in job.cases}
            for tuned in ("screen", "grid"):
                if found := read_rows(self.cfg, replace(job.arm, round=tuned)):
                    return [answer for answer in found if answer["id"] in ids]
            raise AssertionError(f"{job.arm.label} was timed but never tuned")
        version = stamped_version(self.cfg, job.arm)
        cases = list(job.cases)
        if job.arm.profile_id == "start":
            return graded_rows(job.arm, cases, right=0, version=version)
        given = self.held_out.get("assembled", {})
        spec = {"only": answerable(cases) - given.get("more_wrong", 0), **given}
        return graded_rows(job.arm, cases, right=0, version=version, spec=spec)

    def save(self) -> None:
        (out(self.cfg) / "sweep.json").write_text(json.dumps(self.state))


def play_after_search(cfg, *, held_out: dict | None = None, hosted: bool = False, server_running: int = 0,
                      load: float | Callable[[], float] = 0.5, wait_s: float = 0.0) -> ScriptedRounds:
    """The timing and confirm rounds of a sweep whose search is written, played by `ScriptedRounds` through the code
    `run` calls. The server reads `server_running` requests running before each timed case, or publishes no count when
    `hosted`. The sweep machine has 8 cores, and its 1-minute load reads `load`, or each value `load()` returns in turn.
    Before the timing round the sweep waits up to `wait_s` for that load to fall under the cores, reading it again at
    once each time."""
    from mnemiq.sweep import propose

    read = load if callable(load) else (lambda: load)
    rounds = ScriptedRounds(cfg, sweep_json(cfg), held_out or {})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(propose, "server_running", lambda base_url: None if hosted else server_running)
        patch.setattr(propose, "machine_load", lambda: (read(), 8))
        patch.setattr(propose, "LOAD_WAIT_S", wait_s)
        patch.setattr(propose, "LOAD_WAIT_EVERY_S", 0.0)
        propose.after_search(rounds, None)
    return rounds


def idle_deployment(tmp_path: Path) -> TinyDeployment:
    """The tiny deployment, its store built, on a stub that is stopped once it is: a sweep whose rows are written calls
    no model."""
    with StubModel() as idle:
        return tiny_deployment(tmp_path, idle)


def graded_sweep(tmp_path: Path, *, arms: dict | None = None, grid: dict | None = None, picks: dict | None = None,
                 modes=("instant",), held_out: dict | None = None, hosted: bool = False, server_running: int = 0,
                 load: float = 0.5, real_profiles: StubModel | None = None, cases: tuple[int, int] = (14, 12),
                 play: bool = True, no_room: tuple[str, ...] = (), **sections):
    """A screen-then-grid sweep of model `m` over `cases` (questions, answerable) synthetic questions, its search written
    as rows, one run per arm, loaded. By default 9 answerable questions and 2 to decline are tuned on, and 3 answerable
    held out.

    `arms` (screen) and `grid` map a label to a dict for `graded_rows`, in every mode. The start gets right as many
    answerable cases as the largest `other` asked for. `picks` maps each mode to the label that wins there (9 more right
    and none lost) or to "start", where that label wins nothing.

    `no_room` names the modes whose search the record says stopped for want of room after the start's first run.

    The timing and confirm rounds follow (`play_after_search`, with `hosted`, `server_running` and `load`), unless
    `play` is false. With `real_profiles`, a stub, they run for real instead, `run` driving them in workers on the
    product path, on that stub."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    modes = list(modes)
    per_mode = {mode: dict(arms or {}) for mode in modes}
    if picks:
        won = sorted({label for label in picks.values() if label != "start"})
        per_mode = {mode: {label: dict(only=9, other=0) if picks[mode] == label else dict(only=0, other=0)
                           for label in won} for mode in modes}
    space: dict[str, list] = {}
    for label in sorted({label for mine in per_mode.values() for label in mine} | set(grid or {})):
        for name, value in label_changes(label).items():
            if value not in space.setdefault(name, []):
                space[name].append(value)
    dataset = tiny_dataset(tmp_path, cases=cases[0], answerable=cases[1], stub=real_profiles)
    deployment = tiny_deployment(tmp_path, real_profiles) if real_profiles is not None else idle_deployment(tmp_path)
    from mnemiq.sweep.arms import arm_of
    from mnemiq.sweep.config import load_config
    from mnemiq.sweep.run import _begin, _parts, run

    cfg = load_config(write_config(tmp_path, deployment=deployment, model=real_profiles, dataset=dataset, space=space,
                                   modes=modes, sweep={"repeats": 1}, **sections))
    state = _begin(cfg) | {"starts": {"m": {mode: None for mode in modes}}}   # as the first round's plan records it
    (out(cfg) / "sweep.json").write_text(json.dumps(state))
    if no_room:
        stopped = {"misses": 0, "needed": None, "compared": 1, "stopped": "no room"}
        state["room"] = {"m": {mode: stopped for mode in no_room}}
        (out(cfg) / "sweep.json").write_text(json.dumps(state))
    tuned = _parts(cfg)["tuning"]
    for mode in modes:
        specs = [*per_mode[mode].values(), *(grid or {}).values()]
        right = max((spec.get("other", 0) for spec in specs), default=0)
        start = arm_of("m", mode, {}, round="screen")
        write_run(cfg, start, graded_rows(start, tuned, right=right, version=stamped_version(cfg, start)))
        for round, labelled in (("screen", per_mode[mode]), ("grid", grid or {})):
            for label, spec in labelled.items():
                arm = arm_of("m", mode, label_changes(label), round=round)
                write_run(cfg, arm, graded_rows(arm, tuned, right=right, version=stamped_version(cfg, arm), spec=spec))
    if real_profiles is not None:
        run(cfg)
    elif play:
        play_after_search(cfg, held_out=held_out, hosted=hosted, server_running=server_running, load=load)
    return cfg


def checked_rows(arm, cases: list[EvaluationCase], spec: dict, version: str) -> list[dict]:
    """One run of a check-mode arm: its first `errors` cases failed by the model's endpoint, so not measured; then
    `right` and `wrong` answerable cases, and `made_up` answers to cases that have none; every other case declined."""
    errors, right, wrong, made_up = (spec.get(name, 0) for name in ("errors", "right", "wrong", "made_up"))
    rows = []
    for mode in arm.modes:
        answered = invented = 0
        for number, case in enumerate(cases):
            if number < errors:
                rows.append(row(case.id, "error", mode=mode, error_kind="endpoint", version=version))
            elif case.answerable:
                outcome = "correct" if answered < right else "wrong" if answered < right + wrong else "deferred_wrongly"
                answered += 1
                rows.append(row(case.id, outcome, mode=mode, version=version))
            else:
                rows.append(row(case.id, "wrong", mode=mode, made_up=True, version=version) if invented < made_up
                            else row(case.id, "deferred_correctly", mode=mode, version=version))
                invented += 1
        assert errors >= len(cases) or (answered >= right + wrong and invented >= made_up)
    return rows


def checked_sweep(tmp_path: Path, *, start: dict, defaults: dict, modes=("instant",)):
    """A check-mode sweep of model `m` already run, one run per arm, written as rows, over 26 synthetic questions, 17 of
    them with an answer and 9 to decline, loaded. `start` and `defaults` are dicts for `checked_rows`."""
    from mnemiq.sweep.arms import check_arms
    from mnemiq.sweep.config import load_config
    from mnemiq.sweep.run import _begin, _parts

    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = load_config(write_config(tmp_path, deployment=idle_deployment(tmp_path), modes=list(modes),
                                   dataset=tiny_dataset(tmp_path, cases=26, answerable=17),
                                   search={"strategy": "check"}, sweep={"repeats": 1},
                                   start=start_that_differs(tmp_path)))
    state = _begin(cfg) | {"starts": {"m": {mode: None for mode in modes}}}   # as the first round's plan records it
    (out(cfg) / "sweep.json").write_text(json.dumps(state))
    for arm in check_arms(cfg):
        spec = start if arm.profile_id == "start" else defaults
        write_run(cfg, arm, checked_rows(arm, _parts(cfg)["tuning"], spec, stamped_version(cfg, arm)))
    play_after_search(cfg)
    return cfg


def evidence(cfg, model: str = "m") -> dict:
    """The evidence `propose` wrote for `model`."""
    return json.loads((out(cfg) / "proposal" / model / "evidence.json").read_text())


def proposal_file(cfg, model: str = "m") -> Path:
    return out(cfg) / "proposal" / model / "profiles.toml"


def _assembled_folders(cfg, model: str) -> list[Path]:
    return sorted(folder for folder in (out(cfg) / model / "confirm").glob("*/assembled-*") if folder.is_dir())


def confirm_arms_run(cfg, model: str = "m") -> int:
    """How many assembled arms the confirm round ran for `model`."""
    return len(_assembled_folders(cfg, model))


def confirm_modes(cfg, model: str = "m") -> list[str]:
    """The modes the confirm round's assembled arm answered in, from its rows."""
    (folder,) = _assembled_folders(cfg, model)
    return sorted({answer["mode"] for answer in json_lines(folder / "run-1" / "rows.jsonl")})


def json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


# --- finished sweeps a starter is published from ----------------------------------------------------------------------

# The start's own mode sections read the schema as `CREATE TABLE` (`card_style = "ddl"`), which the engine's defaults
# never send the writer, so a scripted rule can tell the start's requests from the defaults'.
_START_CARDS = "ddl"
_START_MARK = "CREATE TABLE"


def _start_rule(stub: StubModel, *, declines: int, wrong: bool, from_ask: int = 1):
    """A rule for the scripted model: under the start's settings, as its cards show, it declines the first `declines`
    answerable cases it knows from its `from_ask`-th ask of each on (in instant mode, one ask a run), or with `wrong`
    answers every case wrong. Every other request is answered as scripted."""
    asked: dict[str, int] = {}
    lock = threading.Lock()

    def rule(request: dict) -> str | None:
        system, user = _messages(request)
        if _WRITER not in system or _START_MARK not in f"{system}\n{user}":
            return None
        case = stub._script.find(user)
        if case is None:
            return None
        with lock:
            asked[case.id] = count = asked.get(case.id, 0) + 1
        if wrong and case.wrong_sql:
            return _proposal(case.wrong_sql)
        if count >= from_ask and case.answerable and stub._script.answerable_rank(case) < declines:
            return _proposal(None, "declined under the start's settings")
        return None

    return rule


def _start_profile(tmp_path: Path, settings: dict, marked: list[str]) -> dict:
    """A `[start]` section naming a profile that sets `settings` profile-wide and gives each mode of `marked` a section
    whose cards the scripted model can tell from the defaults' (`_START_CARDS`)."""
    lines = ['default = "start"', "[profiles.start]",
             *(f"{name} = {_toml_value(value)}" for name, value in settings.items())]
    for mode in marked:
        lines += [f"[profiles.start.{mode}]", f'card_style = "{_START_CARDS}"']
    path = write(tmp_path / "start-for-a-starter.toml", "\n".join(lines) + "\n")
    return {"profile_file": str(path)}


def finished_check(tmp_path: Path, monkeypatch, stub: StubModel, *, build=CONFIG_BUILD, llm_context_window=STUB_WINDOW,
                   root: str | None = None, reports_build: bool = True, dataset: dict | None = None, modes=("instant",),
                   defaults_better: bool = False, person_chose: str | None = None, propose: dict | None = None,
                   start_sets: dict | None = None, unfinished: bool = False, config_edited_after: bool = False,
                   start_declines: int | dict[str, int] = 0, answerable_only: bool = False,
                   start_file: Path | None = None, repeats: int = 1, declining_from_run: int = 1):
    """A check-mode sweep of model `m` over the tiny deployment, run on the product path and proposed, loaded, with the
    starters folder patched to `tmp_path / "starters"`. The scripted model answers every case right, and `[models.m]`
    states `build` and `llm_context_window` (each omitted when None). `repeats` runs per arm, one by default.

    The start sets `start_sets` profile-wide (by default retrieval_k = 12), and each mode it is told apart in has a
    section whose cards the scripted model can read (`_start_rule`): every measured mode, or with `start_declines` a
    dict, the modes it names.

    - `root` and `reports_build` go to the scripted model; `dataset` replaces the `[dataset]` keys it names, as
      `two_deployments` gives them; `modes` are the modes the sweep runs; `propose` is its `[propose]` section.
    - `defaults_better`: under the start's settings every answer is wrong, so the rule proposes the defaults.
    - `person_chose`: a person's pick (`propose --choose`) in place of the rule's.
    - `start_declines`: under the start's settings the model declines that many answerable cases it answers right under
      the defaults', so the start has fewer right and no more wrong, and the rule still proposes it. An int applies in
      every measured mode; a dict names the modes, the start level elsewhere. With `declining_from_run`, the declines
      begin at that run of the start, which is level with the defaults before it (instant mode asks a case once a run).
    - `answerable_only`: the tiny question file without its cases that have no answer.
    - `unfinished`: a budget no arm fits in, so the run stops before the first arm and nothing is proposed.
    - `config_edited_after`: the config is edited once the proposal is written, and loaded again.
    - `start_file`: the start is that profiles file's default instead, such as a search's proposal."""
    from mnemiq.sweep.config import load_config
    from mnemiq.sweep.propose import propose as proposing
    from mnemiq.sweep.run import run

    tmp_path.mkdir(parents=True, exist_ok=True)
    starter_folder(tmp_path, monkeypatch, {})
    modes = list(modes)
    named = start_declines if isinstance(start_declines, dict) else dict.fromkeys(modes, start_declines)
    assert len(set(named.values())) <= 1, "the start declines as many cases in each mode it is told apart in"
    marked = [mode for mode in modes if mode in named]
    rule = _start_rule(stub, declines=next(iter(named.values()), 0), wrong=defaults_better, from_ask=declining_from_run)
    model = stub(rule, gold=True, reports_build=reports_build, **({"root": root} if root is not None else {}))
    deployment = tiny_deployment(tmp_path, model)
    questions = None
    if answerable_only:
        questions = write(tmp_path / "questions-answerable.json",
                          json.dumps([_graded(case) for case in CASES if case.get("answerable", True)], indent=2))
    sections: dict = {
        "search": {"strategy": "check"}, "sweep": {"repeats": repeats}, "modes": modes,
        "start": ({"profile_file": str(start_file)} if start_file is not None
                  else _start_profile(tmp_path, {"retrieval_k": 12} if start_sets is None else start_sets, marked)),
        "models": one_model(model, build=build, llm_context_window=llm_context_window),
    }
    if dataset is not None:
        sections["dataset"] = dataset
    if propose is not None:
        sections["propose"] = propose
    if unfinished:
        sections["search"] |= {"budget_hours": 0.1, "seconds_per_question": 1000.0}
    cfg = load_config(write_config(tmp_path, deployment=deployment, model=model, questions=questions, **sections))
    run(cfg)
    if unfinished:
        return cfg
    proposing(cfg, person_chose)
    if config_edited_after:
        cfg = load_config(rewrite(cfg, propose={"alpha": 0.1}))
    return cfg


def finished_search(tmp_path: Path, monkeypatch, stub: StubModel, *, start: bool = False, confirmed: bool = True):
    """A screen-then-grid search of model `m` over 16 questions, run on the product path and proposed, loaded, with the
    starters folder patched to `tmp_path / "starters"`. Its one arm is guided_sql = true, and the scripted model answers
    right only when the writer's request carries `response_format`, as guided decoding sends, so that arm is picked and
    confirmed on the held-out part. `[models.m]` states `CONFIG_BUILD` and the server's window. One run per arm.

    - `start`: the search starts from a profile of its own (`start_that_differs`), not the engine's defaults.
    - `confirmed` false: on the held-out questions the model answers right only without guided decoding, so the pick
      fails the confirm round, and the rule falls back to the start."""
    from mnemiq.sweep import propose as proposal
    from mnemiq.sweep.config import load_config, load_questions
    from mnemiq.sweep.run import run
    from mnemiq.sweep.split import split

    tmp_path.mkdir(parents=True, exist_ok=True)
    starter_folder(tmp_path, monkeypatch, {})
    # The wait for a quiet machine before the timing round would only add minutes on a busy machine.
    monkeypatch.setattr(proposal, "LOAD_WAIT_S", 0.0)
    held_out: set[str] = set()   # the held-out questions once the split is drawn, when the pick is to fail there

    def gold_when(request: dict) -> bool:
        case = stub._script.find(_messages(request)[1])
        return ("response_format" in request) != (case is not None and case.question in held_out)

    model = stub(gold_when=gold_when)
    sections: dict = {"space": {"guided_sql": [True]}, "sweep": {"repeats": 1},
                      "models": one_model(model, build=CONFIG_BUILD, llm_context_window=STUB_WINDOW)}
    if start:
        sections["start"] = start_that_differs(tmp_path)
    cfg = load_config(write_config(tmp_path, deployment=tiny_deployment(tmp_path, model, cases=16), model=model,
                                   **sections))
    if not confirmed:
        held_out.update(case.question for case in split(cfg, load_questions(cfg))["held_out"])
    run(cfg)
    proposal.propose(cfg)
    return cfg


def evidence_file(cfg, model: str = "m") -> Path:
    """The evidence file `propose` wrote for `model`."""
    return out(cfg) / "proposal" / model / "evidence.json"


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def plant(path: Path, /, **fields) -> None:
    """Add `fields` to the JSON object at `path`, as a later field of the proposal's evidence might. Positional, so a
    field may itself be named `path`."""
    write(path, json.dumps(read_json(path) | fields, indent=2))


def drafted_versions(cfg, model: str = "m") -> dict:
    """The proposal evidence's `version` as a map from deployment to version: one version is `{None: version}`."""
    version = evidence(cfg, model)["version"]
    return {None: version} if isinstance(version, str) else dict(version)


def deployment_file_naming(name: str, cfg, model: str = "m") -> str:
    """A deployment's profiles file whose default names the starter `name`, with the model's endpoint and key variables
    from `[models]`, as a deployment serving that build writes it."""
    from mnemiq.sweep.arms import _endpoint

    entry = cfg.models[model]
    judge_key = (cfg.judge if cfg.judge is not None else entry).verify_api_key_env
    lines = ['default = "deployed"', "[profiles.deployed]", f"starter = {_toml_value(name)}",
             f"api_key_env = {_toml_value(entry.api_key_env)}"]
    lines += [f"verify_api_key_env = {_toml_value(judge_key)}"] if judge_key else []
    lines += [f"{field} = {_toml_value(value)}" for field, value in _endpoint(cfg, model).items() if value is not None]
    return "\n".join(lines) + "\n"


def drops_the_instant_section(text: str, name: str, dataset: str) -> str:
    """`starter_text`, with its `.instant` table taken out: a starter that no longer resolves to what was measured."""
    lines, kept, inside = _starter_text(text, name, dataset).splitlines(keepends=True), [], False
    for line in lines:
        if line.startswith("["):
            inside = line.rstrip("\n").endswith(".instant]")
        if not inside:
            kept.append(line)
    assert len(kept) < len(lines), "the starter holds no instant table to take out"
    return "".join(kept)


def missing_from(text: str, names) -> list[str]:
    """Each of `names` the doc's "Shipped starters" section does not name, in code."""
    match = re.search(r"^## Shipped starters\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, "the doc has no '## Shipped starters' section"
    return [name for name in names if f"`{name}`" not in match.group(1)]
