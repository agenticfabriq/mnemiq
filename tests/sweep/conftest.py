"""What every sweep test stands on: the scripted model, a tiny real deployment, and a config file around them.

`tiny_deployment` is a real one: a SQLite source of three tables, a store enriched and indexed by the engine's own
`enrich` and `build` commands, an access policy, an env file and graded cases. Only the model is scripted.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import duckdb
import pytest

from mnemiq.config import DEFAULT_RETRIEVAL_K, Settings
from mnemiq.contract import EvaluationCase
from mnemiq.profiles import PROFILE_SETTINGS

from .stub_model import StubModel

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


def tiny_deployment(tmp_path: Path, stub: StubModel, *, env_extra: dict[str, str] | None = None,
                    withhold: str | None = None, row_filter: bool = False, mask: bool = False) -> TinyDeployment:
    """The tiny deployment, its store built once per `tmp_path`: a second call returns what the first built, so a
    rebuilt store never moves a fingerprint. Its env file points at the first stub given, for chat and embeddings, and
    every stub given learns its cases. `env_extra` adds lines to the env file, as `{"MNEMIQ_NAME": "value"}`, and is
    taken only by the call that builds the deployment. `withhold` takes one table out of the sweep's role, `row_filter`
    gives the role a row filter, and `mask` a PII mask; the store depends on none of them."""
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
    stub.learn(CASES)
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
        "models": {"m": {"llm_model": "stub-model", "llm_base_url": model.url if model is not None else UNREACHABLE,
                         "api_key_env": KEY_ENV}},
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
