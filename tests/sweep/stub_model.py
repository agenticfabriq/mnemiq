"""A scripted model server: an OpenAI-compatible endpoint on 127.0.0.1 that answers the engine's real requests.

`StubModel` serves the doors the engine calls -- chat completions, embeddings, vLLM's `/tokenize` and `/metrics`, and
`/v1/models` -- from a script, so a test can drive the product path (`build_runtime`, `Runtime.ask`, `run_case`) with no
model behind it. A chat request is answered as whoever sent it, read from its system prompt: the SQL writer, the result
judge, deep mode's selector, the corrector, the sentence that follows the rows, and enrichment's column notes.

The writer is answered from the cases the stub has learned (`learn`): the case whose question the request carries.
Answering right means that case's gold SQL, or a decline when the case has no answer. Answering wrong means the case's
wrong SQL, which returns other rows, or rows for a question that has none. A plain stub answers wrong; `gold`, or
`varies` or `start_misses` passed without `gold_when`, answers right. A test can also pass a `rule`, which sees every
chat request and returns the reply text, an HTTP status, or None for the scripted reply (`answer`).

A rule, predicate or callback that raises is answered with a 500 and kept in `errors`; the `stub_model` fixture fails
the test at teardown when any stub it started kept one, so a broken rule never passes for a scripted failure.

Counts are of HTTP requests. A failure status goes out with `x-should-retry: false`, which the client obeys, so one failed
call is one request. A closed connection carries no header and the client retries it twice, so one failed call is three.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from mnemiq.agent import synthesize
from mnemiq.enrichment.prompts import system_prompt as notes_system_prompt
from mnemiq.execute import select
from mnemiq.generate import correct
from mnemiq.verify import judge

# A prompt longer than the window, as the client recognises it (`mnemiq.llm.client._context_too_long`): a 400 whose body
# carries both the OpenAI code and the words vLLM uses, so either reading maps it to `ContextTooLong`.
CONTEXT_TOO_LONG = (
    "This model's maximum context length is 32768 tokens. However, you requested 36864 tokens (32768 in the "
    "messages, 4096 in the completion). Please reduce the length of the messages or completion."
)
EMBED_DIM = 32
# The reply format that ends every form of the writer's system prompt (`generate.prompts.system_prompt`).
_WRITER = '{"sql": "<the SELECT, or null>"'
_DECLINE = "the tables on the cards hold no data for that"
_WINDOW_STEP = 1024

Rule = Callable[[dict], "str | int | None"]


@dataclass(frozen=True)
class ScriptedCase:
    """One question the stub can answer. `variants` spell the gold SQL differently and return its rows;
    `wrong_variants` do the same for `wrong_sql`."""

    id: str
    question: str
    gold_sql: str | None
    answerable: bool = True
    wrong_sql: str | None = None
    variants: tuple[str, ...] = ()
    wrong_variants: tuple[str, ...] = ()


def _scripted(case: Any) -> ScriptedCase:
    """A `ScriptedCase` from itself, a mapping, or anything with the attributes, such as an `EvaluationCase`."""
    if isinstance(case, ScriptedCase):
        return case
    if isinstance(case, Mapping):
        get = case.get
    else:
        def get(key, default=None):
            return getattr(case, key, default)
    return ScriptedCase(id=str(get("id")), question=get("question"), gold_sql=get("gold_sql"),
                        answerable=bool(get("answerable", True)), wrong_sql=get("wrong_sql"),
                        variants=tuple(get("variants", ()) or ()),
                        wrong_variants=tuple(get("wrong_variants", ()) or ()))


def _carries(text: str, question: str) -> bool:
    """The text asks this question: `QUESTION: <question>` ending at a line break or at the end."""
    marker = f"QUESTION: {question}"
    at = text.find(marker)
    while at >= 0:
        end = at + len(marker)
        if end == len(text) or text[end] == "\n":
            return True
        at = text.find(marker, at + 1)
    return False


class _Script:
    """The cases a stub has learned. Shared with every stub it starts, so a stub started after the deployment was built
    still knows the deployment's questions. Kept in the order learned, which is the order `start_misses` counts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cases: dict[str, ScriptedCase] = {}

    def learn(self, cases: Iterable[Any]) -> None:
        with self._lock:
            for case in cases:
                scripted = _scripted(case)
                self._cases[scripted.question] = scripted

    def find(self, text: str) -> ScriptedCase | None:
        with self._lock:
            found = [case for case in self._cases.values() if _carries(text, case.question)]
        return max(found, key=lambda case: len(case.question), default=None)

    def answerable_rank(self, case: ScriptedCase) -> int:
        with self._lock:
            answerable = [c.question for c in self._cases.values() if c.answerable]
        return answerable.index(case.question)


def embedding(text: str) -> list[float]:
    """The same unit vector for the same text, from any stub: a store built through one stub is read through another."""
    raw = [byte - 127.5 for byte in hashlib.sha256(text.encode()).digest()]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


def _tokens(chars: int) -> int:
    """Four characters a token, rounded up, and never within a few tokens of a multiple of 1,024. The client reads more
    than six characters a token, or a count on a window's edge, as a server that may have cut the prompt, and probes it
    with a second call (`LLMClient.complete`)."""
    tokens = max(1, -(-chars // 4))
    if tokens >= _WINDOW_STEP and min(tokens % _WINDOW_STEP, -tokens % _WINDOW_STEP) <= 8:
        tokens += 16
    return tokens


def _messages(request: dict) -> tuple[str, str]:
    """The system text and the user text of a chat request."""
    parts: dict[str, list[str]] = {"system": [], "user": []}
    for message in request.get("messages") or []:
        content = message.get("content")
        if isinstance(content, str) and message.get("role") in parts:
            parts[message["role"]].append(content)
    return "\n".join(parts["system"]), "\n".join(parts["user"])


def _proposal(sql: str | None, reason: str = "") -> str:
    return json.dumps({"sql": sql, "reason": reason or ("the cards answer it" if sql else _DECLINE)})


def _sql_to_correct(user: str) -> str:
    """The corrector's SQL, unchanged (`correct.user_text`): the stub fixes nothing."""
    body = user.removeprefix("SQL:\n")
    return body.split("\n\nProblem to fix", 1)[0]


def _column_notes(user: str) -> str:
    """A note for each column the enrichment prompt lists (`enrichment.prompts.render_table_facts`), none for the
    other columns it names for context."""
    names, listing = [], False
    for line in user.splitlines():
        if line == "COLUMNS:":
            listing = True
        elif listing and line.startswith("- "):
            names.append(line[2:].split(" (", 1)[0])
        elif listing:
            break
    return json.dumps({"columns": [
        {"name": name, "description": f"The {name.replace('_', ' ')} recorded on each row.",
         "semantic_type": None, "pii_level": "none", "code_meanings": {}}
        for name in names]})


class StubModel:
    """A threaded HTTP server on `127.0.0.1:0`, answering as a scripted model. See the module docstring.

    Options:
      rule          called with every chat request: reply text, an HTTP status to return, or None for `answer`.
      gold          answer every question right. `gold_when(request)` answers right only where it returns true.
                    With neither, a stub answers wrong, unless `varies` or `start_misses` is passed.
      sql           answer every writer call with this SQL, whatever the question.
      varies        repeated asks of one question get the answer's other spellings in turn, right or wrong, with the
                    same rows (a case's `variants` or `wrong_variants`). Passed without `gold_when`, right elsewhere.
      seconds       delay before each chat reply.
      start_misses  answer the first N answerable cases wrong on every request, however `gold` reads. Passed
                    without `gold_when`, right everywhere else.
      on_call       called with the running chat request count, as each request arrives.
      fail_first    the first N chat requests fail with `status`.
      fail_after    every chat request after the first N fails with `status`, until `recover()`.
      status        an HTTP code, "connect" (close the socket unanswered) or "context_too_long" (a 400 carrying the
                    window error the client maps to `ContextTooLong`).
      confidence    what the result judge answers.
      window        the context window `/tokenize` and `/v1/models` report.

    Calling a stub, `stub(**options)`, starts another that shares its cases and stops when it stops.
    """

    def __init__(self, rule: Rule | None = None, *, gold: bool = False,
                 gold_when: Callable[[dict], bool] | None = None, sql: str | None = None, varies: bool = False,
                 seconds: float = 0.0, start_misses: int = 0, on_call: Callable[[int], None] | None = None,
                 fail_first: int = 0, fail_after: int | None = None, status: int | str = 500,
                 confidence: float = 0.9, window: int = 32768, model: str = "stub-model",
                 script: _Script | None = None) -> None:
        if gold and gold_when is not None:
            raise ValueError("gold and gold_when say the same thing: pass one")
        if isinstance(status, bool) or not (isinstance(status, int) or status in ("connect", "context_too_long")):
            raise ValueError(f"status must be an HTTP code, 'connect' or 'context_too_long', not {status!r}")
        self.rule = rule
        # `varies` and `start_misses` describe how a right model departs from right, so either alone means one.
        right = gold or (gold_when is None and (varies or start_misses > 0))
        self.gold_when = (lambda request: True) if right else gold_when
        self.sql = sql
        self.varies = varies
        self.seconds = seconds
        self.start_misses = start_misses
        self.on_call = on_call
        self.fail_first = fail_first
        self.fail_after = fail_after
        self.status = status
        self.confidence = confidence
        self.window = window
        self.model = model
        # What a test reads.
        self.calls = 0  # chat requests, failed ones included
        self.requests: list[dict] = []  # each chat request's body, in arrival order
        self.embedding_calls = 0
        self.tokenize_calls = 0
        self.running: int | None = None  # `/metrics` reports this when set, else the chat requests in flight
        # Raised by a rule, a predicate or `on_call`. Each is answered as a 500, so `stub_model` fails the test for it.
        self.errors: list[BaseException] = []
        self._lock = threading.Lock()
        self._in_flight = 0
        self._recovered = False
        self._asked: dict[str, int] = {}  # writer calls per question, for `varies`
        self._script = script or _Script()
        self._started: list[StubModel] = []
        self._stopped = False
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.stub = self
        self._thread = threading.Thread(target=self._server.serve_forever, name="stub-model", daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __call__(self, rule: Rule | None = None, **options) -> StubModel:
        other = StubModel(rule, script=self._script, **options)
        self._started.append(other)
        return other

    def __enter__(self) -> StubModel:
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def learn(self, cases: Iterable[Any]) -> None:
        """Know these questions: `EvaluationCase`s, mappings or `ScriptedCase`s, with optional `wrong_sql` and
        `variants`. A question learned again replaces what was known for it."""
        self._script.learn(cases)

    def all_errors(self) -> list[BaseException]:
        """`errors` from this stub and every stub it started, in that order."""
        return [*self.errors, *(error for other in self._started for error in other.all_errors())]

    def recover(self) -> None:
        """End a scripted failure: every later request is answered."""
        with self._lock:
            self._recovered = True

    def stop(self) -> None:
        for other in self._started:
            other.stop()
        if self._stopped:
            return
        self._stopped = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    # --- the scripted replies ---------------------------------------------------------------------------------------

    def answer(self, request: dict) -> str:
        """The scripted reply to a chat request, as whoever sent it."""
        system, user = _messages(request)
        if _WRITER in system:
            return self._write(request, user)
        if system == judge._SYSTEM:
            return json.dumps({"confidence": self.confidence})
        if system == select._SYSTEM:
            return json.dumps({"choice": 0})
        if system == correct._SYSTEM:
            return _sql_to_correct(user)
        if system.startswith(synthesize._PERSONA):
            return "The rows shown answer the question."
        if system == notes_system_prompt():
            return _column_notes(user)
        return "{}"

    def _write(self, request: dict, user: str) -> str:
        if self.sql is not None:
            return _proposal(self.sql)
        case = self._script.find(user)
        if case is None:
            return _proposal(None, "no scripted answer for that question")
        with self._lock:
            asked = self._asked.get(case.question, 0)
            self._asked[case.question] = asked + 1
        missed = case.answerable and self._script.answerable_rank(case) < self.start_misses
        if not missed and self.gold_when is not None and self.gold_when(request):
            if not case.answerable:
                return _proposal(None)
            return _proposal(self._spelled(case.gold_sql, case.variants, asked))
        if case.wrong_sql is None:
            return _proposal(None, "no wrong answer is scripted for that question")
        return _proposal(self._spelled(case.wrong_sql, case.wrong_variants, asked))

    def _spelled(self, sql: str, variants: tuple[str, ...], asked: int) -> str:
        """The `asked`-th spelling of an answer under `varies`, else the answer as scripted."""
        spellings = (sql, *variants) if self.varies else (sql,)
        return spellings[asked % len(spellings)]

    # --- the doors --------------------------------------------------------------------------------------------------

    def _chat(self, handler: _Handler, body: dict) -> None:
        with self._lock:
            self.calls += 1
            n = self.calls
            self.requests.append(body)
            self._in_flight += 1
        try:
            try:
                if self.on_call is not None:
                    self.on_call(n)
            except Exception as exc:  # noqa: BLE001 -- kept for `stub_model` to fail the test
                self.errors.append(exc)
            if self.seconds:
                time.sleep(self.seconds)
            with self._lock:
                failing = not self._recovered and (
                    n <= self.fail_first or (self.fail_after is not None and n > self.fail_after))
            if failing:
                handler.fail(self.status)
                return
            try:
                reply = self.rule(body) if self.rule is not None else None
                if reply is None:
                    reply = self.answer(body)
            except Exception as exc:  # noqa: BLE001 -- kept for `stub_model` to fail the test
                self.errors.append(exc)
                handler.fail(500)
                return
            if isinstance(reply, bool) or not isinstance(reply, (str, int)):
                self.errors.append(TypeError(f"a rule returned {reply!r}, not text, a status or None"))
                handler.fail(500)
            elif isinstance(reply, int):
                handler.fail(reply)
            else:
                handler.send_json(200, self._completion(body, reply, n))
        finally:
            with self._lock:
                self._in_flight -= 1

    def _completion(self, body: dict, reply: str, n: int) -> dict:
        system, user = _messages(body)
        prompt, completion = _tokens(len(system) + len(user)), max(1, len(reply) // 4)
        return {
            "id": f"chatcmpl-stub-{n}", "object": "chat.completion", "created": 0,
            "model": body.get("model") or self.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
        }

    def _embeddings(self, body: dict) -> dict:
        texts = body.get("input")
        texts = [texts] if isinstance(texts, str) else list(texts or [])
        with self._lock:
            self.embedding_calls += 1
        return {"object": "list", "model": body.get("model") or self.model,
                "data": [{"object": "embedding", "index": i, "embedding": embedding(str(text))}
                         for i, text in enumerate(texts)],
                "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)}}

    def _tokenize(self, body: dict) -> dict:
        system, user = _messages(body)
        with self._lock:
            self.tokenize_calls += 1
        return {"count": _tokens(len(system) + len(user) + len(str(body.get("prompt") or ""))),
                "max_model_len": self.window, "tokens": []}

    def _metrics(self) -> str:
        running = self.running if self.running is not None else self._in_flight
        return ("# HELP vllm:num_requests_running Number of requests currently running.\n"
                "# TYPE vllm:num_requests_running gauge\n"
                f'vllm:num_requests_running{{model_name="{self.model}"}} {float(running)}\n')

    def _models(self) -> dict:
        return {"object": "list",
                "data": [{"id": self.model, "object": "model", "owned_by": "stub", "max_model_len": self.window}]}


class _Handler(BaseHTTPRequestHandler):
    server: ThreadingHTTPServer

    def log_message(self, format, *args) -> None:
        pass  # one line per request would bury a failing test's output

    def do_GET(self) -> None:
        stub: StubModel = self.server.stub
        path = self.path.split("?", 1)[0]
        if path == "/metrics":
            self.send_body(200, stub._metrics().encode(), "text/plain; version=0.0.4")
        elif path in ("/v1/models", "/models"):
            self.send_json(200, stub._models())
        else:
            self.send_json(404, {"error": {"message": f"no route {path}", "type": "not_found", "code": "404"}})

    def do_POST(self) -> None:
        stub: StubModel = self.server.stub
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self.send_json(400, {"error": {"message": "the body is not JSON", "type": "invalid_request_error"}})
            return
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            stub._chat(self, body)
        elif path == "/v1/embeddings":
            self.send_json(200, stub._embeddings(body))
        elif path == "/tokenize":
            self.send_json(200, stub._tokenize(body))
        else:
            self.send_json(404, {"error": {"message": f"no route {path}", "type": "not_found", "code": "404"}})

    def send_body(self, status: int, data: bytes, content_type: str, *, retry: bool = True) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if not retry:
            self.send_header("x-should-retry", "false")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, status: int, payload: dict, *, retry: bool = True) -> None:
        self.send_body(status, json.dumps(payload).encode(), "application/json", retry=retry)

    def fail(self, status: int | str) -> None:
        if status == "connect":
            self.close_connection = True  # nothing is written: the client sees the socket close unanswered
            return
        if status == "context_too_long":
            error = {"message": CONTEXT_TOO_LONG, "type": "invalid_request_error", "param": "messages",
                     "code": "context_length_exceeded"}
            self.send_json(400, {"error": error}, retry=False)
            return
        error = {"message": f"scripted failure {status}", "type": "server_error", "param": None, "code": str(status)}
        self.send_json(int(status), {"error": error}, retry=False)
