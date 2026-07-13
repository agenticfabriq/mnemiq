import json

from .seams import IdentityContext, Trace
from .semantic import Snapshot


def export_json_schema() -> dict:
    return {
        "snapshot": Snapshot.model_json_schema(),
        "trace": Trace.model_json_schema(),
        "identity_context": IdentityContext.model_json_schema(),
    }


def write_schema(path: str) -> None:
    with open(path, "w") as fh:
        json.dump(export_json_schema(), fh, indent=2, sort_keys=True)
        fh.write("\n")
