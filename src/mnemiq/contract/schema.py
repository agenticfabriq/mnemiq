import json

from .seams import IdentityContext, Trace
from .semantic import Snapshot

# Semver of the record FORMAT (the contract's shape), distinct from a record's content hash.
# Bump minor for additive changes (new optional field / new object_type), major for breaking
# ones. Regenerate schemas/mnemiq-contract.schema.json on any bump.
FORMAT_VERSION = "1.5.0"


def export_json_schema() -> dict:
    # Function-level import of OntologyRecords: ontology/records.py imports from contract, so
    # importing it at module top would be circular. (contract/records.py no longer would -- its
    # ConceptScheme now lives in contract/concepts.py, M129 -- but it stays beside its sibling.)
    from mnemiq.contract.records import CertifiedRecord
    from mnemiq.ontology.records import OntologyRecords

    return {
        "format_version": FORMAT_VERSION,
        "snapshot": Snapshot.model_json_schema(),
        "trace": Trace.model_json_schema(),
        "identity_context": IdentityContext.model_json_schema(),
        "ontology_records": OntologyRecords.model_json_schema(),
        "certified_record": CertifiedRecord.model_json_schema(),
    }


def write_schema(path: str) -> None:
    with open(path, "w") as fh:
        json.dump(export_json_schema(), fh, indent=2, sort_keys=True)
        fh.write("\n")
