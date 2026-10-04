"""Each package imports on its own, first, in a fresh interpreter (M129).

`mnemiq.contract` and `mnemiq.ontology` imported each other, so `import mnemiq.ontology` alone raised
ImportError -- and nothing in the suite could see it, because by the time a test runs some earlier
import has already loaded `mnemiq.contract` in the right order. A subprocess starts with nothing.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

FIRST_IMPORTS = ["mnemiq.ontology", "mnemiq.ontology.records", "mnemiq.ontology.binder",
                 "mnemiq.contract", "mnemiq.contract.records", "mnemiq.semantic.fit",
                 "mnemiq.enrichment.certified"]


@pytest.mark.parametrize("module", FIRST_IMPORTS)
def test_it_imports_first_in_a_fresh_interpreter(module):
    run = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-600:]


def test_the_ontology_names_are_the_contracts_own_classes():
    """Re-exported, not copied: a scheme built through either name is the same type."""
    run = subprocess.run([sys.executable, "-c",
                          "import mnemiq.ontology.records as o, mnemiq.contract.concepts as c; "
                          "assert o.ConceptScheme is c.ConceptScheme and o.Concept is c.Concept"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-600:]
