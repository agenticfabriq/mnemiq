FIXTURES = "tests/fixtures/ontology"


def test_digest_reads_a_skos_concept_scheme():
    from mnemiq.ontology.digest import digest_ontology

    records = digest_ontology([f"{FIXTURES}/skos_scheme.ttl"])
    scheme = next(s for s in records.schemes if s.label == "Colour Codes")
    by_notation = {c.notation: c for c in scheme.concepts}

    assert set(by_notation) == {"R", "G", "B"}  # the un-notated concept cannot ground a code
    assert by_notation["R"].pref_label == "Red"
    assert by_notation["R"].alt_labels == ["Crimson"]
    assert by_notation["R"].definition == "The colour red."
    assert records.version  # content-addressed, non-empty


def test_digest_falls_back_to_owl_subclass_grouping():
    from mnemiq.ontology.digest import digest_ontology

    records = digest_ontology([f"{FIXTURES}/owl_fallback.ttl"])
    scheme = next(s for s in records.schemes if s.label == "Shape Codes")
    assert {c.notation for c in scheme.concepts} == {"CIR", "SQR"}


def test_digest_emits_one_definition_per_scheme_plus_non_member_classes():
    from mnemiq.ontology.digest import digest_ontology

    records = digest_ontology([f"{FIXTURES}/skos_scheme.ttl", f"{FIXTURES}/owl_fallback.ttl"])
    terms = {d.term for d in records.definitions}

    assert "Colour Codes" in terms   # the scheme itself
    assert "Sample Batch" in terms   # a class with a definition, member of no scheme
    assert "Red" not in terms        # scheme MEMBERS never become Definitions
    assert all(d.bound_objects == [] for d in records.definitions)  # binding happens later
