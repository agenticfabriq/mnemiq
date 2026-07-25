FIXTURES = "tests/fixtures/ontology"


def test_digest_reads_a_skos_concept_scheme():
    from mnemiq.ontology.digest import digest_ontology

    records = digest_ontology([f"{FIXTURES}/skos_scheme.ttl"])
    scheme = next(s for s in records.schemes if s.label == "Colour Codes")
    by_notation = {c.notation: c for c in scheme.concepts}

    # the un-notated concept cannot ground a code, so it is not a member of the scheme
    assert set(by_notation) == {"R", "G", "B", "Y", "P"}
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


def test_definitions_are_nonpublic_by_default_and_public_when_declared():
    from mnemiq.ontology.digest import digest_ontology

    # Fail-closed: without --public the produced definitions are not globally visible.
    default = digest_ontology([f"{FIXTURES}/skos_scheme.ttl"])
    assert default.definitions and all(d.public is False for d in default.definitions)

    declared = digest_ontology([f"{FIXTURES}/skos_scheme.ttl"], public=True)
    assert declared.definitions and all(d.public is True for d in declared.definitions)
