# Grounding code columns

Databases are full of short codes — `R`, `E11`, `NC-17` — whose meaning lives outside the row.
mnemiq's guiding rule is **grounded-or-bare**: it will tell the model what a code means *only* when
it has evidence, and otherwise leaves the raw code untouched. It never guesses, because a guessed
meaning produces wrong SQL that nothing downstream can catch.

This page covers the two ways you can supply that evidence — an **ontology** (a standard code
system in TTL/SKOS/OWL) and an **operator dictionary** (a hand-written JSON file) — and how they
combine with what mnemiq already learns from the data itself.

## The precedence chain

`mnemiq enrich` fills each code's meaning from up to four sources. When more than one applies to the
same code, the later one wins:

```
ontology  <  correlated  <  lookup  <  dictionary
```

- **ontology** — a standard code system you supply (this page). Lowest precedence: it *fills* codes
  that nothing else explained, and never overrides a meaning the database itself provided.
- **correlated** — a sibling label column in the same table that the code functionally determines
  (e.g. `status_code` → `status_name`). Learned from data, no configuration.
- **lookup** — a foreign-key dimension table's label column (e.g. `film.language_id` →
  `language.name`). Learned from data, no configuration.
- **dictionary** — your operator dictionary (below). Highest precedence: an explicit human
  statement overrides everything.

The design principle: an in-database label describes *this* database, while an external standard
describes the world, so the database wins wherever it speaks. A human operator overrides both.

Every grounded meaning records where it came from, so a card can show `R = Restricted` and the
provenance is auditable.

---

## Ontology grounding

An ontology is a curated, offline vocabulary — ICD-10 diagnosis codes, NAICS industry codes, a USDA
food taxonomy. mnemiq consumes it in two steps: an offline **digest** turns the source TTL into a
portable records file, and `enrich` then **binds** each scheme to the columns that draw from it.

### 1. Install the extra

The digest needs `rdflib`, which is an optional dependency (nothing on the query path uses it):

```
uv sync --extra ontology
```

### 2. Digest the ontology into records

```
mnemiq digest-ontology --ttl path/to/vocab.ttl --out records.json
```

`--ttl` accepts a single `.ttl` file or a directory (searched recursively), and is repeatable.
The command reads only standard vocabularies — it is namespace-agnostic, so any publisher's TTL
works — and recognises two shapes:

- a `skos:ConceptScheme` whose members declare `skos:inScheme`, and
- (fallback) an `owl:Class` whose `rdfs:subClassOf` children carry a `skos:notation`.

For each concept it extracts:

| From the TTL | Becomes |
|---|---|
| `skos:notation` | the **code as stored** — the only thing matched against column values |
| `skos:prefLabel` ‖ `rdfs:label` | the meaning shown to the model |
| `skos:altLabel` | synonyms, used for question-time matching |
| `skos:definition` ‖ `rdfs:comment` | a definition |
| `skos:broader` ‖ `rdfs:subClassOf` | the parent concept |

A concept with no `skos:notation` cannot ground a code (there is nothing to match against) and is
dropped from the scheme. The output `records.json` is a portable artifact; you can inspect or edit
it by hand.

### 3. Enrich with the records

```
export MNEMIQ_ONTOLOGY_RECORDS_PATH=records.json
mnemiq enrich
mnemiq build
```

During `enrich`, mnemiq binds each scheme to the columns it belongs to (below), fills any bare codes
those columns harvested, and — for large code systems — builds a concept index that resolves a
question's wording to candidate codes at ask time. `mnemiq build` is unchanged; the index is
persisted with the snapshot, so `ask` needs only the store, not the records file.

### How auto-binding decides (precision-first)

A scheme binds to a column only when **all three** gates pass. When in doubt it binds nothing —
leaving the column bare is always safe; a wrong binding is not.

1. **Eligibility** — the column is not a key or a sensitive/personal column.
2. **Value containment** — the column's observed values are (almost entirely) valid notations of
   the scheme. Sentinel junk (`N/A`, `UNKNOWN`, `NULL`, `-1`, `?`, empty) is forgiven; a genuine
   foreign code is not. A column needs at least a handful of distinct real codes to qualify — a
   two-value column matching by chance is not evidence.
3. **Name affinity** — the column *name* resembles the scheme's label. Containment proves the
   values fit; affinity is what makes the fit mean something. Without it, a scheme whose notations
   are `1/2/3` would "contain" half your database.

If two schemes pass for the same column, mnemiq binds neither.

### Explicit binding (when the name doesn't match)

Some columns hold standard codes under a name no heuristic will connect — `dx_cd` for ICD-10,
`mean` for a residue-determination code. For these, assert the binding yourself in the records file:

```json
{
  "version": "...",
  "schemes": [ ... ],
  "bindings": {
    "resultsdata.dx_cd": "http://example.org/vocab#Icd10",
    "resultsdata.mean":  "http://example.org/vocab#ResidueDetermination"
  }
}
```

A key in `bindings` is a **column id** (`table.column`); the value is a **scheme id** (the scheme's
IRI, as it appears under `schemes[].id` in the same file). An explicit binding is taken on your word
— it bypasses all three gates. This is the escape hatch that carries any column auto-binding can't
reach.

### What the model sees

- **Small coded columns** (a handful of distinct values) get their meanings rendered on the card:
  `Values: G = General Audiences; PG = Parental Guidance Suggested; … Codes from MPAA Film Rating.`
- **Large code systems** (hundreds or thousands of codes) are *not* dumped onto the card. Instead,
  when a question's wording matches concept labels, the relevant codes are surfaced as candidates:

  ```
  CODE VOCABULARY (candidate codes for this question — filter on the CODE, never the label):
  - patient.icd10_cd uses ICD-10-CM. Candidates: E11 = Type 2 diabetes mellitus; E10 = Type 1 …
  ```

  Matching is deterministic (no model, no network), and the model is instructed to filter on the
  raw code, never the label.

---

## Operator dictionary

When you don't have an ontology — or want to override what the database and ontology inferred — a
data dictionary is a hand-written JSON file mapping columns to descriptions and code meanings. It is
the **highest-precedence** source.

```json
{
  "columns": {
    "orders.status": {
      "description": "Order fulfilment status.",
      "codes": { "N": "New", "S": "Shipped", "X": "Cancelled" }
    }
  }
}
```

```
export MNEMIQ_DICTIONARY_PATH=dictionary.json
mnemiq enrich
```

Only codes the column actually has are grounded; a dictionary entry for a code not present in the
data is ignored, and a meaning is never invented for a code you didn't list. The dictionary and an
ontology can be used together — the dictionary wins on any code they both address.

---

## Refreshing

Grounding is computed at `enrich` time and cached with the snapshot. Editing the ontology records or
the dictionary and re-running `enrich` re-grounds from scratch. Under the eval/benchmark cache, the
records and dictionary paths are folded into the snapshot cache key, so switching files never reuses
a stale snapshot; editing a file in place still needs `--refresh` (as grounding always does).
