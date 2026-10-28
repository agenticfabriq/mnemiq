# Access policy

mnemiq scopes everything to the asking identity: the tables retrieval may show, the columns a card
names, and the SQL the deciders approve. The policy is a JSON file named by `MNEMIQ_AUTHZ_PATH`.
If it cannot be read, every role is denied, and the error is logged.

## The file

```json
{
  "roles": {
    "analyst": ["customer", "product", "order_line"],
    "store1_manager": {
      "read": ["store", "staff", "inventory", "rental", "payment", "customer"],
      "pii_clearance": [],
      "pii_mask": ["pii"],
      "row_filters": {
        "store": "store_id = 1",
        "staff": "store_id = 1",
        "inventory": "store_id = 1",
        "customer": "store_id = 1",
        "rental": "inventory_id IN (SELECT inventory_id FROM inventory WHERE store_id = 1)",
        "payment": "customer_id IN (SELECT customer_id FROM customer WHERE store_id = 1)"
      }
    }
  }
}
```

A role given as a list grants read access to those tables. A role given as an object grants these:

- `read` and `write`: the tables it may read and write. A writable table is always readable.
- Column classification:
  - A column classified `pii` or `phi` is read raw where `pii_clearance` lists every level it carries.
  - It is masked where each of its levels is in `pii_clearance` or `pii_mask`. A masked column may be selected, but not used in a filter, join, grouping, ordering or aggregate.
  - Otherwise it is denied.
- `row_filters`: the rows of a table this role may see.

An identity holding several roles may read every table any of them grants. Two roles filtering the same table combine with `OR`. A filter one role sets still applies when another role grants the same table without one.

## Row filters

**A filter is a predicate on its own table.** It may read another table only inside a subquery, as `rental` and `payment` do above. That subquery is resolved against the whole source, not against the caller's grants: an entitlements table is the usual case, and the caller is never granted one.

**A filter is written in the source's own SQL.** It is applied only to a statement decided as the engine that runs it, which is how every mnemiq entry point decides. Translated between dialects, a filter can keep a row it means to remove: DuckDB's `greatest` and `concat` skip a NULL argument, and SQLite's and MySQL's give NULL. So a statement decided in one dialect for another engine is refused on a table with a filter. That applies when you call `decide` directly with a `dialect` and `target` that differ, and its defaults (DuckDB for Postgres) are such a pair. Masks are not translated text and still apply.

**A filter applies to its own table and nothing else.** Nothing in a policy says that a payment belongs to a customer, so filtering `customer` leaves `payment` readable in full. At startup, mnemiq checks each declared role on its own and logs each granted table, and each granted view, whose rows hang off a filtered table through foreign keys but carry no filter of their own. With a control-plane connection configured (`MNEMIQ_CONTROL_DSN`), it checks again when the snapshot changes. Most of what it names is a hole the policy author meant to close; a view the deciders refuse anyway can be named too. A gap that appears only when one identity holds two roles together is not logged: one role granting `payment` unfiltered, another filtering `customer`.

**Filters intersect across a join.** A query joining several filtered tables sees only rows that pass every filter. The engine applies exactly what the policy says, and the result can still differ from what its author meant.

Here is a Pagila example, using the policy above:
- `rental` is filtered by the store that owns the inventory, and `payment` by the customer's store.
- So in store 1's sales, a payment is kept only when its customer belongs to store 1 as well. A store-2 customer renting store-1 inventory drops out, and in Pagila that is a large share of store 1's sales.
- If "store 1's sales" means sales of store 1's inventory, filter `payment` through the rental's inventory instead of through the customer.

Check a multi-table total before trusting it. Nothing flags an intersection as wrong, because it is what the policy says.

**A view over a filtered table is refused.** mnemiq cannot apply a table's filter through a view's own SQL, so a query reading such a view is refused. The refusal names only the view. When the view reads exactly one filtered table, its SQL names that table as the policy does (case aside), and the caller holds it, the caller is told to query it directly.

**A filter naming nothing in the source refuses every read.** A key must name a table or view in mnemiq's snapshot of the source, so a real table whose profiling failed counts as unknown too; a misspelt key would otherwise drop its filter silently. An identity's roles merge into one policy, so the refusal covers everything that identity reads, including what its other roles grant. The key is reported at startup as well.

**A filter whose conversion can fail is refused.** A filter runs over every row of its table, including the rows it removes. So a conversion that fails on one hidden row fails every read of the table, and the database's error can quote that row's value. mnemiq refuses a filter holding any of these:

- an explicit conversion of a column: `CAST(region AS INTEGER)`, `TO_NUMBER(region)`, `TO_DATE(...)`. A cast is kept where the snapshot's column types show every value converts, such as an INTEGER cast to BIGINT. TO_NUMBER without its DEFAULT clause, and any TO_DATE, over a column are always refused;
- a DuckDB JSON function reading text, unless the snapshot types the column as JSON or the text comes from `TRY_CAST(value AS JSON)`, directly or through parentheses, `COALESCE`, `CASE` or `IF` whose every branch is one of those, NULL or a literal that is JSON (a constant on a branch is read only on the rows that take it, so it must be JSON itself); and `json_transform_strict` or `from_json_strict` over any column, since they fail on a well-formed value that does not fit the structure asked for, where `json_transform` and `from_json` give NULL;
- a comparison, or `NULLIF`, that converts a column the snapshot types as text to another type: a number, a date, a boolean, an interval, a list, a struct or a row, a scalar subquery of one SELECT whose one projection is a typed column, a constant, MIN, MAX, COUNT, SUM or AVG of one, or a `COALESCE`, `CASE`, `IF` or `NULLIF` of those, where a NULL or a text literal among the branches takes the others' type: `region = 7`, `region = (SELECT id FROM entitlement)`, `region = CASE WHEN id > 0 THEN 7 END`, in parentheses or not. A projection of any other shape, such as a set operation or a function the snapshot cannot type, is unknown and not refused;
- on DuckDB, a `COALESCE`, `CASE` or `IF` whose branches a row chooses between, where a branch is JSON and another is text or of a type mnemiq cannot tell (every other branch is converted to JSON on the rows that take it), or where a text literal stands beside a branch of another type and does not convert to the type DuckDB gives the branches: `CASE WHEN id = 1 THEN 7 ELSE 'north' END = 7` fails on every row that takes `'north'`. The branches' type is asked of DuckDB without running anything, for a column, a literal, a cast, arithmetic, a list, a struct, a nested branched value or a scalar subquery projecting one of those; a text literal beside a branch of any other shape, a function call or an aggregate, is refused as well, since the type it would be converted to cannot be told. Write such a literal as a cast, `CAST('7' AS DOUBLE)`, which is a typed branch and not a literal; a text column or cast among the branches keeps a literal as text. A text literal beside a list, struct or map branch is refused whatever it holds, since DuckDB's `TRY_CAST` to a container cannot say whether its elements convert; write it as a cast too. The same reading applies inside a list, a struct or a map (a UNION beside JSON is refused wherever it sits: DuckDB unifies two UNION types by the operator and the operand order, COALESCE and a list keeping the first operand's type and CASE and IF the last's, so the operator that keeps the text member runs and is refused all the same), on a branch, on either side of a comparison, or among the elements of one list (spelled `[...]`, `array_value`, `list_value` or `list_pack`), the keys or values of one map, or a list's element and the value a list search looks for (`list_contains`, `array_position`, `list_has_any`, `list_has_all`, `<@` and `@>`, and their spellings, and `contains` where its first argument is a list the snapshot types; a list-only search over a list the snapshot cannot type, such as one a function returns, is refused whatever it looks for, while `contains` over such a value is read as the string search it may be; and a map's key type and the key it is looked up by, `element_at`, `map_extract`, `map_extract_value`, `map_contains` or `map[key]`, a map literal subscripted at once included; a map-only lookup of a map the snapshot cannot type is refused). A container accessor that builds its value from its arguments (`map_keys`, `map_values`, `map_entries`, `map_from_entries`, `list_extract`, `list_element`, `struct_extract`, `struct_pack`, `row`, `list_slice`, `list_reverse`, `list_distinct`, `list_unique`, `list_sort`, `flatten` and their spellings) is typed as DuckDB types it, so its value is read like the container's; a value a function the snapshot does not type returns is unknown (`[attrs, region]` is a list of JSON, and region is parsed on every row); the text-literal rule applies to a branched value alone: `CASE WHEN id = 1 THEN [attrs] ELSE [region] END` and `[attrs] = [region]` convert `region` to JSON on the rows that take it, where `[id]` beside `[attrs]` is serialised; write `[TRY_CAST(region AS JSON)]`.

Write the form that gives NULL instead of failing. On DuckDB that is `TRY_CAST(value AS type)`, a JSON function reads from `TRY_CAST(value AS JSON)`, and `json_transform` or `from_json` stands in for its strict form. On Oracle it is `CAST(value AS type DEFAULT NULL ON CONVERSION ERROR)` with no format argument, or `TO_NUMBER(value DEFAULT NULL ON CONVERSION ERROR)`.

The check matches known spellings; it is not a proof. It does not catch an expression over a text column, such as `UPPER(region) > 0` or, on Oracle, `NVL(id, TRIM(region))`; a comparison with a column the snapshot records no type for; or an expression that fails on data without converting anything. Such a filter still fails every read when a row trips it, and the model sees only the error's class, never its text.
