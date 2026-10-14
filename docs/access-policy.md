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

**A filter applies to its own table and nothing else.** Nothing in a policy says that a payment belongs to a customer, so filtering `customer` leaves `payment` readable in full. At startup, mnemiq checks each declared role on its own and logs each granted table, and each granted view, whose rows hang off a filtered table through foreign keys but carry no filter of their own. With a control-plane connection configured (`MNEMIQ_CONTROL_DSN`), it checks again when the snapshot changes. Most of what it names is a hole the policy author meant to close; a view the deciders refuse anyway can be named too. A gap that appears only when one identity holds two roles together is not logged: one role granting `payment` unfiltered, another filtering `customer`.

**Filters intersect across a join.** A query joining several filtered tables sees only rows that pass every filter. The engine applies exactly what the policy says, and the result can still differ from what its author meant.

Here is a Pagila example, using the policy above:
- `rental` is filtered by the store that owns the inventory, and `payment` by the customer's store.
- So in store 1's sales, a payment is kept only when its customer belongs to store 1 as well. A store-2 customer renting store-1 inventory drops out, and in Pagila that is a large share of store 1's sales.
- If "store 1's sales" means sales of store 1's inventory, filter `payment` through the rental's inventory instead of through the customer.

Check a multi-table total before trusting it. Nothing flags an intersection as wrong, because it is what the policy says.

**A view over a filtered table is refused.** mnemiq cannot apply a table's filter through a view's own SQL, so a query reading such a view is refused. The refusal names only the view. When the view reads exactly one filtered table, its SQL names that table as the policy does (case aside), and the caller holds it, the caller is told to query it directly.

**A filter naming nothing in the source refuses every read.** A key must name a table or view in mnemiq's snapshot of the source, so a real table whose profiling failed counts as unknown too; a misspelt key would otherwise drop its filter silently. An identity's roles merge into one policy, so the refusal covers everything that identity reads, including what its other roles grant. The key is reported at startup as well.
