# Running mnemiq on BEAVER

[BEAVER](https://beaverbench.github.io) is MIT's benchmark of enterprise text-to-SQL questions over
real data-warehouse schemas ([paper](https://arxiv.org/abs/2409.02038)). `scripts/run_beaver.py`
runs mnemiq on it and writes each question's SQL, so anyone with the dataset can measure the same
setup we did, on the public questions or on a held-back set.

The dataset is gated on Hugging Face and none of it is in this repository: you need
`beaverbench/beaver-query` (the questions) and `beaverbench/beaver-table` (the MySQL dump,
`beaver_db.zip`).

## 1. A MySQL server for BEAVER's databases

mnemiq answers on BEAVER's own engine, MySQL, because BEAVER's gold SQL depends on MySQL's type
rules: many numbers are stored as text and compared through MySQL's implicit casts. The server
needs two settings, and the runner refuses to start without them:

- **`lower_case_table_names=1`**, because the gold SQL spells table names in a different case than
  the dump creates them. MySQL reads this setting only when the data directory is first created,
  so set it on the first start; changing it later means a new data directory.
- **`ANSI_QUOTES` in `sql_mode`**, because mnemiq quotes identifiers with double quotes. No BEAVER
  gold query uses a double quote, so this changes nothing for the gold.

With Docker (MySQL 8.4, as we ran it):

```bash
docker run -d --name beaver-mysql -p 127.0.0.1:3307:3306 \
  -e MYSQL_ROOT_PASSWORD="$BEAVER_ROOT_PASSWORD" -v beaver-mysql-data:/var/lib/mysql \
  mysql:8.4 --lower-case-table-names=1 --max_allowed_packet=1G --innodb_buffer_pool_size=4G \
  --sql-mode=ANSI_QUOTES,ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION
```

That `sql_mode` is MySQL 8.4's default with `ANSI_QUOTES` added. Load the three dumps from
`beaver_db.zip` (`dw.sql`, `nova.sql`, `neutron.sql`; each creates its own database), then give
mnemiq a user that can only read them:

```sql
CREATE USER 'beaver'@'%' IDENTIFIED BY '...';
GRANT SELECT ON dw.* TO 'beaver'@'%';
GRANT SELECT ON nova.* TO 'beaver'@'%';
GRANT SELECT ON neutron.* TO 'beaver'@'%';
```

The runner reaches MySQL through DuckDB's `mysql` extension, which it installs on first use, so
there is no MySQL driver to add.

## 2. Configure

```bash
export MNEMIQ_BEAVER_DIR=/path/to/beaver        # holds query/data/<set>-00000-of-00001.parquet
export MNEMIQ_MYSQL_DSN="host=127.0.0.1 port=3307 user=beaver password=..."   # no database
export MNEMIQ_LLM_BASE_URL=... MNEMIQ_LLM_API_KEY=... MNEMIQ_LLM_MODEL=...
export MNEMIQ_EMBED_BASE_URL=... MNEMIQ_EMBED_MODEL=...
```

## 3. Prove the grading path, then run

```bash
uv sync --extra dev
uv run python scripts/run_beaver.py --control
```

`--control` answers every question with its own gold SQL, so every row it grades must come out
correct before any model is asked. It needs no model. A row whose gold fails or returns more than
1,000 rows (the limit mnemiq puts on every answer) is not graded and is reported as such.

Then the three configurations we measured:

```bash
uv run python scripts/run_beaver.py --annotations none       --out eval-reports/beaver-none.jsonl
uv run python scripts/run_beaver.py --annotations all        --out eval-reports/beaver-all.jsonl
uv run python scripts/run_beaver.py --annotations no-sub-sql --out eval-reports/beaver-no-sub-sql.jsonl
```

- `none` asks the question alone, with every table in the database available.
- `all` appends BEAVER's annotations for the question (its tables, join keys, column mapping,
  domain knowledge, and sub-questions with their SQL, as the paper's Setting 2 gives them) and
  grants the question only the tables its annotation names.
- `no-sub-sql` is `all` without the sub-questions and their SQL.

`--sets` picks sets (default `dw dw_real nova neutron`), `--limit N` the first N questions of each,
and `--ids FILE` the questions listed one id per line. For a set without gold, add `--no-grade`:
the runner then records each question's SQL and grades nothing.

A results file holds one JSON line per question: `id`, `set`, `db`, `sql` (null when mnemiq
declined), `reason_code` (why it declined), `answer`, `model`, `engine` (the mnemiq commit,
marked `-dirty` for uncommitted changes), `settings` (every `MNEMIQ_*` setting that differs from
its default, endpoints and credentials left out), `data` (which copy of the set's file it was
asked from), the run's `annotations`, and, when graded, `outcome` with the row counts. A re-run
with the same `--out` skips the questions already in it, asks again one whose line a killed run
cut short, and refuses a file written under a different configuration, settings or copy of a set.

## The setup behind our BEAVER figures

- **Model:** GPT-5.5, through an OpenAI-compatible endpoint.
- **Embeddings:** bge-m3, served by Ollama.
- **Settings:** product defaults, except the answer verifier and its judge on (`MNEMIQ_VERIFY=1`,
  `MNEMIQ_VERIFY_JUDGE=1`) and guided SQL decoding off (`MNEMIQ_GUIDED_SQL=0`).
- **Enrichment:** structural only (catalog, keys and profiling, no model-written descriptions),
  which is what the runner does.
- **Questions:** 200 public questions, 60 `dw`, 40 `dw_real`, 50 `nova` and 50 `neutron`. The gold
  control graded all 200 correct.
- **Engine:** commits `598def4` (no annotations, all annotations) and `360a227` (all but the
  sub-questions), through an earlier internal version of this runner. The engine has changed
  since, so a run on a later commit can differ.
