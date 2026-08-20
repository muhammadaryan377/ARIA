# ARIA Schema Agent

Fully automated — connects to your PostgreSQL database and produces `schema_mapping.json` with
tables, columns, primary keys, declared foreign keys, null/data-quality stats, and **inferred**
relationships (for foreign keys that aren't formally declared as constraints — common in messy,
real-world databases).

## 1. Install dependencies

```
pip install psycopg2-binary python-dotenv
```

## 2. Configure your database connection

Copy `.env.example` to `.env` and fill in your real values:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=postgres
DB_PASSWORD=your_password
```

## 3. Run it

```
python schema_agent.py
```

That's it — no manual steps, no pgAdmin, no copy-pasting query results. It will print a summary
and write `schema_mapping.json` in the same folder.

### Optional: override connection info on the command line instead of .env

```
python schema_agent.py --host localhost --port 5432 --db mydb --user postgres --password mypass
```

### Optional: point it at a different schema (default is "public")

```
python schema_agent.py --schema sales
```

## What it does

1. Reads every table/column/data type/nullability from `information_schema.columns`
2. Reads declared primary keys from `information_schema.table_constraints`
3. Reads declared foreign keys the same way
4. Computes null counts and null % for every column (real row counts, no guessing)
5. **Heuristic inference step**: for any column that looks like a foreign key (ends in `id`)
   but has *no* declared FK constraint, checks whether its name matches another table's primary
   key column name (e.g. `store_id` in `sales` matching `stores.store_id`). If so, it's added
   to `inferred_relationships` with a `"confidence": "heuristic-name-match"` tag, so you can
   always tell a real declared relationship apart from a guessed one.
6. Assembles everything into `schema_mapping.json`

## Tested against

This script was tested end-to-end against a live PostgreSQL 16 database containing a mix of:
- Properly declared foreign keys (`sales.cust_id -> customers.cust_id`)
- Deliberately undeclared "messy" relationships (`sales.store_id`, `inventory.prod_id`,
  `inventory.store_id` — no FK constraints, only matching column names)

The script correctly extracted both declared relationships in `declared_relationships` and
correctly guessed all three undeclared ones in `inferred_relationships` — confirming it works on
both clean and messy schemas.

## Re-running

Safe to re-run any time (e.g. after a schema change) — it simply overwrites `schema_mapping.json`
with a fresh extraction.
