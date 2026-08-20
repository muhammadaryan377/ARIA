"""
ARIA Schema Agent
-----------------
Role: Data Understanding

Runs once (or on schema change) to parse and map messy/undocumented relational
databases and semi-structured (JSON/CSV/PDF) sources. It identifies tables,
columns, data types, null patterns, primary keys, declared foreign keys, and
INFERRED relationships for columns that look like foreign keys but have no
formal constraint declared.

Semi-structured sources (JSON/CSV/PDF) are parsed with the Qwen3-4B model via
Ollama, which extracts entities/fields and maps them into the unified schema.
Qwen3-4B is used exclusively here — never for SQL generation or storytelling.

Output: schema_mapping.json  (single source of truth for the Goal Agent)

Usage:
    python schema_agent.py
    python schema_agent.py --host localhost --port 5432 --db northwind --user postgres --password 12345
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras
from psycopg2 import sql

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

from llm_provider import LLMProvider, create_provider
from core.config import SCHEMA_DIR

logger = logging.getLogger("aria.schema_agent")


# Read-only introspection must never hang indefinitely on one huge table:
# bound every statement on the session (30s) so a slow table degrades
# gracefully instead of stalling the whole run.
QUERY_TIMEOUT_MS = 30_000


def _apply_session_timeout(conn, db_type="postgresql"):
    """Set a session-level statement timeout for read-only introspection.

    PostgreSQL: `statement_timeout` applies to every statement.
    MySQL: `MAX_EXECUTION_TIME` applies to SELECTs (MySQL 5.7.8+). If the
    server is older and rejects it, the timeout is skipped (query still runs,
    just unbounded — better than failing the connection).
    """
    try:
        cur = conn.cursor()
        if db_type == "mysql":
            cur.execute(f"SET SESSION MAX_EXECUTION_TIME = {QUERY_TIMEOUT_MS}")
        else:
            cur.execute(f"SET statement_timeout = {QUERY_TIMEOUT_MS}")
        conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Connection handling
# ---------------------------------------------------------------------------

def get_connection(args):
    """Build a database connection (PostgreSQL or MySQL) from CLI args / env vars."""
    db_type = getattr(args, "db_type", None) or "postgresql"
    host = args.host or os.getenv("DB_HOST", "localhost")
    # The port must never leak across engine types: DB_PORT in .env targets
    # PostgreSQL, so MySQL must not inherit it. Only --port (or MYSQL_PORT) can
    # override MySQL's default.
    if db_type == "mysql":
        port = args.port or os.getenv("MYSQL_PORT", "3306")
    else:
        port = args.port or os.getenv("DB_PORT", "5432")
    dbname = args.db or os.getenv("DB_NAME")
    user = args.user or os.getenv("DB_USER")
    password = args.password or os.getenv("DB_PASSWORD")

    missing = [name for name, val in
               [("database", dbname), ("user", user), ("password", password)]
               if not val]
    if missing:
        sys.exit(
            f"ERROR: missing required connection info: {', '.join(missing)}.\n"
            f"Set them via .env or pass --db --user --password on the command line."
        )

    if db_type == "mysql":
        if not MYSQL_AVAILABLE:
            sys.exit("ERROR: pymysql is not installed. Run: pip install pymysql")
        try:
            conn = pymysql.connect(
                host=host, port=int(port), user=user, password=password,
                database=dbname, charset="utf8mb4", connect_timeout=10,
            )
        except pymysql.MySQLError as e:
            sys.exit(
                f"ERROR: could not connect to MySQL at {host}:{port}/{dbname} as {user}: {e}\n"
                "Check: is the MySQL server running on that host/port? Are the "
                "database name, user, and password correct, and does the server "
                "accept TCP connections?"
            )
        _apply_session_timeout(conn, "mysql")
        return conn

    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password,
            connect_timeout=10,
        )
        # Read-only introspection: autocommit prevents one failed query from
        # leaving the connection in the "current transaction is aborted" state,
        # which would silently fail every later query on the same connection.
        conn.autocommit = True
    except psycopg2.OperationalError as e:
        sys.exit(
            f"ERROR: could not connect to PostgreSQL at {host}:{port}/{dbname} as {user}: {e}\n"
            "Check: is the PostgreSQL service running on that host/port? Are the "
            "database name, user, and password correct? Does pg_hba.conf allow "
            "TCP connections from this host?"
        )
    _apply_session_timeout(conn, "postgresql")

    return conn


def _dict_cursor(conn, db_type="postgresql"):
    """Return a dictionary-row cursor for the connected database type."""
    if db_type == "mysql":
        return conn.cursor(pymysql.cursors.DictCursor)
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def _lower_keys(row):
    """Normalize a dict row's keys to lowercase.

    PostgreSQL's information_schema returns lowercase column names, but MySQL
    returns UPPERCASE ones (TABLE_NAME, COLUMN_NAME, ...). All downstream code
    reads lowercase keys, so MySQL rows must be normalized or every
    information_schema query fails with KeyError.
    """
    if not row:
        return row
    return {str(k).lower(): v for k, v in row.items()}


def _quote_ident(db_type, name):
    """Return a SQL-safe quoted identifier (double quotes for PG, backticks for MySQL)."""
    if db_type == "mysql":
        return "`" + str(name).replace("`", "") + "`"
    return '"' + str(name).replace('"', "") + '"'


# ---------------------------------------------------------------------------
# Metadata extraction (pure SQL, no LLM)
# ---------------------------------------------------------------------------

def get_tables_and_columns(conn, schema="public", db_type="postgresql"):
    """Return {table_name: [{column, data_type, nullable}, ...]}

    Only BASE TABLEs are profiled; views are skipped (their PK/uniqueness
    checks can be slow or meaningless against a complex view query).
    """
    query = """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name IN (
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s AND table_type = 'BASE TABLE'
          )
        ORDER BY table_name, ordinal_position;
    """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema, schema))
        rows = cur.fetchall()

    tables = {}
    for row in rows:
        row = _lower_keys(row)
        tables.setdefault(row["table_name"], []).append({
            "column": row["column_name"],
            "data_type": row["data_type"],
            "nullable": row["is_nullable"] == "YES",
        })
    return tables


def get_primary_keys(conn, schema="public", db_type="postgresql"):
    """Return {table_name: [pk_column, ...]}"""
    # NOTE: the join must include tc.table_name. In MySQL every primary key
    # constraint is literally named 'PRIMARY', so joining on constraint_name
    # alone cross-joins every table's PK columns to every other table.
    query = """
        SELECT tc.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
         AND tc.table_name = kcu.table_name
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = %s
        ORDER BY tc.table_name, kcu.ordinal_position;
    """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema,))
        rows = cur.fetchall()

    pks = {}
    for row in rows:
        row = _lower_keys(row)
        pks.setdefault(row["table_name"], []).append(row["column_name"])
    return pks


def get_unique_keys(conn, schema="public", db_type="postgresql"):
    """Return {table_name: [[col, ...], ...]} — one entry per UNIQUE constraint.

    Declared UNIQUE constraints are authoritative: a column covered by one is
    already known to be unique, so it should never be re-inferred or treated as
    a duplicate/weak key downstream.
    """
    query = """
        SELECT tc.table_name, kcu.column_name, tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
         AND tc.table_name = kcu.table_name
        WHERE tc.constraint_type = 'UNIQUE'
          AND tc.table_schema = %s
        ORDER BY tc.table_name, tc.constraint_name, kcu.ordinal_position;
    """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema,))
        rows = cur.fetchall()

    uniques = {}
    for row in rows:
        row = _lower_keys(row)
        uniques.setdefault(row["table_name"], {})
        uniques[row["table_name"]].setdefault(row["constraint_name"], []).append(
            row["column_name"]
        )
    return {t: list(cols.values()) for t, cols in uniques.items()}


def get_declared_foreign_keys(conn, schema="public", db_type="postgresql"):
    """Return list of {table_name, column_name, references_table, references_column}"""
    if db_type == "mysql":
        # MySQL exposes the referenced table/column directly on key_column_usage.
        query = """
            SELECT
                kcu.table_name AS table_name,
                kcu.column_name AS column_name,
                kcu.referenced_table_name AS references_table,
                kcu.referenced_column_name AS references_column
            FROM information_schema.key_column_usage kcu
            WHERE kcu.referenced_table_name IS NOT NULL
              AND kcu.table_schema = %s
            ORDER BY kcu.table_name, kcu.ordinal_position;
        """
    else:
        # PostgreSQL: join pg_constraint directly and align conkey/confkey by
        # ordinal position. Joining information_schema.key_column_usage to
        # constraint_column_usage on the constraint name alone would cross-produce
        # composite foreign keys (a 2-column FK would yield 4 wrong pairs).
        query = """
            SELECT
                refcon.relname AS table_name,
                refatt.attname AS column_name,
                conrel.relname AS references_table,
                conatt.attname AS references_column
            FROM pg_constraint c
            JOIN pg_namespace n ON n.oid = c.connamespace
            JOIN pg_class refcon ON refcon.oid = c.conrelid
            JOIN pg_class conrel ON conrel.oid = c.confrelid
            JOIN unnest(c.conkey) WITH ORDINALITY AS srckeys(attnum, ord) ON TRUE
            JOIN unnest(c.confkey) WITH ORDINALITY AS tgtkeys(attnum, ord) ON srckeys.ord = tgtkeys.ord
            JOIN pg_attribute refatt ON refatt.attrelid = c.conrelid AND refatt.attnum = srckeys.attnum
            JOIN pg_attribute conatt ON conatt.attrelid = c.confrelid AND conatt.attnum = tgtkeys.attnum
            WHERE c.contype = 'f'
              AND n.nspname = %s
            ORDER BY refcon.relname, srckeys.ord;
        """
    with _dict_cursor(conn, db_type) as cur:
        cur.execute(query, (schema,))
        return [dict(row) for row in cur.fetchall()]


def get_null_stats(conn, tables, schema="public", db_type="postgresql"):
    """Return {table_name: {column_name: {"total_rows": n, "nulls": n, "null_pct": p}}}

    A table that cannot be counted (permission error, exotic type, huge cost)
    is skipped rather than killing the whole run.
    """
    stats = {}
    with conn.cursor() as cur:
        for table_name, columns in tables.items():
            table_stats = {}
            try:
                if db_type == "mysql":
                    qualified = f"{_quote_ident(db_type, schema)}.{_quote_ident(db_type, table_name)}"
                    cur.execute(f"SELECT COUNT(*) FROM {qualified}")
                    total_rows = cur.fetchone()[0]

                    for col in columns:
                        col_name = col["column"]
                        qcol = _quote_ident(db_type, col_name)
                        cur.execute(
                            f"SELECT COUNT(*) - COUNT({qcol}) FROM {qualified}"
                        )
                        nulls = cur.fetchone()[0]
                        null_pct = round((nulls / total_rows) * 100, 2) if total_rows else 0.0
                        table_stats[col_name] = {
                            "total_rows": total_rows,
                            "nulls": nulls,
                            "null_pct": null_pct,
                        }
                else:
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}.{}")
                           .format(sql.Identifier(schema), sql.Identifier(table_name))
                    )
                    total_rows = cur.fetchone()[0]

                    for col in columns:
                        col_name = col["column"]
                        cur.execute(
                            sql.SQL("SELECT COUNT(*) - COUNT({}) FROM {}.{}")
                               .format(sql.Identifier(col_name), sql.Identifier(schema), sql.Identifier(table_name))
                        )
                        nulls = cur.fetchone()[0]
                        null_pct = round((nulls / total_rows) * 100, 2) if total_rows else 0.0
                        table_stats[col_name] = {
                            "total_rows": total_rows,
                            "nulls": nulls,
                            "null_pct": null_pct,
                        }
            except Exception:
                # A single unreadable table must not poison the whole mapping.
                stats[table_name] = {}
                continue
            stats[table_name] = table_stats
    return stats


# ---------------------------------------------------------------------------
# Heuristic relationship inference (fills the gap for "messy" databases)
# ---------------------------------------------------------------------------

def normalize_identifier(value):
    """Normalize names to compare database identifiers across styles."""
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _singularize(name):
    """Best-effort singular of an English identifier (for PK/name matching)."""
    if name.endswith("ies") and len(name) > 4:
        return name[:-3] + "y"
    if name.endswith("s") and not name.endswith("ss") and len(name) > 2:
        return name[:-1]
    return name


# Tokens that signal "this word is about identifiers/keys", never content words.
_ID_TOKENS = {"id", "no", "num", "number", "code", "key", "fk", "ref", "uuid", "sk", "nk"}


def _tokenize_words(name):
    """Split an identifier into lowercase word tokens (snake_case, camelCase, acronyms).

    "MediaTypeId" -> ["media", "type", "id"]; "support_rep_id" -> ["support", "rep", "id"].
    This is what lets name matching work on ANY naming convention, not just X_id.
    """
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    return [w for w in re.split(r"[^a-zA-Z0-9]+", s.lower()) if w]


def _norm_value(v):
    """Normalize a sampled value for cross-column comparison (numeric-aware)."""
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    return ("str", str(v))


class _ValueProfiler:
    """Samples distinct column values so relationships can be confirmed from DATA.

    This is what makes inference work for ANY database: a column whose values
    overlap a primary key's values is a de-facto foreign key, regardless of how
    it is named (e.g. support_rep_id -> employee, reports_to -> employee).
    """

    MIN_OVERLAP = 0.85
    MAX_SAMPLE = 500
    MAX_PK_SAMPLE = 100000
    MAX_PROFILE_QUERIES = 400

    def __init__(self, conn, schema, db_type, tables, primary_keys):
        self.conn = conn
        self.schema = schema
        self.db_type = db_type
        self.tables = tables
        self._src_cache = {}
        self._pk_cache = {}
        self._rows_cache = {}
        self._dtype_cache = {}
        self._profile_budget = self.MAX_PROFILE_QUERIES
        self.pk_index = []
        for table, cols in primary_keys.items():
            for col in cols:
                self.pk_index.append(
                    (table, col, self._distinct(table, col, self._pk_cache, self.MAX_PK_SAMPLE)))

    def column_type(self, table, col):
        """Return the (normalized) data type of a column, or '' when unknown."""
        key = (table, col)
        if key not in self._dtype_cache:
            dtype = ""
            cols = self.tables.get(table)
            if isinstance(cols, list):
                for c in cols:
                    if c.get("column") == col:
                        dtype = str(c.get("data_type", "")).lower()
                        break
            elif isinstance(cols, dict):
                for c in cols.get("columns", []):
                    if c.get("column") == col:
                        dtype = str(c.get("data_type", "")).lower()
                        break
            self._dtype_cache[key] = dtype
        return self._dtype_cache[key]

    def types_compatible(self, src_table, src_col, tgt_table, tgt_col):
        """True when two columns are plausibly joinable by data type.

        Numeric-to-numeric (int/bigint/numeric/decimal/float/serial) and
        text-to-text (varchar/text/char/uuid) are compatible. Crossing numeric
        with text is a strong sign the columns are NOT a real FK even when a few
        values overlap.
        """
        st = self.column_type(src_table, src_col)
        tt = self.column_type(tgt_table, tgt_col)
        if not st or not tt:
            return True  # unknown -> don't reject
        if st == tt:
            return True
        numeric = ("int", "bigint", "smallint", "numeric", "decimal", "float",
                   "double", "real", "serial", "money")
        text = ("char", "varchar", "text", "uuid", "string")
        is_s_num = any(tok in st for tok in numeric)
        is_t_num = any(tok in tt for tok in numeric)
        is_s_txt = any(tok in st for tok in text)
        is_t_txt = any(tok in tt for tok in text)
        if is_s_num and is_t_num:
            return True
        if is_s_txt and is_t_txt:
            return True
        return False

    def _distinct(self, table, col, cache, limit):
        key = (table, col)
        if key in cache:
            return cache[key]
        if self._profile_budget <= 0:
            cache[key] = set()
            return cache[key]
        self._profile_budget -= 1
        try:
            values = set()
            with _dict_cursor(self.conn, self.db_type) as cur:
                cur.execute(
                    f"SELECT DISTINCT {_quote_ident(self.db_type, col)} AS v "
                    f"FROM {_quote_ident(self.db_type, self.schema)}.{_quote_ident(self.db_type, table)} "
                    f"WHERE {_quote_ident(self.db_type, col)} IS NOT NULL "
                    f"ORDER BY {_quote_ident(self.db_type, col)} LIMIT {limit}"
                )
                for row in cur.fetchall():
                    values.add(_norm_value(row["v"]))
        except Exception as exc:
            logger.debug("distinct sample failed for %s.%s: %s", table, col, exc)
            values = set()
        cache[key] = values
        return values

    def source_values(self, table, col):
        return self._distinct(table, col, self._src_cache, self.MAX_SAMPLE)

    def row_count(self, table):
        if table not in self._rows_cache:
            try:
                with _dict_cursor(self.conn, self.db_type) as cur:
                    cur.execute(
                        f"SELECT COUNT(*) AS n FROM {_quote_ident(self.db_type, self.schema)}.{_quote_ident(self.db_type, table)}"
                    )
                    self._rows_cache[table] = cur.fetchone()["n"] or 0
            except Exception as exc:
                logger.debug("row count failed for %s: %s", table, exc)
                self._rows_cache[table] = 0
        return self._rows_cache[table]

    @staticmethod
    def _overlap(src, target):
        if not src:
            return 0.0
        return len(src & target) / len(src)

    def strongest_overlap(self, source_set, only_tables=None, min_source=1, high_cardinality=False, row_count=None, exclude=None, require_unambiguous=False, source_table=None, source_col=None):
        """Return (table, pk_col, overlap_ratio, matched) for the best PK target.

        Ties (same ratio and matched count) are broken in favour of the target
        whose primary key set is the smallest that still contains the values —
        the tightest containment is the more likely real reference. When
        `require_unambiguous` is set and the top two targets are indistinguishable
        on (ratio, matched), no target is returned: with opaque names and no name
        evidence a wrong guess is worse than none.
        """
        cands = []
        for table, pk_col, target_set in self.pk_index:
            if only_tables and table not in only_tables:
                continue
            if exclude and (table, pk_col) == exclude:
                continue
            if len(source_set) < min_source:
                continue
            if source_table is not None and source_col is not None:
                if not self.types_compatible(source_table, source_col, table, pk_col):
                    continue
            ratio = self._overlap(source_set, target_set)
            if ratio < self.MIN_OVERLAP:
                continue
            if high_cardinality:
                if len(source_set) < 10:
                    continue
                if row_count and len(source_set) < 0.05 * row_count:
                    continue
            cands.append((ratio, len(source_set & target_set), -len(target_set),
                          table, pk_col, len(source_set & target_set)))
        if not cands:
            return None
        cands.sort(reverse=True)
        # "Unambiguous" means indistinguishable on the FULL decisive tuple
        # (ratio, matched, key size). A much tighter containment (e.g. an
        # airports.id with 500 distinct vs a flights.id with 30000) is a real
        # discriminator, so it must not be discarded as ambiguous.
        if require_unambiguous and len(cands) >= 2 and cands[0][:3] == cands[1][:3]:
            return None
        ratio, _, _, table, pk_col, matched = cands[0]
        # Item 4 (ambiguous): a relationship is ambiguous when a runner-up
        # candidate carries nearly identical data evidence (ratio within 3 pts
        # and matched within a small absolute delta). We still return the best
        # candidate, but flag it so the caller/downstream can defer to a human.
        ambiguous = False
        if len(cands) >= 2:
            r2, m2 = cands[1][0], cands[1][5]
            if (cands[0][0] - r2) < 0.03 and abs(matched - m2) <= max(2, int(0.05 * matched)):
                ambiguous = True
        return table, pk_col, ratio, matched, ambiguous

    def _distinct_raw(self, table, col, limit):
        """Raw (un-normalized) distinct values, for exact SQL containment checks."""
        values = []
        try:
            with _dict_cursor(self.conn, self.db_type) as cur:
                cur.execute(
                    f"SELECT DISTINCT {_quote_ident(self.db_type, col)} AS v "
                    f"FROM {_quote_ident(self.db_type, self.schema)}.{_quote_ident(self.db_type, table)} "
                    f"WHERE {_quote_ident(self.db_type, col)} IS NOT NULL "
                    f"ORDER BY {_quote_ident(self.db_type, col)} LIMIT {limit}"
                )
                for row in cur.fetchall():
                    v = row["v"]
                    if v is not None:
                        values.append(v)
        except Exception as exc:
            logger.debug("distinct sample failed for %s.%s: %s", table, col, exc)
            return []
        return values

    def _exact_matched(self, table, pk_col, values):
        """Count how many of `values` actually occur in the FULL target key column."""
        if not values:
            return None
        ident = _quote_ident(self.db_type, pk_col)
        try:
            with _dict_cursor(self.conn, self.db_type) as cur:
                if self.db_type == "mysql":
                    marks = ", ".join(["%s"] * len(values))
                    cur.execute(
                        f"SELECT count(DISTINCT {ident}) AS matched "
                        f"FROM {_quote_ident(self.db_type, table)} "
                        f"WHERE {ident} IN ({marks})",
                        tuple(values),
                    )
                else:
                    cur.execute(
                        f"SELECT count(DISTINCT {ident}) AS matched "
                        f"FROM {_quote_ident(self.db_type, self.schema)}.{_quote_ident(self.db_type, table)} "
                        f"WHERE {ident} = ANY(%s)",
                        (list(values),),
                    )
                return cur.fetchone()["matched"] or 0
        except Exception as exc:
            logger.debug("exact containment check failed for %s.%s: %s", table, pk_col, exc)
            try:
                self.conn.rollback()
            except Exception:
                pass
            return None

    def strongest_exact_overlap(self, source_table, source_col, target_tables, exclude=None):
        """Like strongest_overlap but verifies against the FULL target key column.

        Sampling both sides dilutes containment when the target table is huge
        (e.g. a 1M-row sales key vs a 30k-row warranty that references it),
        hiding real relationships. When the column name already points at a
        concrete target, an exact SQL containment count is cheap and accurate.
        """
        src = self._distinct_raw(source_table, source_col, self.MAX_SAMPLE)
        if len(src) < 2:
            return None
        col_tokens = set(_tokenize_words(source_col))
        best = None
        for table, pk_col, _ in self.pk_index:
            if table not in target_tables:
                continue
            if exclude and (table, pk_col) == exclude:
                continue
            if not self.types_compatible(source_table, source_col, table, pk_col):
                continue
            matched = self._exact_matched(table, pk_col, src)
            if matched is None:
                continue
            ratio = matched / len(src)
            if ratio < self.MIN_OVERLAP:
                continue
            # Tiebreaker: prefer the target whose key CONTAINS the values most
            # tightly (smallest key that still matches at this ratio) — this is
            # what makes airports.id win over flights.id for origin_id when both
            # contain every value. A partial token match between column name and
            # table name adds a smaller positive nudge; there is NO blanket
            # self-reference boost because that made origin_id/dest_id wrongly
            # resolve to flights.id.
            boost = 0
            if col_tokens:
                table_tokens = set(_tokenize_words(table))
                if any(ct in tt or tt.startswith(ct)
                       for ct in col_tokens for tt in table_tokens):
                    boost = 5
            # Key size: number of distinct values in the target PK.
            key_size = len(self._distinct(table, pk_col, self._pk_cache, self.MAX_PK_SAMPLE))
            cand = (ratio, matched, -key_size, boost, table, pk_col)
            if best is None or cand > best:
                second = best
                best = cand
            elif second is None or cand > second:
                second = cand
        if best is None:
            return None
        ratio, matched, _neg, _boost, table, pk_col = best
        # Item 4 (ambiguous): flag when the runner-up is within a hair of the
        # winner on ratio (boost/key-size tiebreakers aside).
        ambiguous = False
        if second is not None and (ratio - second[0]) < 0.03:
            ambiguous = True
        return table, pk_col, ratio, matched, ambiguous

    def should_profile(self, col):
        """Only sample columns that could plausibly reference a key (bounds cost)."""
        name = str(col["column"]).lower()
        if name.endswith(("_id", "_code", "id", "_no", "_num", "_number", "_key", "_uuid")):
            return True
        dtype = str(col.get("data_type", "")).upper()
        return any(tok in dtype for tok in ("INT", "NUMERIC", "DECIMAL", "NUMBER", "REAL", "FLOAT"))


def _is_reference_to_other_table(col_name, table_name, tables):
    """True when `col_name` (X_id style) names ANOTHER table (X / Xs / Xes / Xies).

    A column like `order_id` in a table that also has an `orders` table is
    almost certainly a foreign key, NOT the table's own primary key. Promoting
    it to a PK turns a reference into a target: it then wins overlap
    tiebreakers over the real referenced table (e.g. olist payments.order_id
    vs orders.order_id) and hides genuine relationships. This guard keeps
    FK-like columns out of PK inference.

    Matching is token-based (txt item 1: name-match evidence), so it works for
    ANY naming convention — plain `orders`, prefixed `olist_orders_dataset`,
    suffixed `tbl_order_items`, snake/camel — without per-database patches.
    """
    base = col_name[:-3].lower() if col_name.endswith("_id") else col_name.lower()
    if not base or base in _ID_TOKENS:
        return False
    base = _singularize(base)
    forms = {base, base + "s", base + "es"}
    if base.endswith("y") and len(base) > 1:
        forms.add(base[:-1] + "ies")
    forms.add(_singularize(base))

    def _matches(t):
        """Does any word-token of table `t` (or its whole name) singularize to base?"""
        if normalize_identifier(t) in forms:
            return True
        return any(_singularize(tok) in forms for tok in _tokenize_words(t))

    # Find the MOST SPECIFIC table the base could name (shortest name that
    # still contains the base as a token). `orders` names `olist_orders_dataset`
    # but `order` inside `olist_order_payments_dataset` is just a qualifier —
    # the shorter/simpler name wins. If that most-specific match is the table's
    # OWN table, the column is its own identity; otherwise it is a reference.
    best = None
    for t in tables:
        if not _matches(t):
            continue
        if best is None or len(t) < len(best):
            best = t
    return best is not None and best != table_name


def _is_code_column(col_name):
    """True for categorical label/code columns that collide with PKs by chance.

    txt item 8: `status`, `country_code`, `category`, `year`, `department` etc.
    often share values with primary keys (1..n) and must be penalized heavily
    as FK candidates — they are almost never real references.
    """
    low = col_name.lower()
    base = low[:-3] if low.endswith("_id") else low
    tokens = set(_tokenize_words(base))
    code_words = {
        "status", "category", "subcategory", "country", "city", "state",
        "region", "year", "month", "day", "week", "department", "gender",
        "channel", "segment", "tier", "type", "flag", "currency", "language",
        "color", "size", "brand", "name", "label", "grade", "level",
    }
    return any(t in code_words for t in tokens)


def infer_primary_keys(tables, declared_primary_keys):
    """Return likely PKs when the database does not declare them explicitly."""
    inferred = {}

    for table_name, columns in tables.items():
        if declared_primary_keys.get(table_name):
            inferred[table_name] = declared_primary_keys[table_name]
            continue

        table_norm = normalize_identifier(table_name)
        table_singular = _singularize(table_norm)
        candidates = []
        for col in columns:
            col_name = col["column"]
            normalized = normalize_identifier(col_name)
            if (normalized == "id"
                    or normalized == table_norm + "id"
                    or normalized == table_singular + "id"):
                candidates.append(col_name)

        if len(candidates) == 1:
            inferred[table_name] = candidates
            continue

        non_null_columns = [col["column"] for col in columns if not col["nullable"]]
        if len(non_null_columns) == 1:
            # Never trust a single non-null column as PK if it looks like a
            # foreign key (e.g. a table whose only NOT NULL column is
            # `customer_id` almost certainly points elsewhere). Only accept it
            # when it does not look like a reference to another table.
            candidate = non_null_columns[0]
            normalized = normalize_identifier(candidate)
            is_id_like = normalized.endswith("id") and normalized not in ("id", normalize_identifier(table_name) + "id")
            if not is_id_like:
                inferred[table_name] = non_null_columns
            continue

        id_like = [col["column"] for col in columns if col["column"].lower().endswith("_id") or col["column"].lower() == "id"]
        if len(id_like) == 1:
            candidate = id_like[0]
            if not _is_reference_to_other_table(candidate, table_name, tables):
                inferred[table_name] = id_like

    return inferred


def _qname(db_type, name):
    """SQL-safe quoted identifier (double quotes for PG, backticks for MySQL)."""
    if db_type == "mysql":
        return "`" + str(name).replace("`", "") + "`"
    return '"' + str(name).replace('"', "") + '"'


def _table_rows(conn, table_name, schema, db_type):
    """Best-effort row-count for the table (None when unknown).

    PostgreSQL uses pg_class.reltuples (a planner estimate, cheap and stable).
    MySQL uses a real COUNT(*) — information_schema.TABLES.TABLE_ROWS is an
    InnoDB estimate that can be 0 right after a bulk load or stale, which would
    wrongly gate PK verification. The session statement_timeout bounds the cost.
    """
    try:
        if db_type == "postgresql":
            cur = conn.cursor()
            cur.execute(
                "SELECT c.reltuples::bigint FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = %s AND c.relname = %s",
                (schema, table_name),
            )
            row = cur.fetchone()
            return row[0] if row else None
        cur = conn.cursor()
        table_ref = _qname(db_type, table_name)
        cur.execute(f"SELECT COUNT(*) FROM {table_ref}")
        row = cur.fetchone()
        return row[0] if row else None
    except Exception as exc:
        logger.debug("row count failed for %s.%s: %s", schema, table_name, exc)
        return None


def _column_is_unique_key(conn, table_name, column, schema, db_type):
    """Return True when the column is fully populated and every value is unique."""
    try:
        cur = conn.cursor()
        table_ref = (f"{_qname(db_type, schema)}.{_qname(db_type, table_name)}"
                     if db_type == "postgresql" else _qname(db_type, table_name))
        col_ref = _qname(db_type, column)
        cur.execute(
            f"SELECT count(*) AS total, count({col_ref}) AS non_null, "
            f"count(DISTINCT {col_ref}) AS distinct_values FROM {table_ref}"
        )
        row = cur.fetchone()
        if not row:
            return False
        total, non_null, distinct_values = row[0], row[1], row[2]
        return (total is not None and total > 0
                and non_null == total and distinct_values == total)
    except Exception as exc:
        logger.debug("uniqueness check failed for %s.%s: %s", table_name, column, exc)
        return False


def _column_is_empty(conn, table_name, column, schema, db_type):
    """Return True when a column holds no populated values (all NULL or blank).

    A primary key or foreign key built entirely on such a column carries no
    information, so it is dropped from the mapping instead of being reported as
    a real key.
    """
    try:
        cur = conn.cursor()
        table_ref = (f"{_qname(db_type, schema)}.{_qname(db_type, table_name)}"
                     if db_type == "postgresql" else _qname(db_type, table_name))
        col_ref = _qname(db_type, column)
        if db_type == "mysql":
            cur.execute(
                f"SELECT COUNT({col_ref}) AS non_null, "
                f"COUNT(NULLIF(TRIM(CAST({col_ref} AS CHAR)), '')) AS non_blank "
                f"FROM {table_ref}"
            )
        else:
            cur.execute(
                f"SELECT COUNT({col_ref}) AS non_null, "
                f"COUNT(NULLIF(TRIM(CAST({col_ref} AS TEXT)), '')) AS non_blank "
                f"FROM {table_ref}"
            )
        row = cur.fetchone()
        return row is not None and (row[0] or 0) == 0 and (row[1] or 0) == 0
    except Exception as exc:
        logger.debug("empty-column check failed for %s.%s: %s", table_name, column, exc)
        return False


def infer_primary_keys_from_data(conn, tables, schema, db_type, existing_pks, max_rows=500000, exclude_columns=None):
    """Data-backed PK inference for tables the name heuristics could not resolve.

    When a table declares no PK and no column *name* points to one, look at the
    actual data: an id-like column that is fully populated and whose every value
    is unique (distinct count == row count) is the de-facto key. This handles
    flat/denormalized tables whose natural key is a prefixed text id
    (e.g. 'T0000001'), which naming conventions cannot detect, and never guesses
    for a column that is merely shared with other tables (customer_id, ...).

    `exclude_columns` ((table, column) pairs) lets confirmed foreign-key columns
    be ruled out as key candidates — a column that provably references another
    table is a reference, not the table's own identity.
    """
    exclude_columns = exclude_columns or set()
    inferred = {}
    for table_name, columns in tables.items():
        if existing_pks.get(table_name):
            continue
        id_like = [c["column"] for c in columns
                   if str(c["column"]).lower().endswith("_id") or str(c["column"]).lower() == "id"]
        id_like = [c for c in id_like
                   if (table_name, c) not in exclude_columns
                   # A column whose name matches ANOTHER table (order_id -> orders)
                   # is a disguised FK, not this table's identity. Never promote
                   # it to a primary key from data alone.
                   and not _is_reference_to_other_table(c, table_name, tables)]
        if not id_like:
            continue
        rows = _table_rows(conn, table_name, schema, db_type)
        if rows is not None and rows > max_rows:
            # Too large to verify with a scan; don't guess on partial evidence.
            continue
        unique = [col for col in id_like
                  if _column_is_unique_key(conn, table_name, col, schema, db_type)]
        if not unique:
            continue
        if len(unique) == 1:
            inferred[table_name] = unique
            continue
        # Several unique id columns: only commit when exactly one carries the
        # primary-looking name (id / <table>id / <table_singular>id).
        table_norm = normalize_identifier(table_name)
        primary_named = [c for c in unique if normalize_identifier(c) in
                         ("id", table_norm + "id", _singularize(table_norm) + "id")]
        if len(primary_named) == 1:
            inferred[table_name] = primary_named
    return inferred


def infer_relationships(tables, primary_keys, declared_fks, conn=None, schema="public", db_type="postgresql", null_stats=None):
    """
    Infer relationships for columns that look like foreign keys but have NO
    declared constraint. Two general signals, in order of strength:

      1. DATA: when `conn` is given, sample distinct column values and confirm a
         relationship by value overlap with another table's primary key. This
         works for ANY naming convention (support_rep_id -> employee, reports_to
         -> employee, ...) because it relies on the values, not the names.
      2. NAME: tokenized word matching (snake_case / camelCase / plurals), which
         is a generic convention rule, not a per-database patch.

    Confidence levels:
      - "data-confirmed"     value overlap PLUS a name hint (strongest)
      - "data-inferred"      value overlap only (e.g. opaque names)
      - "heuristic-name-match"  name match only (offline / no data)
    """
    declared_pairs = {(fk["table_name"], fk["column_name"]) for fk in declared_fks}
    null_stats = null_stats or {}

    pk_owner = {}
    table_pk = {}
    for table_name, pk_cols in primary_keys.items():
        table_pk[table_name] = pk_cols
        for pk_col in pk_cols:
            pk_owner.setdefault(pk_col, []).append(table_name)

    profiler = _ValueProfiler(conn, schema, db_type, tables, primary_keys) if conn is not None else None

    def table_name_candidates(col_name):
        """For 'X_id', return tables whose name (singular or plural) is X."""
        base = col_name[:-3].lower() if col_name.endswith("_id") else col_name.lower()
        forms = {base, base + "s", base + "es"}
        if base.endswith("y") and len(base) > 1:
            forms.add(base[:-1] + "ies")
        forms.add(_singularize(base))
        names = [t for t in tables if t.lower() in forms]
        if not names:
            # Generalized token match: any naming style, not just X_id.
            col_tokens = {_singularize(tok) for tok in _tokenize_words(col_name) if tok not in _ID_TOKENS}
            if col_tokens:
                for t in tables:
                    t_tokens = {_singularize(tok) for tok in _tokenize_words(t)}
                    if col_tokens <= t_tokens:
                        names.append(t)
        return sorted(set(names))

    def _emit(table_name, col_name, candidate_table, ref_col, confidence, note,
              overlap_ratio=None, name_hint=False, ambiguous=False):
        rel = {
            "table": table_name,
            "column": col_name,
            "references_table": candidate_table,
            "references_column": ref_col,
            "confidence": confidence,
            "note": note,
        }
        # Item 4 (ambiguous): a near-tie with a runner-up target means the data
        # cannot distinguish the real reference; downstream agents should defer
        # to the human rather than guess.
        if ambiguous:
            rel["ambiguous"] = True
        # Cardinality: source distinct values (data available) vs referenced key.
        # Distinct source < referenced rows means many source rows share one
        # target -> many-to-one; equal-ish counts -> one-to-one.
        if profiler is not None and isinstance(col_name, str) and isinstance(ref_col, str):
            src_vals = profiler.source_values(table_name, col_name)
            tgt_vals = profiler.source_values(candidate_table, ref_col)
            if src_vals is not None and tgt_vals is not None and len(tgt_vals) > 0:
                ratio = len(src_vals) / len(tgt_vals)
                rel["cardinality"] = (
                    "one-to-one" if ratio >= 0.9
                    else "one-to-many" if ratio > 1.1
                    else "many-to-one"
                )

        # Item 1: explainable multi-factor confidence score.
        evidence = []
        score = 0.0
        # Priority 2 (txt): structured evidence object alongside the readable list.
        detail = {
            "name_match": False,
            "datatype_match": None,   # None when no data profiler available
            "target_unique": False,
            "value_overlap": overlap_ratio,
            "null_rate": None,
            "cardinality": rel.get("cardinality"),
        }

        # name_match (item 1) — column base matches the target table name.
        name_hit = name_hint
        if not name_hit:
            base = _singularize(col_name[:-3] if col_name.endswith("_id") else col_name)
            tgt_tokens = {_singularize(tok) for tok in _tokenize_words(candidate_table)}
            if base in tgt_tokens:
                name_hit = True
        detail["name_match"] = name_hit
        if name_hit:
            score += 25
            evidence.append("column name matches target table")

        # datatype_match (item 4).
        if profiler is not None and isinstance(col_name, str) and isinstance(ref_col, str):
            dt_ok = profiler.types_compatible(table_name, col_name, candidate_table, ref_col)
            detail["datatype_match"] = bool(dt_ok)
            if dt_ok:
                score += 15
                evidence.append("datatype compatible")
            else:
                evidence.append("datatype mismatch")

        # overlap_ratio (item 5) — tiered value evidence.
        if overlap_ratio is not None:
            detail["value_overlap"] = overlap_ratio
            if overlap_ratio >= 0.99:
                score += 30
                evidence.append("~100% values found in target PK")
            elif overlap_ratio >= 0.95:
                score += 25
                evidence.append(">=95% values found in target PK")
            elif overlap_ratio >= 0.80:
                score += 18
                evidence.append(">=80% values found in target PK")
            else:
                evidence.append("partial value overlap")

        # uniqueness: target column is a primary key.
        tgt_pk = set(table_pk.get(candidate_table, []))
        target_unique = False
        if isinstance(ref_col, str) and ref_col in tgt_pk:
            target_unique = True
            score += 20
            evidence.append("target is a primary key")
        elif not isinstance(ref_col, str):
            # Composite: referenced columns are key columns by construction.
            if all(c in tgt_pk for c in ref_col):
                target_unique = True
                score += 20
                evidence.append("target is a composite primary key")
        detail["target_unique"] = target_unique

        # cardinality: many-to-one / one-to-one favour a real FK over 1:1 sharing.
        if rel.get("cardinality") in ("many-to-one", "one-to-many"):
            score += 10
            evidence.append("many source rows map to one target row")

        # Item 6: nullable FK intelligence. High null rate lowers confidence
        # (sparse reference) but does NOT kill a name+datatype+overlap match.
        if profiler is not None and isinstance(col_name, str):
            ns = null_stats.get(table_name, {}).get(col_name, {})
            null_pct = ns.get("null_pct") if isinstance(ns, dict) else None
            if null_pct is not None:
                detail["null_rate"] = null_pct
                if null_pct > 0.8:
                    score -= 25
                    evidence.append(f"high null rate ({null_pct:.0%}) — relationship optional")
                elif null_pct > 0.5:
                    score -= 10
                    evidence.append(f"nullable ({null_pct:.0%})")

        rel["confidence_score"] = max(0, min(100, round(score)))
        rel["evidence"] = evidence
        rel["evidence_detail"] = detail

        # Item 10: review tier.
        cs = rel["confidence_score"]
        rel["review_status"] = (
            "auto-accept" if cs >= 80
            else "flagged" if cs >= 60
            else "review"
        )

        # Priority 5 (txt): CONFIRMED / PROBABLE / UNCERTAIN / REJECTED taxonomy.
        # REJECTED candidates are never emitted, so emitted relationships use the
        # three positive states; a null/empty/nullable-heavy hit that scores below
        # the review floor would instead be surfaced as UNCERTAIN for a human.
        rel["relationship_state"] = (
            "CONFIRMED" if cs >= 80
            else "PROBABLE" if cs >= 60
            else "UNCERTAIN"
        )

        # Item 2: self-referencing FK flag.
        rel["self_referencing"] = bool(
            isinstance(col_name, str) and table_name == candidate_table
            and col_name != ref_col
        )

        # Priority 3 (txt): unified relationship type.
        if rel.get("self_referencing"):
            rel["relationship_type"] = "self-referencing"
        elif rel.get("cardinality"):
            rel["relationship_type"] = rel["cardinality"]
        else:
            rel["relationship_type"] = "foreign-key"

        inferred.append(rel)

    def _is_ref_flavored(name):
        """Expanded FK name heuristic: catches _id, _by, _to, _from, _via, _for, etc."""
        low = name.lower()
        if low == "id":
            return True
        for suffix in ("_id", "_by", "_to", "_from", "_via", "_for", "_at",
                        "_with", "_against", "_on", "_of", "_type"):
            if low.endswith(suffix):
                return True
        if low.endswith("id") and len(low) > 2:
            return True
        return False

    inferred = []
    for table_name, columns in tables.items():
        for col in columns:
            col_name = col["column"]

            if (table_name, col_name) in declared_pairs:
                continue

            own_pk = primary_keys.get(table_name, [])
            if col_name in own_pk:
                # A sole primary key is the table's own identity, never a FK.
                # A column of a COMPOSITE key may also reference another table
                # (junction tables), so it stays eligible but only via DATA.
                if len(own_pk) == 1:
                    continue
                is_composite_key_col = True
                # Composite PK columns with very few distinct values that form
                # a contiguous range starting from 1 are line counters or sort
                # keys (e.g. item_id=1,2,3,4,5), not FKs. They overlap 100%
                # with every small PK and create false positives.
                if profiler is not None:
                    _pk_vals = profiler.source_values(table_name, col_name)
                    if _pk_vals is not None and len(_pk_vals) <= 6:
                        # Extract numeric values from normalized tuples
                        _nums = sorted(v[1] if isinstance(v, tuple) else v
                                       for v in _pk_vals
                                       if (isinstance(v, tuple) and len(v) >= 2
                                           and isinstance(v[1], (int, float)))
                                       or isinstance(v, (int, float)))
                        if (_nums and _nums[0] == 1
                                and _nums == list(range(1, len(_nums) + 1))):
                            continue
            else:
                is_composite_key_col = False

            is_ref_flavored = _is_ref_flavored(col_name)
            if not is_ref_flavored:
                # When connected, allow numeric/UUID columns through — they may
                # be FKs with non-standard names (reports_to, ship_via, …).
                if not (profiler and profiler.should_profile(col)):
                    continue

            candidate_tables = set(pk_owner.get(col_name, []))
            table_cands = set(table_name_candidates(col_name))
            if not table_cands:
                # Fallback: a column that shares its name with ANOTHER table's
                # primary key (only when that name is meaningful). A same-named
                # non-key column is a coincidental attribute, not a relationship.
                same_name = [
                    other_table
                    for other_table in tables
                    if other_table != table_name
                    and col_name in table_pk.get(other_table, [])
                    and len(normalize_identifier(col_name)) >= 3
                ]
                if len(same_name) == 1:
                    candidate_tables.update(same_name)
            candidate_tables.update(table_cands)
            candidate_tables.discard(table_name)

            if profiler is not None:
                src_set = profiler.source_values(table_name, col_name)
                rows = profiler.row_count(table_name)
                hit = None
                if is_ref_flavored or is_composite_key_col:
                    # id-suffixed columns: the name is already a strong signal,
                    # so data only needs a modest overlap to confirm. When the
                    # name gives NO target at all (opaque support_rep_id), let
                    # the data speak against every primary key — but only pick a
                    # target it can identify unambiguously.
                    hit = profiler.strongest_overlap(
                        src_set,
                        only_tables=candidate_tables or None,
                        min_source=2,
                        exclude=(table_name, col_name),
                        require_unambiguous=not candidate_tables,
                        source_table=table_name,
                        source_col=col_name,
                    )
                elif not candidate_tables:
                    # Non-id columns (reports_to): only data can find them, and
                    # only when high-cardinality enough to not be a code/status.
                    hit = profiler.strongest_overlap(
                        src_set,
                        high_cardinality=True,
                        row_count=rows,
                        exclude=(table_name, col_name),
                        source_table=table_name,
                        source_col=col_name,
                    )
                if hit is None and candidate_tables and (is_ref_flavored or is_composite_key_col):
                    # Name already narrows the target; verify against the FULL
                    # target key (exact containment) so huge key tables are not
                    # diluted by sampling (e.g. warranty -> 1M-row sales).
                    hit = profiler.strongest_exact_overlap(
                        table_name, col_name, candidate_tables,
                        exclude=(table_name, col_name),
                    )
                if hit is None and not candidate_tables and is_ref_flavored:
                    # Small integers (reports_to, ship_via, …) match every PK
                    # in the sample. Use exact SQL containment against ALL tables
                    # to find the real target via full-key containment ratio.
                    all_tables = set(t for t, _, _ in profiler.pk_index)
                    hit = profiler.strongest_exact_overlap(
                        table_name, col_name, all_tables,
                        exclude=(table_name, col_name),
                    )
                if hit is not None:
                    target_table, target_col, ratio, matched, ambiguous = hit
                    # Negative evidence (txt item 7) + common-value collisions
                    # (txt item 8): a code/label column (status, category, year,
                    # country_code, ...) whose values merely overlap a key is
                    # NOT a real FK. Only accept it when the name itself points
                    # at the target table (e.g. `order_status` -> status table)
                    # — otherwise its "overlap" is coincidence.
                    if _is_code_column(col_name):
                        name_hint = target_table in candidate_tables
                        if not name_hint:
                            hit = None
                    if hit is not None:
                        target_table, target_col, ratio, matched, ambiguous = hit
                        if target_table in candidate_tables:
                            confidence, note = "data-confirmed", (
                                f"No FK constraint declared; confirmed by value overlap "
                                f"({matched}/{len(src_set)} sample values) with {target_table}.{target_col}.")
                        else:
                            confidence, note = "data-inferred", (
                                f"No FK constraint declared and no name match; inferred from "
                                f"value overlap ({matched}/{len(src_set)} sample values) with {target_table}.{target_col}.")
                    if hit is not None:
                        ref_cols = table_pk.get(target_table) or [target_col]
                        for ref_col in ref_cols:
                            # A column can never reference itself (same table + same column).
                            if table_name == target_table and col_name == ref_col:
                                continue
                            _emit(table_name, col_name, target_table, ref_col, confidence, note,
                                  overlap_ratio=ratio, name_hint=(target_table in candidate_tables),
                                  ambiguous=ambiguous)
                        continue

            if is_composite_key_col:
                # Composite-key columns only produce relationships when the data
                # confirms them; a name match alone is not enough.
                continue

            # Negative evidence (txt item 7): a column whose most-specific
            # table-name match is its OWN table is its identity, not a reference.
            # e.g. orders.order_id matches both `orders` and `order_items` tokens;
            # the shorter/most-specific name `orders` is the table itself, so the
            # column must not be emitted as a name-only FK to `order_items`.
            if not _is_reference_to_other_table(col_name, table_name, tables):
                continue

            for candidate_table in sorted(candidate_tables):
                ref_cols = table_pk.get(candidate_table) or [col_name]
                for ref_col in ref_cols:
                    _emit(table_name, col_name, candidate_table, ref_col,
                          "heuristic-name-match",
                          "No FK constraint declared in the database; relationship inferred from column naming convention.",
                          name_hint=True)

    # Composite relationships: (a, b) -> (x, y). A source table may reference a
    # target table whose PRIMARY KEY spans several columns (junction/detail rows).
    # Detect it by name first, then confirm the composite tuple by data.
    if profiler is not None:
        for target_table, target_pk in sorted(table_pk.items()):
            if len(target_pk) < 2:
                continue
            declared_tgt = {
                (fk["table_name"], fk["references_table"], fk["references_column"])
                for fk in declared_fks
            }
            for src_table, columns in tables.items():
                if src_table == target_table:
                    continue
                src_pk = primary_keys.get(src_table, [])
                cols = {c["column"] for c in columns}
                matched = [c for c in target_pk if c in cols and c not in src_pk]
                if len(matched) < 2:
                    continue
                if all((src_table, target_table, c) in declared_tgt for c in matched):
                    continue
                matched_norm = set(normalize_identifier(c) for c in matched)
                src_names = set(normalize_identifier(c) for c in cols)
                table_norm = normalize_identifier(target_table)
                # Require the matched names to actually resemble the target's key
                # (e.g. order_id + product_id vs order_item's order_id+product_id),
                # not two coincidental shared names.
                name_ok = any(
                    n == table_norm + "_" + mn or mn in (n, n + "_id")
                    for n in src_names for mn in matched_norm
                )
                if not name_ok:
                    continue
                if not profiler.types_compatible(src_table, matched[0], target_table, target_pk[0]):
                    continue
                # Confirm composite containment by data: fraction of source rows
                # whose (matched...) tuple exists in the target composite key.
                try:
                    matched_cols = [_quote_ident(profiler.db_type, c) for c in matched]
                    tgt_cols = [_quote_ident(profiler.db_type, c) for c in target_pk]
                    if profiler.db_type == "mysql":
                        join_sql = " AND ".join(
                            f"s.{mc} = t.{tc}" for mc, tc in zip(matched_cols, tgt_cols)
                        )
                        q = (
                            f"SELECT (COUNT(DISTINCT s.{matched_cols[0]}) / NULLIF(COUNT(DISTINCT t.{tgt_cols[0]}),0)) "
                            f"FROM {_quote_ident(profiler.db_type, src_table)} s "
                            f"LEFT JOIN {_quote_ident(profiler.db_type, target_table)} t ON {join_sql}"
                        )
                    else:
                        join_sql = " AND ".join(
                            f"s.{mc} = t.{tc}" for mc, tc in zip(matched_cols, tgt_cols)
                        )
                        q = (
                            f"SELECT COUNT(DISTINCT s.{matched_cols[0]})::float / NULLIF(COUNT(DISTINCT t.{tgt_cols[0]}),0) "
                            f"FROM {_quote_ident(profiler.db_type, profiler.schema)}.{_quote_ident(profiler.db_type, src_table)} s "
                            f"LEFT JOIN {_quote_ident(profiler.db_type, profiler.schema)}.{_quote_ident(profiler.db_type, target_table)} t ON {join_sql}"
                        )
                    with _dict_cursor(profiler.conn, profiler.db_type) as cur:
                        cur.execute(q)
                        ratio = cur.fetchone().get("ratio") or 0.0
                except Exception as exc:
                    logger.debug("composite check failed %s -> %s: %s", src_table, target_table, exc)
                    continue
                if ratio >= profiler.MIN_OVERLAP:
                    _emit(
                        src_table, matched, target_table, target_pk, "composite-data-confirmed",
                        f"Composite FK inferred: ({', '.join(matched)}) references "
                        f"{target_table}({', '.join(target_pk)}) with {ratio:.0%} of "
                        f"source keys present in the target composite key.",
                        overlap_ratio=ratio, name_hint=True,
                    )
    return inferred


# ---------------------------------------------------------------------------
# LLM-assisted schema reasoning (optional enrichment over the heuristics)
#
# Runs a single structured pass with whichever LLM backend is active — local
# (Ollama via `aria-goal`) or cloud (Groq) — through the unified
# LLMProvider.chat() interface. The LLM:
#   * confirms or disproves the heuristic candidate relationships,
#   * may add genuine FK-like links the heuristics missed,
#   * suggests a primary key only when none was declared or inferred,
#   * attaches plain-English table descriptions and measure/dimension hints.
#
# Every suggestion is validated against the real schema before it is merged, so
# a hallucinated table/column is never accepted. Any timeout, failure or
# non-JSON response leaves the deterministic mapping untouched.
# ---------------------------------------------------------------------------

_SCHEMA_REASONING_ROLE = "schema"

_SCHEMA_REASONING_PROMPT = """\
You are a careful database schema analyst for a BI system. Below is a schema
summary (tables, columns, data types, nullability, declared/inferred primary
keys, and UNVERIFIED candidate relationships). Reason about it and return STRICT
JSON only, with exactly this shape:

{
  "tables": {
    "<table_name>": {
      "description": "one-line plain-English purpose of this table",
      "pk_candidates": ["column names likely to be a primary key, ONLY if the table has none declared or inferred; else []"],
      "measures": ["numeric columns that are business measurements to aggregate (never ids, codes, flags or timestamps)"],
      "dimensions": ["categorical/text columns useful for grouping and filtering"]
    }
  },
  "relationships": [
    {"table": "...", "column": "...", "references_table": "...", "references_column": "...",
     "kind": "confirm", "reason": "short justification"}
  ]
}

Rules:
- Refer ONLY to tables and columns that exist in the schema above. Never invent names.
- "kind": "confirm" for a candidate relationship you believe is a real foreign key;
  "add" for a genuine FK-like link the candidates missed. Skip same-name columns
  that merely coincide (e.g. two unrelated "amount" columns).
- CRITICAL: Look for foreign keys beyond X_id naming. Columns like reports_to,
  ship_via, created_by, parent_id, managed_by, etc. may reference other tables.
  A column referencing the same table (self-reference) is valid (e.g. reports_to
  in employees referencing employee_id).
- Use exact, case-sensitive table and column names.
- When unsure, omit rather than guess. No markdown, no prose, JSON only.
"""


_SCHEMA_REASONING_PROMPT_TAIL = "\nSchema:\n"


def enrich_with_llm(mapping, llm):
    """Optional LLM-assisted reasoning pass over a built schema mapping.

    Works identically for every provider (Ollama 'local' or Groq 'cloud').
    Returns the mapping unchanged when no LLM is given, the call fails/times
    out, or the response is unusable.
    """
    if llm is None:
        return mapping

    summary = _condense_schema(mapping)
    if not summary.strip():
        return mapping

    try:
        timeout = 300 if getattr(llm, "provider", None) == "local" else 60
        content = llm.chat(
            _SCHEMA_REASONING_ROLE,
            messages=[{"role": "user", "content": _SCHEMA_REASONING_PROMPT + _SCHEMA_REASONING_PROMPT_TAIL + summary}],
            temperature=0.0,
            num_predict=2500,
            timeout=timeout,
        )
    except Exception as exc:
        print(f"WARNING: schema LLM reasoning skipped ({exc}); keeping heuristic results.")
        return mapping

    parsed = _parse_schema_reasoning(content)
    if not parsed or not (isinstance(parsed.get("tables"), dict)
                          or isinstance(parsed.get("relationships"), list)):
        print("WARNING: schema LLM reasoning returned no usable JSON; keeping heuristic results.")
        return mapping

    try:
        model = llm.model_for(_SCHEMA_REASONING_ROLE)
    except Exception:
        model = None
    mapping["llm_reasoning"] = {
        "provider": getattr(llm, "provider", None),
        "model": model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _apply_schema_reasoning(mapping, parsed)
    return mapping


def _condense_schema(mapping, max_tables=60, max_chars=12000):
    """Compact text summary of the mapping for the reasoning LLM."""
    tables = mapping.get("tables", {})
    if not tables:
        return ""
    lines = [f"Database: {mapping.get('database', 'unknown')} "
             f"(schema: {mapping.get('schema', '?')})"]

    for idx, (table_name, info) in enumerate(tables.items()):
        if idx >= max_tables:
            lines.append("... (additional tables omitted)")
            break
        pk = info.get("primary_key") or info.get("inferred_primary_key") or []
        pk_str = ", ".join(pk) if isinstance(pk, (list, tuple)) and pk else "none declared/inferred"
        lines.append(f"TABLE {table_name} (PK: {pk_str})")
        for col in info.get("columns", []):
            npct = info.get("null_stats", {}).get(col["column"], {}).get("null_pct")
            null_info = "NULL" if col.get("nullable") else "NOT NULL"
            if npct is not None:
                null_info += f" ({npct}% null)"
            lines.append(f"  {col['column']} {col.get('data_type', '?')} {null_info}")
        lines.append("")

    declared = mapping.get("declared_relationships", [])
    if declared:
        lines.append("DECLARED FOREIGN KEYS:")
        for d in declared:
            lines.append(f"  {d['table_name']}.{d['column_name']} -> "
                         f"{d['references_table']}.{d['references_column']}")
        lines.append("")

    inferred = mapping.get("inferred_relationships", [])
    if inferred:
        lines.append("CANDIDATE RELATIONSHIPS (unverified, from naming heuristics):")
        for r in inferred:
            lines.append(f"  {r['table']}.{r['column']} -> "
                         f"{r['references_table']}.{r['references_column']}")

    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n... (truncated)"
    return summary


def _parse_schema_reasoning(content):
    """Robustly parse the reasoning response into a dict ({} on any failure).

    Falls back to the largest object-bounded prefix when the model's reply is
    truncated mid-JSON (the token cap can be hit before the closing brace), so
    a partial but otherwise-correct answer is not discarded wholesale.
    """
    if not content or not isinstance(content, str):
        return {}
    text = content.strip()
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    candidate = text[start:end + 1]
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    # Truncation salvage: try each `{` boundary before the final `}` and keep
    # the largest prefix that still parses as a dict.
    for open_idx in range(start, end):
        if text[open_idx] != "{":
            continue
        try:
            data = json.loads(text[open_idx:end + 1])
        except Exception:
            continue
        if isinstance(data, dict):
            return data
    return {}


_NUMERIC_TYPE_HINTS = (
    "int", "numeric", "decimal", "float", "real", "double", "serial", "money",
)


def _is_numeric_type(data_type):
    t = (data_type or "").lower()
    return any(h in t for h in _NUMERIC_TYPE_HINTS)


def _column_data_type(table_info, col_name):
    for c in table_info.get("columns", []):
        if c["column"] == col_name:
            return c.get("data_type", "")
    return ""


def _apply_schema_reasoning(mapping, parsed):
    """Merge validated LLM suggestions into the mapping (never invents names)."""
    tables = mapping.setdefault("tables", {})
    declared = mapping.get("declared_relationships", [])
    declared_pairs = {(d["table_name"], d["column_name"]) for d in declared}
    inferred = mapping.setdefault("inferred_relationships", [])
    edges = mapping.setdefault("relationship_edges", [])

    def pair(r):
        return (r.get("table"), r.get("column"), r.get("references_table"), r.get("references_column"))

    # -- table-level annotations ------------------------------------------
    llm_tables = parsed.get("tables") or {}
    if isinstance(llm_tables, dict):
        for table_name, info in llm_tables.items():
            if table_name not in tables or not isinstance(info, dict):
                continue
            table_info = tables[table_name]
            real_columns = {c["column"] for c in table_info.get("columns", [])}

            desc = info.get("description")
            if isinstance(desc, str) and desc.strip():
                table_info["description"] = desc.strip()

            measures = [c for c in (info.get("measures") or []) if isinstance(c, str)]
            dims = [c for c in (info.get("dimensions") or []) if isinstance(c, str)]
            measures = [c for c in measures if c in real_columns and _is_numeric_type(_column_data_type(table_info, c))]
            dims = [c for c in dims if c in real_columns and c not in measures]
            if measures or dims:
                table_info.setdefault("semantic_tags", {}).update(
                    {"measures": measures, "dimensions": dims}
                )

            existing_pk = table_info.get("primary_key") or table_info.get("inferred_primary_key") or []
            pk_cands = [c for c in (info.get("pk_candidates") or []) if isinstance(c, str) and c in real_columns]
            if not existing_pk and len(pk_cands) == 1:
                cand = pk_cands[0]
                nullable = next((c.get("nullable") for c in table_info.get("columns", []) if c["column"] == cand), True)
                null_pct = table_info.get("null_stats", {}).get(cand, {}).get("null_pct")
                if nullable is False or null_pct in (0, 0.0):
                    table_info["inferred_primary_key"] = [cand]
                    if not table_info.get("primary_key"):
                        table_info["primary_key"] = [cand]

    # -- relationship confirm / add ----------------------------------------
    llm_rels = parsed.get("relationships") or []
    if isinstance(llm_rels, list):
        for rel in llm_rels:
            if not isinstance(rel, dict):
                continue
            table = rel.get("table")
            column = rel.get("column")
            ref_table = rel.get("references_table")
            ref_col = rel.get("references_column")
            kind = rel.get("kind")
            if not all(isinstance(x, str) and x for x in (table, column, ref_table, ref_col)):
                continue
            if table not in tables or ref_table not in tables:
                continue  # hallucinated table
            if column not in {c["column"] for c in tables[table].get("columns", [])}:
                continue
            if ref_col not in {c["column"] for c in tables[ref_table].get("columns", [])}:
                continue
            # The source column must look like a reference (id-like name, FK
            # pattern like _by/_to/_via, or an exact match to the target column):
            # this stops the LLM from "confirming" a numeric measure (e.g.
            # `total`) as a foreign key.
            col_norm = normalize_identifier(column)
            ref_norm = normalize_identifier(ref_col)
            col_dtype = _column_data_type(tables.get(table, {}), column).lower()
            is_id_like = col_norm.endswith("id")
            is_fk_pattern = column.lower().endswith(("_id", "_by", "_to", "_from", "_via", "_for", "_at", "_with", "_against", "_on", "_of", "_type"))
            is_numeric_type = any(tok in col_dtype for tok in ("int", "numeric", "decimal", "serial"))
            if not (is_id_like or is_fk_pattern or col_norm == ref_norm or is_numeric_type):
                continue
            if (table, column) in declared_pairs:
                continue  # already constrained; don't second-guess the DB

            new_pair = (table, column, ref_table, ref_col)
            if kind == "confirm":
                existing = [r for r in inferred
                            if r.get("table") == table and r.get("column") == column]
                if existing:
                    # Only upgrade confidence if LLM agrees with the existing
                    # heuristic target. Never replace a heuristic target with
                    # the LLM's guess — small LLMs often pick wrong targets.
                    for r in existing:
                        if (r.get("references_table") == ref_table
                                and r.get("references_column") == ref_col):
                            r["confidence"] = "llm-confirmed"
                            r["note"] = ("Confirmed by LLM reasoning "
                                         "(no FK constraint declared in the database).")
                            # Also update the matching edge
                            for e in edges:
                                if (e.get("source_table") == table
                                        and e.get("source_column") == column
                                        and e.get("target_table") == ref_table):
                                    e["confidence"] = "llm-confirmed"
                else:
                    # No existing heuristic — LLM is the only signal; add it.
                    inferred.append({
                        "table": table, "column": column,
                        "references_table": ref_table, "references_column": ref_col,
                        "confidence": "llm-confirmed",
                        "note": "Confirmed by LLM reasoning (no FK constraint declared in the database).",
                    })
                    edges.append({
                        "source_table": table, "source_column": column,
                        "target_table": ref_table, "target_column": ref_col,
                        "type": "inferred", "confidence": "llm-confirmed",
                    })
            elif kind == "add" and new_pair not in {pair(r) for r in inferred}:
                # Only add when this column has no inferred FK yet (no ambiguity).
                if not any(r.get("table") == table and r.get("column") == column for r in inferred):
                    inferred.append({
                        "table": table, "column": column,
                        "references_table": ref_table, "references_column": ref_col,
                        "confidence": "llm-reasoned",
                        "note": "Proposed by LLM reasoning (no FK constraint declared in the database).",
                    })
                    edges.append({
                        "source_table": table, "source_column": column,
                        "target_table": ref_table, "target_column": ref_col,
                        "type": "inferred", "confidence": "llm-reasoned",
                    })


def _canonicalize_mapping(mapping):
    """Validation pass: every relationship/edge must reference real objects.

    Runs AFTER all sources (declared FKs, heuristics, LLM) have merged so the
    final payload is guaranteed internally consistent:
      * tables/columns in relationships exist in `mapping["tables"]`
      * relationship_edges mirror declared+inferred relationships (no strays)
      * no duplicate relationship entries
    """
    tables = mapping.get("tables", {})
    known_columns = {
        t: {c["column"] for c in info.get("columns", [])}
        for t, info in tables.items()
    }

    def valid_rel(r, kind):
        if kind == "declared":
            src, src_col = r.get("table_name"), r.get("column_name")
            tgt, tgt_col = r.get("references_table"), r.get("references_column")
        else:
            src, src_col = r.get("table"), r.get("column")
            tgt, tgt_col = r.get("references_table"), r.get("references_column")
        if src not in known_columns or tgt not in known_columns:
            return False
        # Composite relationships carry a LIST of columns.
        cols = src_col if isinstance(src_col, list) else [src_col]
        ref_cols = tgt_col if isinstance(tgt_col, list) else [tgt_col]
        if any(c not in known_columns[src] for c in cols):
            return False
        if any(c not in known_columns[tgt] for c in ref_cols):
            return False
        return True

    mapping["declared_relationships"] = [
        r for r in mapping.get("declared_relationships", []) if valid_rel(r, "declared")
    ]
    mapping["inferred_relationships"] = [
        r for r in mapping.get("inferred_relationships", []) if valid_rel(r, "inferred")
    ]

    # Rebuild edges from the authoritative relationship lists.
    edges = []
    seen_edges = set()
    for r in mapping["declared_relationships"]:
        edges.append({
            "source_table": r["table_name"], "source_column": r["column_name"],
            "target_table": r["references_table"], "target_column": r["references_column"],
            "type": "declared", "confidence": "declared",
        })
    for r in mapping["inferred_relationships"]:
        edges.append({
            "source_table": r["table"], "source_column": r["column"],
            "target_table": r["references_table"], "target_column": r["references_column"],
            "type": "inferred", "confidence": r.get("confidence", "inferred"),
            "cardinality": r.get("cardinality"),
        })
    for e in edges:
        key = (e["source_table"], str(e["source_column"]), e["target_table"], str(e["target_column"]))
        if key not in seen_edges:
            seen_edges.add(key)
    deduped = []
    for e in edges:
        key = (e["source_table"], str(e["source_column"]), e["target_table"], str(e["target_column"]))
        if key in seen_edges:
            deduped.append(e)
            seen_edges.discard(key)
    mapping["relationship_edges"] = deduped


def _build_relationship_graph(mapping):
    """Build an explicit per-table relationship graph (txt item 9).

    For every table: its primary key, columns, and separately its OUTGOING
    (this table references others) and INCOMING (others reference this table)
    relationships. Also detects MANY-TO-MANY (txt item 2) via junction tables:
    a table whose composite primary key columns each reference another table is
    a join table; the referenced tables get a many-to-many link through it.
    """
    tables = mapping.get("tables", {})
    edges = mapping.get("relationship_edges", [])
    graph = {}

    # Normalize an edge to (table, column) -> (target, target_column).
    def _out_edges(t):
        out = []
        for e in edges:
            if e.get("source_table") == t:
                out.append(e)
        return out

    def _in_edges(t):
        incoming = []
        for e in edges:
            if e.get("target_table") == t:
                incoming.append(e)
        return incoming

    for t, info in tables.items():
        pks = info.get("primary_key") or info.get("inferred_primary_key") or []
        graph[t] = {
            "primary_key": pks,
            "columns": [c["column"] for c in info.get("columns", [])],
            "outgoing_relationships": [],
            "incoming_relationships": [],
            "many_to_many": [],
        }
        for e in _out_edges(t):
            graph[t]["outgoing_relationships"].append({
                "source_column": e.get("source_column"),
                "target_table": e.get("target_table"),
                "target_column": e.get("target_column"),
                "type": e.get("type"),
                "confidence": e.get("confidence"),
                "cardinality": e.get("cardinality"),
                "ambiguous": e.get("ambiguous"),
            })
        for e in _in_edges(t):
            graph[t]["incoming_relationships"].append({
                "source_table": e.get("source_table"),
                "source_column": e.get("source_column"),
                "target_column": e.get("target_column"),
                "type": e.get("type"),
                "confidence": e.get("confidence"),
                "cardinality": e.get("cardinality"),
                "ambiguous": e.get("ambiguous"),
            })

    # Many-to-many via junction tables (composite PK, each PK column a FK).
    for t, info in tables.items():
        pks = info.get("primary_key") or info.get("inferred_primary_key") or []
        if len(pks) < 2:
            continue
        refs = {}
        for e in _out_edges(t):
            if e.get("source_column") in pks and e.get("target_table") != t:
                refs[e["source_column"]] = e["target_table"]
        if len(refs) >= 2:
            # A junction: every PK column points at a different table.
            involved = sorted(set(refs.values()))
            if len(involved) >= 2:
                junction = {
                    "junction_table": t,
                    "columns": list(refs.keys()),
                    "tables": involved,
                }
                for parent in involved:
                    graph[parent]["many_to_many"].append(junction)
    return graph


# ---------------------------------------------------------------------------
# Assemble & write schema_mapping.json
# ---------------------------------------------------------------------------

def schema_has_tables(conn, schema):
    """Return True when the given Postgres schema contains at least one base table."""
    if not schema:
        return False
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
            )
            """,
            (schema,),
        )
        return bool(cur.fetchone()[0])


def find_schema_with_tables(conn):
    """Return the largest non-system Postgres schema that contains base tables.

    Prefers the schema with the most tables (best signal for relationship
    inference). Falls back to 'public' when only it has tables.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
              AND table_schema NOT LIKE 'pg_%'
              AND table_type = 'BASE TABLE'
            GROUP BY table_schema
            ORDER BY count(*) DESC, table_schema
            LIMIT 1
            """
        )
        row = cur.fetchone()
        return row[0] if row else "public"


def _drop_empty_key_columns(mapping, conn, schema, db_type):
    """Remove any primary/foreign key whose source column is fully empty.

    A key built on a column with no populated values (all NULL or blank) cannot
    identify or reference anything, so it is dropped from the mapping. The
    removals are recorded under mapping["empty_key_columns_removed"] so the
    decision is transparent. On any query failure the column is kept (safe).
    """
    tables = mapping.setdefault("tables", {})
    removed_pks, removed_fks = [], []

    for table_name, info in tables.items():
        col_names = {c["column"] for c in info.get("columns", [])}
        for key_field in ("primary_key", "inferred_primary_key"):
            kept = []
            for pk in info.get(key_field, []):
                if pk in col_names and _column_is_empty(conn, table_name, pk, schema, db_type):
                    removed_pks.append((table_name, pk))
                else:
                    kept.append(pk)
            info[key_field] = kept
        if not info.get("primary_key"):
            info["primary_key"] = []

    def source_pair(rel, declared):
        if declared:
            return rel.get("table_name"), rel.get("column_name")
        return rel.get("table"), rel.get("column")

    removed_fk_pairs = set()
    for rel_list, declared in ((mapping.get("declared_relationships", []), True),
                               (mapping.get("inferred_relationships", []), False)):
        kept = []
        for rel in rel_list:
            table_name, col_name = source_pair(rel, declared)
            col_names = {c["column"] for c in tables.get(table_name, {}).get("columns", [])}
            if (table_name in tables and col_name in col_names
                    and _column_is_empty(conn, table_name, col_name, schema, db_type)):
                removed_fk_pairs.add((table_name, col_name))
                removed_fks.append((table_name, col_name, declared))
            else:
                kept.append(rel)
        rel_list[:] = kept

    if removed_fk_pairs:
        mapping["relationship_edges"] = [
            e for e in mapping.get("relationship_edges", [])
            if (e.get("source_table"), e.get("source_column")) not in removed_fk_pairs
        ]

    mapping["empty_key_columns_removed"] = {
        "primary_keys": sorted(set(removed_pks)),
        "foreign_keys": sorted(set(removed_fks)),
    }
    return mapping


def detect_schema_drift(previous, current):
    """Compare two schema mappings (txt item 5) and report what changed.

    Detects new/removed tables, added/removed columns, changed data types,
    changed PK/FK structure, and newly inferred relationships. Used when the
    agent runs repeatedly against a live database whose schema evolves.
    """
    prev_tables = previous.get("tables", {}) if previous else {}
    curr_tables = current.get("tables", {}) if current else {}

    def _norm(rel, source_side):
        return {
            "table": rel[source_side],
            "column": rel["column"] if source_side == "table" else rel["column"],
            "references_table": rel["references_table"],
            "references_column": rel["references_column"],
        }

    def _edge_set(mapping, declared=True):
        out = set()
        for rel in mapping.get("declared_relationships", []):
            if declared:
                out.add((rel["table_name"], str(rel["column_name"]),
                         rel["references_table"], str(rel["references_column"])))
        return out

    def _inferred_set(mapping):
        out = set()
        for rel in mapping.get("inferred_relationships", []):
            col = rel.get("column")
            if isinstance(col, list):
                col = ",".join(col)
            ref = rel.get("references_column")
            if isinstance(ref, list):
                ref = ",".join(ref)
            out.add((rel.get("table"), str(col),
                     rel.get("references_table"), str(ref)))
        return out

    drift = {"has_changes": False, "changed_tables": []}

    new_tables = sorted(set(curr_tables) - set(prev_tables))
    removed_tables = sorted(set(prev_tables) - set(curr_tables))
    if new_tables:
        drift["new_tables"] = new_tables
        drift["has_changes"] = True
    if removed_tables:
        drift["removed_tables"] = removed_tables
        drift["has_changes"] = True

    added_columns = {}
    removed_columns = {}
    datatype_changed = {}
    pk_changed = {}
    for table in sorted(set(curr_tables) & set(prev_tables)):
        prev_cols = {c["column"]: c for c in prev_tables[table].get("columns", [])}
        curr_cols = {c["column"]: c for c in curr_tables[table].get("columns", [])}
        new_cols = sorted(set(curr_cols) - set(prev_cols))
        gone_cols = sorted(set(prev_cols) - set(curr_cols))
        type_changes = {}
        for col in sorted(set(curr_cols) & set(prev_cols)):
            if str(prev_cols[col].get("data_type", "")).lower() != str(curr_cols[col].get("data_type", "")).lower():
                type_changes[col] = {
                    "from": prev_cols[col].get("data_type"),
                    "to": curr_cols[col].get("data_type"),
                }
        prev_pk = sorted(prev_tables[table].get("primary_key", []) or [])
        curr_pk = sorted(curr_tables[table].get("primary_key", []) or [])
        table_changed = new_cols or gone_cols or type_changes or (prev_pk != curr_pk)
        if table_changed:
            added_columns[table] = new_cols
            removed_columns[table] = gone_cols
            datatype_changed[table] = type_changes
            pk_changed[table] = {"from": prev_pk, "to": curr_pk}
            drift["changed_tables"].append(table)
            drift["has_changes"] = True

    if added_columns:
        drift["columns_added"] = added_columns
    if removed_columns:
        drift["columns_removed"] = removed_columns
    if datatype_changed:
        drift["datatypes_changed"] = datatype_changed
    if pk_changed:
        drift["primary_keys_changed"] = pk_changed

    # FK structure: declared constraints added/removed.
    prev_declared = _edge_set(previous)
    curr_declared = _edge_set(current)
    declared_added = sorted(prev_declared - curr_declared, key=str)
    declared_removed = sorted(curr_declared - prev_declared, key=str)
    if declared_added:
        drift["declared_fk_removed"] = declared_added
        drift["has_changes"] = True
    if declared_removed:
        drift["declared_fk_added"] = declared_removed
        drift["has_changes"] = True

    # Inferred relationships added/removed.
    prev_inferred = _inferred_set(previous)
    curr_inferred = _inferred_set(current)
    inf_added = sorted(curr_inferred - prev_inferred, key=str)
    inf_removed = sorted(prev_inferred - curr_inferred, key=str)
    if inf_added:
        drift["relationships_inferred_added"] = inf_added
        drift["has_changes"] = True
    if inf_removed:
        drift["relationships_inferred_removed"] = inf_removed
        drift["has_changes"] = True

    return drift


def build_schema_mapping(conn, schema="public", db_type="postgresql", llm=None, database_name=None,
                         previous_mapping=None):
    tables = get_tables_and_columns(conn, schema, db_type)
    primary_keys = get_primary_keys(conn, schema, db_type)
    unique_keys = get_unique_keys(conn, schema, db_type)
    inferred_primary_keys = infer_primary_keys(tables, primary_keys)
    inferred_primary_keys.update(infer_primary_keys_from_data(
        conn, tables, schema, db_type,
        {**primary_keys, **inferred_primary_keys},
    ))
    declared_fks = get_declared_foreign_keys(conn, schema, db_type)
    null_stats = get_null_stats(conn, tables, schema, db_type)

    merged_primary_keys = {**primary_keys, **inferred_primary_keys}
    inferred_fks = infer_relationships(tables, merged_primary_keys, declared_fks,
                                       conn=conn, schema=schema, db_type=db_type,
                                       null_stats=null_stats)

    # Refinement: a column confirmed (by data) to reference another table is a
    # foreign key, not the table's own identity. Ruling those out can resolve
    # ambiguous key candidates (e.g. a warranty table where both claim_id and
    # sale_id are unique — sale_id provably points at sales, so claim_id wins).
    fk_sources = {(r["table"], r["column"]) for r in inferred_fks
                  if r.get("confidence") in ("data-confirmed", "data-inferred")}
    if fk_sources:
        refined = infer_primary_keys_from_data(
            conn, tables, schema, db_type,
            {**primary_keys, **inferred_primary_keys},
            exclude_columns=fk_sources,
        )
        if refined:
            inferred_primary_keys.update(refined)
            merged_primary_keys = {**primary_keys, **inferred_primary_keys}
            inferred_fks = infer_relationships(tables, merged_primary_keys, declared_fks,
                                               conn=conn, schema=schema, db_type=db_type)

    mapping = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "database": database_name or os.getenv("DB_NAME") or "unknown_database",
        "schema": schema,
        "tables": {},
        "unique_keys": unique_keys,
        "declared_relationships": declared_fks,
        "inferred_relationships": inferred_fks,
        "relationship_edges": [],
    }

    for rel in declared_fks:
        mapping["relationship_edges"].append({
            "source_table": rel["table_name"],
            "source_column": rel["column_name"],
            "target_table": rel["references_table"],
            "target_column": rel["references_column"],
            "type": "declared",
        })

    for rel in inferred_fks:
        mapping["relationship_edges"].append({
            "source_table": rel["table"],
            "source_column": rel["column"],
            "target_table": rel["references_table"],
            "target_column": rel["references_column"],
            "type": "inferred",
            "confidence": rel.get("confidence", "heuristic"),
            "confidence_score": rel.get("confidence_score"),
            "review_status": rel.get("review_status"),
            "ambiguous": rel.get("ambiguous"),
            "self_referencing": rel.get("self_referencing"),
            "cardinality": rel.get("cardinality"),
        })

    for table_name, columns in tables.items():
        table_primary_key = primary_keys.get(table_name, inferred_primary_keys.get(table_name, []))
        mapping["tables"][table_name] = {
            "columns": columns,
            "primary_key": table_primary_key,
            "inferred_primary_key": inferred_primary_keys.get(table_name, []),
            "unique_keys": unique_keys.get(table_name, []),
            "null_stats": null_stats.get(table_name, {}),
        }

    # Track all tables discovered in the DB (before dropping empties).
    all_db_tables = set(tables.keys())

    # Empty tables (0 rows) stay in the mapping with their structure intact so
    # the schema is fully described, but they are excluded from relationship
    # inference (a key built on no data cannot be confirmed).
    empty_tables = {
        t for t, col_stats in null_stats.items()
        if col_stats and all(s.get("total_rows", 1) == 0 for s in col_stats.values())
    }
    if empty_tables:
        for t in empty_tables:
            if t in mapping["tables"]:
                mapping["tables"][t]["empty"] = True
                mapping["tables"][t]["row_count"] = 0
        mapping["declared_relationships"] = [
            r for r in mapping["declared_relationships"]
            if r["table_name"] not in empty_tables and r["references_table"] not in empty_tables
        ]
        mapping["inferred_relationships"] = [
            r for r in mapping["inferred_relationships"]
            if r.get("table") not in empty_tables and r.get("references_table") not in empty_tables
        ]
        mapping["relationship_edges"] = [
            e for e in mapping["relationship_edges"]
            if e["source_table"] not in empty_tables and e["target_table"] not in empty_tables
        ]

    # Build summary metadata for the API response.
    mapping["summary"] = {
        "total_db_tables": len(all_db_tables),
        "tables_used": len(mapping["tables"]),
        "empty_tables": sorted(empty_tables),
        "declared_fk_count": len(mapping["declared_relationships"]),
        "inferred_fk_count": len(mapping["inferred_relationships"]),
        "edge_count": len(mapping["relationship_edges"]),
        "tables_with_pk": sorted(
            t for t in mapping["tables"]
            if mapping["tables"][t].get("primary_key")
        ),
        "tables_without_pk": sorted(
            t for t in mapping["tables"]
            if not mapping["tables"][t].get("primary_key")
        ),
    }

    if llm is not None:
        mapping = enrich_with_llm(mapping, llm)

    if mapping.get("tables"):
        _drop_empty_key_columns(mapping, conn, schema, db_type)

    _canonicalize_mapping(mapping)

    # Item 9: build the explicit relationship graph (per-table outgoing and
    # incoming relationships) ARIA can consume directly for SQL generation /
    # RAG / query planning. Also detects many-to-many via junction tables
    # (item 2).
    mapping["relationship_graph"] = _build_relationship_graph(mapping)

    # Rich health report (drives the UI dashboard / QA checks).
    inferred = mapping.get("inferred_relationships", [])
    strong = [r for r in inferred if r.get("confidence") in
              ("data-confirmed", "llm-confirmed", "composite-data-confirmed")]
    weak = [r for r in inferred if r.get("confidence") not in
            ("data-confirmed", "llm-confirmed", "composite-data-confirmed")]
    # Item 10: human-review list for uncertain relationships.
    review_list = [
        {
            "table": r.get("table"),
            "column": r.get("column"),
            "references_table": r.get("references_table"),
            "references_column": r.get("references_column"),
            "confidence_score": r.get("confidence_score"),
            "confidence": r.get("confidence"),
            "evidence": r.get("evidence", []),
        }
        for r in inferred if r.get("review_status") in ("flagged", "review")
    ]
    mapping["summary"]["health"] = {
        "strong_inferred_fk_count": len(strong),
        "weak_inferred_fk_count": len(weak),
        "empty_table_count": len(empty_tables),
        "profile_budget_exhausted": False,
        "warning": None,
    }
    mapping["summary"]["review_list"] = review_list
    mapping["summary"]["review_count"] = len(review_list)

    # Item 5: schema drift vs the previous run of this database (when supplied).
    if previous_mapping:
        mapping["drift"] = detect_schema_drift(previous_mapping, mapping)

    return mapping


SNAPSHOT_KEEP = 20  # max snapshot files retained per (user, database) file set


def _prune_snapshots(directory, stem, suffix, keep=SNAPSHOT_KEEP):
    """Delete the oldest snapshot files for a file set, keeping the newest `keep`.

    Never touches the base file ({stem}{suffix}) or the {stem}_latest alias.
    """
    try:
        snapshots = [
            p for p in Path(directory).glob(f"{stem}_*.json")
            if p.name != f"{stem}_latest{suffix}"
        ]
        if len(snapshots) <= keep:
            return
        snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in snapshots[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass


def get_output_paths(output_path, db_name=None, snapshot_id=None):
    """Return the base output, a unique snapshot, the latest alias, and the snapshot id.

    Every database (when `db_name` is given) gets its own file set:

        schema_mapping_<db>.json          - canonical base (overwritten each run)
        schema_mapping_<db>_latest.json   - stable pointer for the active session
        schema_mapping_<db>_<id>.json     - unique snapshot, never overwritten

    The snapshot file name embeds the same `snapshot_id` that the mapping JSON
    carries, so the JSON and its file are always correlated. Old snapshots beyond
    SNAPSHOT_KEEP are pruned. Returns (base, snapshot, latest, snapshot_id).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    stem, suffix = output.stem, output.suffix or ".json"

    if db_name:
        safe_db = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(db_name))[:64] or "db"
        stem = f"{stem}_{safe_db}"

    if not snapshot_id:
        snapshot_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    # Guarantee a filename that does not already exist, so a snapshot is never
    # overwritten (unique within and across users, DBs, and processes).
    candidate = f"{stem}_{snapshot_id}{suffix}"
    counter = 1
    while (output.parent / candidate).exists():
        candidate = f"{stem}_{snapshot_id}_{counter}{suffix}"
        counter += 1

    base = output.with_name(f"{stem}{suffix}")
    snapshot = output.with_name(candidate)
    latest = output.with_name(f"{stem}_latest{suffix}")

    _prune_snapshots(output.parent, stem, suffix)
    used_id = candidate[len(stem) + 1:-len(suffix)]
    return base, snapshot, latest, used_id


# ---------------------------------------------------------------------------
# File source handling (semi-structured path: CSV / tabular PDF)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ARIA Schema Agent - relational database schema understanding")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--db", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--db-type", choices=["postgresql", "mysql"], default="postgresql")
    parser.add_argument("--output", default=str(SCHEMA_DIR / "schema_mapping.json"))
    parser.add_argument(
        "--provider", choices=["local", "cloud"], default=None,
        help="Enable LLM-assisted reasoning over the extracted schema via this backend "
             "(local = Ollama, cloud = Groq). "
             "When omitted, only deterministic heuristics are used.",
    )
    args = parser.parse_args()

    # MySQL has no named schemas separate from the database: its
    # information_schema uses the database name as the "schema". Resolve the
    # filter the same way the API route does, so `--db-type mysql --db X`
    # works out of the box instead of silently returning zero tables.
    schema = args.schema or (args.db if args.db_type == "mysql" else "public")

    llm = None
    if args.provider:
        llm = create_provider(provider=args.provider)

    conn = get_connection(args)
    try:
        # PostgreSQL: if no schema was requested, auto-detect the one that
        # actually contains tables (public is the default but many databases
        # put their tables in app / core / dbo-equivalent schemas).
        if not args.schema and args.db_type != "mysql":
            schema = find_schema_with_tables(conn)
            print(f"Auto-detected schema '{schema}'.")
        print(f"Connected ({args.db_type}). Extracting schema for '{schema}' schema...")

        # Item 5: seed drift detection from the previous run of this database
        # (the {stem}_latest.json alias), if one exists.
        previous_mapping = None
        try:
            prev_base = Path(args.output)
            prev_stem = prev_base.stem
            if args.db:
                safe_db = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(args.db))[:64] or "db"
                prev_stem = f"{prev_stem}_{safe_db}"
            prev_latest = prev_base.with_name(f"{prev_stem}_latest{prev_base.suffix or '.json'}")
            if prev_latest.exists():
                with open(prev_latest, encoding="utf-8") as fh:
                    previous_mapping = json.load(fh)
        except Exception as exc:
            logger.debug("could not load previous schema for drift detection: %s", exc)

        mapping = build_schema_mapping(conn, schema, db_type=args.db_type, llm=llm,
                                       database_name=args.db,
                                       previous_mapping=previous_mapping)
        if not mapping.get("tables"):
            sys.exit(
                f"ERROR: no tables found in schema '{schema}' ({args.db_type}). "
                "The schema is empty or does not exist; nothing was mapped."
            )
        db_name = mapping.get("database", args.db or "unknown")
    finally:
        conn.close()

    output_path, timestamped_path, latest_path, snapshot_id = get_output_paths(
        args.output, db_name or None, snapshot_id=mapping.get("snapshot_id")
    )
    mapping["snapshot_id"] = snapshot_id

    for path in (output_path, timestamped_path, latest_path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

    n_tables = len(mapping["tables"])
    n_declared = len(mapping["declared_relationships"])
    n_inferred = len(mapping["inferred_relationships"])
    print(f"Done. {n_tables} tables, {n_declared} declared relationships, "
          f"{n_inferred} inferred relationships.")
    if mapping.get("llm_reasoning"):
        print(f"LLM-assisted reasoning applied via {mapping['llm_reasoning'].get('provider')} "
              f"(model {mapping['llm_reasoning'].get('model')}).")
    drift = mapping.get("drift")
    if drift:
        if drift.get("has_changes"):
            print(f"Schema drift detected: {len(drift.get('new_tables', []))} new table(s), "
                  f"{len(drift.get('removed_tables', []))} removed, "
                  f"{len(drift.get('columns_added', {}))} table(s) with column changes, "
                  f"{len(drift.get('relationships_inferred_added', []))} new inferred relationship(s).")
        else:
            print("Schema drift: no changes since the previous run.")
    print(f"Written to {output_path}")
    print(f"Saved fresh snapshot to {timestamped_path}")
    print(f"Updated latest copy at {latest_path}")


if __name__ == "__main__":
    main()
