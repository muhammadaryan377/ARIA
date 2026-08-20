"""
ARIA Goal Agent
---------------
Role: KPI Alignment

Runs per user query. Reads schema_mapping.json (produced by the Schema Agent),
interprets the user's plain-English business goal, maps it to KPIs and data
dimensions, determines the join path across the relationship graph, generates
a SQL query plan (SQLCoder locally via Ollama, or Llama via Groq), executes it,
and cleans/handles missing values.

LLM backend is selected by the user at startup:
    local - Ollama (private, offline, slower)
    cloud - Groq API (fast, but data leaves the machine)

Output: processed_data.json

Usage:
    python goal_agent.py "Show total sales by customer"
"""

import json
import re
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from llm_provider import LLMProvider, create_provider
from core.validation import unknown_sql_tables, empty_sql_tables, unknown_sql_columns
from core.config import SCHEMA_DIR, BASE_DIR

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = None
    StateGraph = None

logging.basicConfig(level=logging.INFO)

DEFAULT_DB_URI = (
    "postgresql://{user}:{password}@{host}:{port}/{dbname}"
).format(
    user=__import__("os").getenv("DB_USER", "postgres"),
    password=__import__("os").getenv("DB_PASSWORD", "postgres"),
    host=__import__("os").getenv("DB_HOST", "localhost"),
    port=__import__("os").getenv("DB_PORT", "5432"),
    dbname=__import__("os").getenv("DB_NAME", "postgres"),
)


class GoalAgent:
    def __init__(self, schema_json_path=None, db_uri=None,
                 provider=None, dialect="postgresql"):
        if schema_json_path:
            schema_file = Path(schema_json_path)
        else:
            # Absolute project-relative default so the process CWD never matters.
            schema_file = SCHEMA_DIR / "schema_mapping_latest.json"
        if not schema_file.exists():
            # Absolute project-relative fallback so the process CWD never matters.
            fallback = BASE_DIR / "schema_mapping.json"
            if fallback.exists():
                schema_file = fallback
        if not schema_file.exists():
            available = sorted(p.name for p in SCHEMA_DIR.glob("schema_mapping_*.json"))
            raise FileNotFoundError(
                f"Schema mapping not found: {schema_file}. "
                + (f"Available mappings: {', '.join(available)}."
                   if available
                   else "Run the schema extraction first (python schema_agent.py <database>).")
            )
        with open(schema_file, "r", encoding="utf-8") as f:
            self.full_schema = json.load(f)

        self.tables = self._normalize_schema(self.full_schema)
        self.relationship_graph = self._build_relationship_graph()
        self.engine = create_engine(db_uri or DEFAULT_DB_URI)
        self.llm = provider or create_provider()
        self.kpi_index = self._build_kpi_index()
        self.dialect = dialect
        self._row_counts_cache = None
        self.preprocessing_report = None

    # ------------------------------------------------------------------
    # Schema mapping normalization
    # ------------------------------------------------------------------

    def _normalize_schema(self, raw_schema):
        raw_tables = raw_schema.get("tables", {})
        if not isinstance(raw_tables, dict) or not raw_tables:
            available = sorted(p.name for p in SCHEMA_DIR.glob("schema_mapping_*.json"))
            raise ValueError(
                "Schema mapping has no tables (stale or hand-edited JSON?). "
                + (f"Available mappings: {', '.join(available)}."
                   if available
                   else "Re-run the schema extraction for this database.")
            )

        tables = {}
        all_edges = raw_schema.get("relationship_edges", [])
        declared = raw_schema.get("declared_relationships", [])
        inferred = raw_schema.get("inferred_relationships", [])
        known_tables = set(raw_tables)

        def add_fk(table_name, source_column, target_table, target_column, rel_type, confidence=None):
            tables.setdefault(table_name, {
                "columns": {}, "primary_key": [], "foreign_keys": [],
            })
            tables[table_name]["foreign_keys"].append({
                "column": source_column,
                "referenced_table": target_table,
                "referenced_column": target_column,
                "type": rel_type,
                "confidence": confidence,
            })

        for table_name, info in raw_schema.get("tables", {}).items():
            columns = info.get("columns", [])
            col_map = {}
            if isinstance(columns, list):
                for col in columns:
                    if isinstance(col, dict):
                        name = col.get("column")
                        col_map[name] = {
                            "data_type": col.get("data_type", "TEXT"),
                            "nullable": col.get("nullable", True),
                        }
                    else:
                        col_map[str(col)] = {"data_type": "TEXT", "nullable": True}
            else:
                col_map = columns

            pk = info.get("primary_key", []) or info.get("inferred_primary_key", []) or []
            for key in pk:
                if key in col_map:
                    col_map[key]["is_primary_key"] = True

            tables[table_name] = {"columns": col_map, "primary_key": pk, "foreign_keys": []}

        for edge in all_edges:
            add_fk(
                edge.get("source_table"), edge.get("source_column"),
                edge.get("target_table"), edge.get("target_column"),
                edge.get("type", "declared"), edge.get("confidence"),
            )
        for rel in declared:
            add_fk(rel.get("table_name"), rel.get("column_name"),
                   rel.get("references_table"), rel.get("references_column"),
                   "declared", None)
        for rel in inferred:
            add_fk(rel.get("table"), rel.get("column"),
                   rel.get("references_table"), rel.get("references_column"),
                   "inferred", rel.get("confidence"))

        for table_name, info in tables.items():
            for fk in info.get("foreign_keys", []):
                target = fk["referenced_table"]
                if target and target not in tables:
                    logging.warning(
                        "Schema mapping references unknown table '%s' from %s.%s "
                        "(stale/manually-edited JSON?)",
                        target, table_name, fk.get("column"),
                    )
                elif target and fk.get("referenced_column") not in tables[target]["columns"]:
                    logging.warning(
                        "Schema mapping references unknown column '%s.%s' from %s.%s",
                        target, fk.get("referenced_column"), table_name, fk.get("column"),
                    )

        return tables

    def _build_relationship_graph(self):
        graph = defaultdict(list)
        for table_name, info in self.tables.items():
            for fk in info.get("foreign_keys", []):
                graph[table_name].append(fk["referenced_table"])
        return graph

    # ------------------------------------------------------------------
    # KPI alignment: map plain-English intent to KPIs and dimensions
    # ------------------------------------------------------------------

    def _build_kpi_index(self):
        """Keyword -> (aggregate, description) map used for KPI alignment."""
        return {
            "total": ("SUM", "aggregate total"),
            "sum": ("SUM", "aggregate total"),
            "count": ("COUNT", "row count"),
            "how many": ("COUNT", "row count"),
            "average": ("AVG", "average value"),
            "avg": ("AVG", "average value"),
            "mean": ("AVG", "average value"),
            "max": ("MAX", "maximum value"),
            "maximum": ("MAX", "maximum value"),
            "min": ("MIN", "minimum value"),
            "minimum": ("MIN", "minimum value"),
            "profit": ("SUM", "profit KPI"),
            "revenue": ("SUM", "revenue KPI"),
            "sales": ("SUM", "sales KPI"),
            "growth": ("AVG", "growth KPI"),
        }

    def _normalize_tokens(self, text):
        tokens = set()
        for word in re.findall(r"[a-zA-Z0-9_]+", str(text)):
            tokens.add(word.lower())
            tokens.update(part for part in word.split("_") if part)
        return {t for t in tokens if t}

    def map_goal_to_kpi(self, user_goal):
        """Return detected KPIs + dimensions for a user goal."""
        goal_lower = user_goal.lower()
        kpis = []
        dimensions = set()

        for keyword, (agg, label) in self.kpi_index.items():
            if keyword in goal_lower:
                kpis.append({"aggregate": agg, "description": label, "match": keyword})
        if not kpis:
            kpis = [{"aggregate": "SUM", "description": "general KPI", "match": None}]

        for table_name in self.tables:
            if table_name.lower() in goal_lower:
                dimensions.add(table_name)

        return {"kpis": kpis, "dimensions": sorted(dimensions)}

    # ------------------------------------------------------------------
    # Relevant table selection + join path
    # ------------------------------------------------------------------

    def _table_keyword_score(self, table_name, info, user_goal_tokens):
        # Token-overlap scoring with singular/plural awareness.
        name_tokens = self._normalize_tokens(table_name)
        score = 0
        name_hit = False
        for tok in user_goal_tokens:
            if tok in name_tokens:
                score += 40
                name_hit = True
            elif tok.endswith("s") and tok[:-1] in name_tokens:
                score += 32
                name_hit = True
            elif name_tokens and (tok + "s") in name_tokens:
                score += 32
                name_hit = True
        for column_name in info.get("columns", {}).keys():
            col_tokens = self._normalize_tokens(column_name)
            for tok in user_goal_tokens:
                if tok in col_tokens:
                    score += 10 if name_hit else 14
                elif tok.endswith("s") and tok[:-1] in col_tokens:
                    score += 6
        score += len(info.get("foreign_keys", [])) * 2
        # Penalty for junction/lookup tables that match only via FK columns.
        if name_hit:
            score += 0
        return score

    def _expand_with_neighbors(self, selected, limit=4):
        """Expand the selected tables through FK neighbors, depth-bounded so
        unrelated tables further out in the graph never get pulled in."""
        expanded = set(selected)
        queue = [(t, 0) for t in selected]
        visited = set()
        while queue and len(expanded) < limit * 2:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if depth >= 2:
                continue
            for neighbor in self.relationship_graph.get(current, []):
                if neighbor in self.tables and neighbor not in expanded:
                    expanded.add(neighbor)
                    queue.append((neighbor, depth + 1))
        return list(expanded)

    def _get_relevant_tables(self, user_goal, max_tables=3):
        goal_tokens = self._normalize_tokens(user_goal)
        if not goal_tokens:
            goal_tokens = {"data", "summary", "report"}

        ranked = []
        for table_name, info in self.tables.items():
            score = self._table_keyword_score(table_name, info, goal_tokens)
            ranked.append((table_name, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        selected = [name for name, _ in ranked if _ > 0][:max_tables]

        if not selected:
            selected = [list(self.tables.keys())[0]] if self.tables else []

        if len(selected) < max_tables:
            selected = self._expand_with_neighbors(selected, max_tables)

        return selected[:max_tables]

    def _find_join_hops(self, start, target):
        """BFS join path between two tables using the relationship graph."""
        if start == target:
            return [start]
        queue = [[start]]
        visited = {start}
        while queue:
            path = queue.pop(0)
            for neighbor in self.relationship_graph.get(path[-1], []):
                if neighbor not in self.tables:
                    continue
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def _determine_join_path(self, relevant_tables):
        """
        Return the ordered join path that connects all relevant tables, using
        the schema's relationship graph. Falls back to the table list itself.
        """
        if not relevant_tables:
            return []
        path = [relevant_tables[0]]
        remaining = relevant_tables[1:]
        while remaining:
            connected = None
            best_local = None
            for table in remaining:
                hop = self._find_join_hops(path[-1], table)
                if hop:
                    best_local = hop
                    connected = table
                    break
            if best_local:
                for node in best_local[1:]:
                    if node not in path:
                        path.append(node)
                remaining = [t for t in remaining if t != connected]
            else:
                # No path found; append remaining tables in order as a fallback.
                path.extend([t for t in remaining if t not in path])
                break
        return path

    # ------------------------------------------------------------------
    # SQL generation / execution
    # ------------------------------------------------------------------

    def _build_schema_ddl(self, join_path, full=False):
        """Build DDL for the tables in the join path.

        When full=True the DDL also includes every other table in the schema
        (FK-connected first), so the LLM can discover the metric tables it
        needs even when the keyword scoring missed them. The join-path tables
        are always listed first and marked as preferred.
        """
        relevant = [t for t in join_path if t in self.tables]
        logging.info(f"Relevant tables (join path): {relevant}")

        tables = list(relevant)
        if full:
            remaining = [t for t in self.tables if t not in relevant]
            # Prefer tables that connect to the join path via foreign keys.
            connected = set()
            for t in relevant:
                for nb in self.relationship_graph.get(t, []):
                    if nb in remaining:
                        connected.add(nb)
            tables += [t for t in remaining if t in connected]
            tables += [t for t in remaining if t not in connected]

        ddl_parts = []
        for table_name in tables:
            info = self.tables[table_name]
            columns = info.get("columns", {})
            col_defs = []
            for column_name, col_info in list(columns.items())[:12]:
                data_type = col_info.get("data_type", "TEXT")
                is_pk = column_name in info.get("primary_key", []) or col_info.get("is_primary_key")
                suffix = " PRIMARY KEY" if is_pk else ""
                col_defs.append(f"    {column_name} {data_type}{suffix}")
            fk_lines = []
            for fk in info.get("foreign_keys", [])[:4]:
                fk_lines.append(
                    f"    FOREIGN KEY ({fk['column']}) REFERENCES {fk['referenced_table']}({fk['referenced_column']})"
                )
            all_defs = ",\n".join(col_defs + fk_lines)
            ddl_parts.append(f"CREATE TABLE {table_name} (\n{all_defs}\n);")
        return "\n\n".join(ddl_parts)

    def _clean_sql(self, raw_sql):
        if not raw_sql or not str(raw_sql).strip():
            return "SELECT 1;"
        cleaned = str(raw_sql).strip()
        cleaned = re.sub(r"```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*", "", cleaned)
        cleaned = cleaned.replace('"', "")
        # Small local models often ECHO the prompt's instruction sentences into
        # the answer ("with null values last. The final SQL statement is
        # provided without explanations or markdown; ..."). A real statement is
        # SELECT, or a genuine CTE (WITH <name> AS ( ... )). Plain English
        # "with ..." must NOT be mistaken for a WITH clause (it silently
        # corrupts the SQL), and the statement start must be the FIRST
        # top-level keyword: taking the last SELECT would capture inner
        # SELECTs from subqueries/CTEs and corrupt the query.
        stmt_pattern = re.compile(
            r"\bWITH\s+[A-Za-z_]\w*\s+AS\s*\(|\bSELECT\b|"
            r"\bINSERT\s+INTO\b|\bUPDATE\b|\bDELETE\s+FROM\b",
            re.IGNORECASE,
        )
        tail_pattern = re.compile(
            r"(?i)\b(the final sql statement|without explanations|no explanations|"
            r"here(?:'s| is) (?:the|your) final|output only[^\n]*|markdown|"
            r"with null values (?:last|first)[^\n]*|"
            r"this (?:query|sql) (?:is|uses|returns)[^\n]*)"
        )
        statements = []
        for fragment in cleaned.split(";"):
            match = stmt_pattern.search(fragment)
            if not match:
                continue
            frag = fragment[match.start():].strip()
            cut = tail_pattern.search(frag)
            if cut:
                frag = frag[:cut.start()].strip()
            if frag:
                statements.append(frag)
        if statements:
            return statements[-1].rstrip(";") + ";"
        # No real statement found: return the raw text so the validation layer /
        # repair loop sees it as broken and asks the model to fix it.
        return str(raw_sql).strip().replace('"', "") + ";"

    def _find_nested_aggregates(self, sql):
        """Return the nested-aggregate snippets found in sql, e.g. AVG(SUM(...)).

        Matches an aggregate keyword directly wrapping another aggregate keyword.
        """
        return re.findall(
            r"\b(AVG|SUM|COUNT|MIN|MAX)\s*\(\s*(?:DISTINCT\s+)?(AVG|SUM|COUNT|MIN|MAX)\s*\(",
            sql,
            re.IGNORECASE,
        )

    def _goal_asks_for_aggregation(self, user_goal):
        """True if the goal text clearly wants an aggregation / breakdown, in
        which case a bare `SELECT *` answer is almost certainly wrong."""
        goal = (user_goal or "").lower()
        markers = (
            "total", "count", "how many", "number of", "sum of", "average", "avg",
            "mean", "per ", "by ", "maximum", "minimum", "highest", "lowest",
            "top ", "share", "percentage", "breakdown", "distribution",
        )
        return any(m in goal for m in markers)

    # ------------------------------------------------------------------
    # Compound-extremes semantic guard
    # ------------------------------------------------------------------
    # A frequent, general failure of small SQL models: a goal like
    # "Which airlines have the highest AND lowest average prices?" is answered
    # with `... ORDER BY avg ASC NULLS LAST LIMIT 1` which returns only ONE
    # extreme. The SQL is syntactically valid, so no existing validation layer
    # catches it. This guard is deterministic and provider-agnostic: when the
    # goal asks for BOTH ends of a ranking, any single-sided `LIMIT 1` is
    # stripped so the full ranked set is returned and both extremes are present.
    _EXTREMES_MAX = {"highest", "largest", "most", "maximum", "max", "best", "top", "greatest", "biggest"}
    _EXTREMES_MIN = {"lowest", "smallest", "least", "minimum", "min", "worst", "bottom", "cheapest"}

    def _is_extremes_goal(self, user_goal):
        """True if the goal explicitly asks for BOTH the top and bottom of a
        ranking (e.g. 'highest and lowest', 'most and least', 'max and min',
        'best and worst', 'top and bottom')."""
        goal = re.sub(r"[^a-z\s]", " ", (user_goal or "").lower())
        words = set(goal.split())
        has_max = bool(words & self._EXTREMES_MAX)
        has_min = bool(words & self._EXTREMES_MIN)
        return has_max and has_min

    def _fix_extremes_sql(self, user_goal, sql):
        """If the goal asks for both extremes and `sql` is a single-sided
        `ORDER BY ... (ASC|DESC) ... LIMIT 1`, strip the trailing LIMIT so the
        full ranking comes back. Returns (sql, changed)."""
        if not sql or not self._is_extremes_goal(user_goal):
            return sql, False
        if not re.search(r"\bORDER\s+BY\b[\s\S]*\b(ASC|DESC)\b", sql, re.IGNORECASE):
            return sql, False
        new_sql, n = re.subn(
            r"\s+LIMIT\s+\d+(?:\s+OFFSET\s+\d+)?(\s*;?\s*)$",
            r"\1",
            sql.strip(),
            count=1,
            flags=re.IGNORECASE,
        )
        if n:
            logging.warning(
                "Extremes goal detected; stripped trailing LIMIT so both the "
                "highest and lowest appear in the result set."
            )
            return new_sql.strip(), True
        return sql, False

    # ------------------------------------------------------------------
    # Missing-FROM-clause guard
    # ------------------------------------------------------------------
    # A very common small-model mistake: the SQL references a table through a
    # qualified column (e.g. `bookings.price`, `bookings.passenger_id`) but the
    # table never appears in FROM/JOIN, which crashes with "missing FROM-clause
    # entry". This is a plain, database-independent grammar bug — catch it
    # deterministically before the DB round-trip and, when the schema has a
    # clear FK edge to a table already in the query, inject the JOIN ourselves
    # instead of asking the (weak) local model to do it.
    _SQL_KEYWORDS = {
        "on", "where", "group", "order", "having", "join", "left", "right",
        "inner", "outer", "cross", "full", "as", "using", "limit", "offset",
        "select", "from", "and", "or", "not", "by", "asc", "desc", "nulls",
        "first", "last", "case", "when", "then", "else", "end", "union", "all",
        "distinct", "top", "with", "values",
    }

    def _fk_edges(self):
        if getattr(self, "_fk_edges_cache", None) is None:
            edges = []
            for table_name, info in self.tables.items():
                for fk in info.get("foreign_keys", []):
                    edges.append((
                        table_name,
                        fk["column"],
                        fk["referenced_table"],
                        fk["referenced_column"],
                    ))
            self._fk_edges_cache = edges
        return self._fk_edges_cache

    def _sql_structure(self, sql):
        """Return a dict describing the FROM/JOIN structure of `sql`:

          - in_from:    set of table names present in FROM/JOIN clauses
          - alias_to_name, name_to_alias: alias mappings
          - first_ref:  reference token (alias or bare name) of the FIRST
                        FROM table — the only table guaranteed visible at the
                        point right after the FROM clause.
          - intro:      table name -> byte position just after the clause that
                        introduces it (end of the FROM/JOIN block).

        Implemented as a positional scan so a JOIN/ON keyword right after a
        table name is never mistaken for an alias.
        """
        out = {
            "in_from": set(),
            "alias_to_name": {},
            "name_to_alias": {},
            "first_ref": None,
            "intro": {},
        }
        pos = 0
        first = True
        while True:
            m = re.search(
                r"\b(?:FROM|JOIN)\s+([\"'`]?)([A-Za-z_][A-Za-z0-9_]*)\1\b",
                sql[pos:], re.IGNORECASE,
            )
            if not m:
                break
            table = m.group(2)
            out["in_from"].add(table)
            after = pos + m.end()
            alias = None
            tail = sql[after:]
            am = re.match(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
            if am:
                alias = am.group(1)
            else:
                am = re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
                if am and am.group(1).lower() not in self._SQL_KEYWORDS:
                    alias = am.group(1)
            if alias and alias.lower() not in self._SQL_KEYWORDS:
                out["alias_to_name"][alias] = table
                out["name_to_alias"][table] = alias
                after += am.end()
            if first:
                out["first_ref"] = out["name_to_alias"].get(table, table)
                first = False
            out["intro"][table] = self._clause_end(sql, after)
            pos = after
        return out

    def _clause_end(self, sql, start):
        """Byte position where the clause starting at `start` ends: the next
        top-level JOIN/WHERE/GROUP/ORDER/HAVING/LIMIT/OFFSET keyword or `;`."""
        depth = 0
        i = start
        n = len(sql)
        while i < n:
            ch = sql[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == ";":
                return i
            elif depth == 0:
                m = re.match(
                    r"\b(?:JOIN|WHERE|GROUP|ORDER|HAVING|LIMIT|OFFSET)\b",
                    sql[i:], re.IGNORECASE,
                )
                if m:
                    return i
            i += 1
        return n

    def _missing_refs(self, sql):
        """Return a list of {'table', 'ref'} for tables referenced via a
        qualified column but missing from FROM/JOIN. `ref` is the token the
        query actually uses for that table (an alias like `b`, or the bare
        table name). Unknown prefixes are resolved to the single schema table
        whose name starts with that prefix (the model invented an alias without
        declaring `FROM <table> <alias>`)."""
        if not sql:
            return []
        struct = self._sql_structure(sql)
        in_from_lc = {t.lower() for t in struct["in_from"]}
        if not in_from_lc:
            return []
        table_names_lc = {t.lower() for t in self.tables}
        found = {}
        for m in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*|\*)\b", sql):
            prefix = m.group(1).lower()
            if prefix in struct["alias_to_name"]:
                table = struct["alias_to_name"][prefix].lower()
            elif prefix in table_names_lc:
                table = prefix
            else:
                candidates = [t for t in self.tables if t.lower().startswith(prefix)]
                if len(candidates) != 1:
                    continue
                table = candidates[0].lower()
            if table not in in_from_lc and table in table_names_lc:
                found[table] = prefix
        return [{"table": t, "ref": r} for t, r in sorted(found.items())]

    def _find_missing_from_tables(self, sql):
        """Return table names that are referenced (via `tbl.col` or a JOIN ON
        condition) but never appear in a FROM/JOIN clause."""
        return [m["table"] for m in self._missing_refs(sql)]

    def _inject_missing_joins(self, sql, missing):
        """Deterministically inject JOINs for tables that are referenced but
        missing from FROM/JOIN. Only inject when there is a clear FK edge to the
        FIRST table in FROM — the one guaranteed to be visible at the insertion
        point (a table referenced in an earlier ON clause must be introduced
        before that clause). Returns (new_sql, injected_count) or (sql, 0)."""
        if not missing:
            return sql, 0
        struct = self._sql_structure(sql)
        first_ref = struct["first_ref"]
        intro_pos = struct["intro"]
        if first_ref is None or not intro_pos:
            return sql, 0
        first_table = struct["alias_to_name"].get(first_ref, first_ref)
        if first_table not in intro_pos:
            return sql, 0
        edges = self._fk_edges()
        joins = []
        for item in self._missing_refs(sql):
            table = item["table"]
            ref = item["ref"]
            real = next((t for t in self.tables if t.lower() == table), table)
            alias_txt = "" if ref == table else f" {ref}"
            candidate = None
            for (src, col, ref_t, ref_col) in edges:
                if src.lower() == table and ref_t.lower() == first_table.lower():
                    candidate = f"JOIN {real}{alias_txt} ON {ref}.{col} = {first_ref}.{ref_col}"
                    break
            if candidate is None:
                for (src, col, ref_t, ref_col) in edges:
                    if ref_t.lower() == table and src.lower() == first_table.lower():
                        candidate = f"JOIN {real}{alias_txt} ON {first_ref}.{col} = {ref}.{ref_col}"
                        break
            if candidate:
                joins.append(candidate)
        if not joins:
            return sql, 0
        inject_at = intro_pos[first_table]
        injection = " " + " ".join(joins)
        new_sql = sql[:inject_at] + injection + " " + sql[inject_at:]
        return new_sql, len(joins)

    def _join_clauses(self, sql):
        """Return (start, end, table, on_start, on_end) for every JOIN clause,
        where on_start/on_end span the ON condition (None if no ON clause)."""
        clauses = []
        pos = 0
        while True:
            m = re.search(
                r"\bJOIN\s+([\"'`]?)([A-Za-z_][A-Za-z0-9_]*)\1\b",
                sql[pos:], re.IGNORECASE,
            )
            if not m:
                break
            start = pos + m.start()
            table = m.group(2)
            after = pos + m.end()
            tail = sql[after:]
            am = re.match(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
            if not am:
                am = re.match(r"\s+([A-Za-z_][A-Za-z0-9_]*)\b", tail, re.IGNORECASE)
                if am and am.group(1).lower() in self._SQL_KEYWORDS:
                    am = None
            if am:
                after += am.end()
            om = re.search(r"\bON\b", sql[after:], re.IGNORECASE)
            if not om:
                end = self._clause_end(sql, after)
                clauses.append((start, end, table, None, None))
            else:
                cond_start = after + om.end()
                cond_end = self._clause_end(sql, cond_start)
                clauses.append((start, cond_end, table, cond_start, cond_end))
            pos = cond_end if om else end
        return clauses

    def _repair_unknown_columns(self, sql, columns_by_table, bad_columns):
        """Deterministically fix hallucinated column references WITHOUT an LLM.

        Handles two failure modes the LLM repair loop repeatedly fails on:
          1. `tbl.col` where `col` actually lives on another table in the query's
             schema -> rewrite the qualifier to the owning table (the missing-FROM
             guard then injects the required FK join).
          2. a JOIN whose ON condition uses an unresolvable column -> drop that
             hallucinated JOIN entirely (only when the joined table is not used
             anywhere else in the query).

        JOIN ON predicates whose rewrite would self-compare the same table
        (``rooms.room_id = rooms.room_id``) are refused and instead dropped or
        left for the LLM. Returns (repaired_sql, changed).
        """
        if not bad_columns or not sql:
            return sql, False
        bad_set = {b.lower() for b in bad_columns}
        table_cols_lc = {t: {c.lower() for c in cols} for t, cols in columns_by_table.items()}
        col_owners = {}
        for t, cols in table_cols_lc.items():
            for c in cols:
                col_owners.setdefault(c, []).append(t)
        struct = self._sql_structure(sql)
        in_from_lc = {t.lower() for t in struct["in_from"]}

        def resolve_prefix(prefix):
            p = prefix.lower()
            if p in struct["alias_to_name"]:
                return struct["alias_to_name"][p].lower()
            if p in table_cols_lc:
                return p
            cands = [t for t in table_cols_lc if t.startswith(p)]
            return cands[0].lower() if len(cands) == 1 else None

        def near_match(table, col):
            norm = re.sub(r"[^a-z0-9]", "", col.lower())
            for c in table_cols_lc.get(table, ()):
                if re.sub(r"[^a-z0-9]", "", c) == norm:
                    return c
            for c in table_cols_lc.get(table, ()):
                if len(c) >= 4 and (norm.rstrip("s") == c.rstrip("s") or norm == c[:-1]):
                    return c
            return None

        on_regions = [(s, e) for (_, _, _, s, e) in self._join_clauses(sql) if s is not None]

        def other_side_table(pos, on_start, on_end):
            """Resolve the table of the operand on the other side of the `=`
            nearest to `pos` within an ON condition."""
            seg = sql[on_start:on_end]
            local = pos - on_start
            eq = seg.rfind("=", 0, local)
            eq2 = seg.find("=", local)
            bounds = []
            if eq != -1:
                bounds.append((seg.rfind("=", 0, eq - 1), eq))
            if eq2 != -1:
                bounds.append((eq, seg.find("=", eq2 + 1)))
            region = None
            for lo, hi in bounds:
                if lo <= local <= hi:
                    region = seg[lo + 1:hi if hi != -1 else len(seg)]
                    break
            if not region:
                return None
            m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\b", region)
            return resolve_prefix(m.group(1)) if m else None

        def in_on(pos):
            return next(((s, e) for s, e in on_regions if s <= pos < e), None)

        # --- pass 1: rewrite qualifiers to the column's real owner -------------
        replacements = []
        for m in re.finditer(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*|\*)\b", sql
        ):
            prefix, col = m.group(1), m.group(2)
            if col == "*":
                continue
            tbl = resolve_prefix(prefix)
            if tbl is None or f"{tbl}.{col}".lower() not in bad_set:
                continue
            real = near_match(tbl, col)
            if real:
                replacements.append((m.start(), m.end(), f"{prefix}.{real}"))
                continue
            owners = [t for t in col_owners.get(col.lower(), []) if t != tbl and t not in in_from_lc]
            if len(owners) != 1:
                continue
            region = in_on(m.start())
            if region:
                other = other_side_table(m.start(), region[0], region[1])
                if other == owners[0]:
                    continue  # would become X.x = X.x (self-comparison); drop/LLM instead
            replacements.append((m.start(), m.end(), f"{owners[0]}.{col}"))
        if replacements:
            out = sql
            for start, end, repl in sorted(replacements, reverse=True):
                out = out[:start] + repl + out[end:]
            if not unknown_sql_columns(out, columns_by_table):
                return out, True

        # --- pass 2: drop hallucinated JOINs (unresolvable ON column) ----------
        for (start, end, table, on_start, on_end) in self._join_clauses(sql):
            if on_start is None:
                continue
            cond = sql[on_start:on_end]
            used_elsewhere = False
            alias = struct["alias_to_name"].get(table)
            for token in {table, alias} - {None}:
                remaining = sql[:start] + sql[end:]
                if re.search(rf"\b{re.escape(token)}\s*\.\s*[A-Za-z_][A-Za-z0-9_]*\b",
                             remaining, re.IGNORECASE):
                    used_elsewhere = True
                    break
            if used_elsewhere:
                continue
            cond_bad = [b for b in bad_set if re.search(
                rf"\b{re.escape(b.split('.')[0])}\s*\.\s*{re.escape(b.split('.')[1])}\b",
                cond, re.IGNORECASE)]
            if not cond_bad:
                continue
            out = sql[:start] + sql[end:]
            if not unknown_sql_columns(out, columns_by_table):
                return out.strip(), True
        return sql, False

    def _find_bare_select_star(self, sql):
        """Return a short description if `sql` is a bare `SELECT *` (no aggregate
        function, no GROUP BY). None otherwise. Never flags `SELECT t.*` that is
        accompanied by an aggregate."""
        m = re.search(r"\bselect\b([\s\S]*?)\bfrom\b", sql, re.IGNORECASE)
        if not m:
            return None
        select_part = m.group(1)
        if "*" not in select_part:
            return None
        has_agg = re.search(
            r"\b(?:count|sum|avg|average|min|max|stddev|std|variance|var_pop|var_samp|median|percentile_cont|percentile_disc)\s*\(",
            sql, re.IGNORECASE,
        )
        if has_agg:
            return None
        return "SELECT * with no aggregation"

    def _find_suspicious_pk_joins(self, sql):
        """Return equality join pairs that equate the PRIMARY KEYS of two
        DIFFERENT tables. In a normalized schema that is almost always a bug
        (e.g. `tracks.id = artists.id`); real joins go PK -> FK."""
        pk_of = {
            name: set(info.get("primary_key", []) or info.get("inferred_primary_key", []) or [])
            for name, info in self.tables.items()
        }

        # Resolve aliases to table names: FROM x [AS] a, JOIN y [AS] b.
        alias_to_table = {}
        for m in re.finditer(
            r"\b(FROM|JOIN)\s+([\"']?)(\w+)\2\s+(?:AS\s+)?([a-zA-Z_]\w*)",
            sql,
            re.IGNORECASE,
        ):
            table, alias = m.group(3), m.group(4)
            alias_to_table[alias.lower()] = table
            if not m.group(4):
                alias_to_table[table.lower()] = table

        suspects = []
        # Capture each ON clause and the equality conditions inside it.
        for on_m in re.finditer(
            r"\bJOIN\s+([\"']?)(\w+)\1\s+(?:AS\s+)?([a-zA-Z_]\w*)\s+ON\s+(.*?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT|;)|$)",
            sql,
            re.IGNORECASE,
        ):
            joined_table, joined_alias, on_clause = on_m.group(2), on_m.group(3).lower(), on_m.group(4)
            for eq in re.finditer(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", on_clause):
                a, ca, b, cb = eq.groups()
                ta = alias_to_table.get(a.lower()) or a
                tb = alias_to_table.get(b.lower()) or b
                if ta == tb:
                    continue
                if ca in pk_of.get(ta, set()) and cb in pk_of.get(tb, set()):
                    suspects.append(f"{ta}.{ca} = {tb}.{cb} (joins primary keys of unrelated tables)")
        return suspects

    def _find_fk_mismatch_joins(self, sql):
        """Return join equalities where an FK column is joined to a table it does
        NOT reference. E.g. tracks.album_id (FK -> albums.id) joined to artists.id
        is wrong: the FK definition says it must join albums, not artists."""
        # {table: {column: (referenced_table, referenced_column)}}
        fk_of = {}
        for name, info in self.tables.items():
            for fk in info.get("foreign_keys", []):
                fk_of.setdefault(name, {})[fk["column"]] = (fk["referenced_table"], fk["referenced_column"])

        alias_to_table = {}
        for m in re.finditer(
            r"\b(FROM|JOIN)\s+([\"']?)(\w+)\2\s+(?:AS\s+)?([a-zA-Z_]\w*)",
            sql,
            re.IGNORECASE,
        ):
            table, alias = m.group(3), m.group(4)
            alias_to_table[alias.lower()] = table
            if not m.group(4):
                alias_to_table[table.lower()] = table

        suspects = []
        for on_m in re.finditer(
            r"\bJOIN\s+([\"']?)(\w+)\1\s+(?:AS\s+)?([a-zA-Z_]\w*)\s+ON\s+(.*?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|LIMIT|;)|$)",
            sql,
            re.IGNORECASE,
        ):
            on_clause = on_m.group(4)
            for eq in re.finditer(r"(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)", on_clause):
                a, ca, b, cb = eq.groups()
                ta = alias_to_table.get(a.lower()) or a
                tb = alias_to_table.get(b.lower()) or b
                if ta == tb:
                    continue
                if ca in fk_of.get(ta, {}):
                    ref_table, ref_col = fk_of[ta][ca]
                    if ref_table != tb or ref_col != cb:
                        suspects.append(
                            f"{ta}.{ca} references {ref_table}.{ref_col} but is joined to {tb}.{cb}"
                        )
                elif cb in fk_of.get(tb, {}):
                    ref_table, ref_col = fk_of[tb][cb]
                    if ref_table != ta or ref_col != ca:
                        suspects.append(
                            f"{tb}.{cb} references {ref_table}.{ref_col} but is joined to {ta}.{ca}"
                        )
        return suspects

    def _llm_unavailable(self, exc):
        """True when the language-model provider is down or rate-limited, as
        opposed to a genuine SQL-generation problem we could repair."""
        msg = str(exc).lower()
        if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
            return True
        if any(k in msg for k in ("connection", "timed out", "timeout", "connect", "refused")):
            return True
        if "ollama" in msg and ("down" in msg or "refused" in msg or "connect" in msg):
            return True
        return False

    def _generate_sql(self, user_goal, join_path, kpi_map):
        schema_ddl = self._build_schema_ddl(join_path, full=False)
        if not schema_ddl:
            return "SELECT 1;"

        kpi_desc = ", ".join(k["description"] for k in kpi_map["kpis"]) or "none"

        if self.dialect == "sqlite":
            dialect_note = (
                "### SQLite notes\n"
                "Use standard SQLite syntax: strftime('%Y', col) / strftime('%Y-%m', col) "
                "for dates instead of EXTRACT, and || for concatenation.\n"
            )
        elif self.dialect == "mysql":
            dialect_note = (
                "### MySQL notes\n"
                "Quote identifiers with backticks when needed. Use DATE_FORMAT(col, '%Y') / "
                "DATE_FORMAT(col, '%Y-%m') for date grouping instead of EXTRACT/strftime. "
                "Use NOW()/CURDATE() for current time, LIKE (there is no ILIKE), LIMIT for "
                "paging, and CONCAT() instead of ||. Backslash is the string escape character.\n"
            )
        else:
            dialect_note = ""

        prompt = f"""
### Task
Generate a single {self.dialect} query that answers the user's business goal.
User goal: {user_goal}
Aligned KPIs: {kpi_desc}
Dimensions: {', '.join(kpi_map['dimensions']) if kpi_map['dimensions'] else 'auto-detect'}

### Database Schema
The query will run on a database with the following schema:
{schema_ddl}
Preferred starting tables (inspect these first, but you may use any table
needed to answer the goal, joining via the FOREIGN KEY relationships above):
{join_path}
{dialect_note}### SQL
Output only the final SQL statement. No explanations, no markdown, nothing after it.
Rules:
- Aggregate functions must NEVER be nested (AVG(SUM(x)), SUM(COUNT(x)), etc. are invalid SQL).
- An average per group is SUM(x) / COUNT(DISTINCT key) or a subquery.
- If the goal asks for BOTH extremes (e.g. "highest AND lowest", "most and least",
  "top and bottom", "max and min", "best and worst"), do NOT use LIMIT 1: return the
  full ranked list (ORDER BY ... ASC or DESC, no LIMIT) so both ends are present.
- If the goal asks for a single extreme ("the highest", "the lowest", "top 5"), a
  one-sided ORDER BY + LIMIT is fine.
- Do not add a trailing semicolon only statement marker; a single trailing semicolon is fine.
"""
        try:
            raw = self.llm.complete("sql", prompt, temperature=0.1, num_predict=500)
            return self._clean_sql(raw)
        except Exception as exc:
            logging.warning(f"LLM SQL generation failed: {exc}.")
            if self._llm_unavailable(exc):
                # Provider down/exhausted: do NOT fall back to a bare SELECT *
                # (that would only trigger pointless repair loops). Signal the
                # caller to degrade gracefully.
                return None
            return self._fallback_sql(join_path)

    def _fallback_sql(self, join_path):
        relevant = [t for t in join_path if t in self.tables]
        if not relevant:
            return "SELECT * FROM information_schema.tables LIMIT 10;"
        return f"SELECT * FROM {relevant[0]} LIMIT 50;"

    def _row_counts(self):
        """Count rows per table (cached per agent) so a genuinely empty table
        can be told apart from a faulty query."""
        if self._row_counts_cache is not None:
            return self._row_counts_cache
        counts = {}
        try:
            with self.engine.connect() as conn:
                for table_name in self.tables:
                    try:
                        result = conn.execute(text(f'SELECT count(*) FROM "{table_name}"'))
                        counts[table_name] = int(result.scalar() or 0)
                    except Exception:
                        counts[table_name] = -1
        except Exception:
            pass
        self._row_counts_cache = counts
        return counts

    def _execute_with_retry(self, user_goal, join_path, kpi_map, raw_sql, retries=2):
        """Execute the SQL with a validation layer + LLM repair.

        Returns (records, final_sql, note). note is None on success, or a dict
        {"message": str} explaining a graceful empty outcome so the UI never
        shows a bare "no data" for a valid-but-empty query.
        """
        if raw_sql is None:
            # raw_sql was None -> the LLM provider is unavailable; degrade
            # gracefully instead of crash-looping through repairs.
            return [], None, {
                "message": (
                    "The language model is temporarily unavailable or rate-limited, "
                    "so this goal could not be answered right now. Please try again "
                    "shortly or switch the model provider."
                )
            }
        current_sql = self._clean_sql(raw_sql)
        columns_by_table = {
            name: list(info["columns"])
            for name, info in self.tables.items()
        }
        for attempt in range(retries + 1):
            # Validation layer: reject hallucinated table names before hitting
            # the database (e.g. `order_items` when the real table is
            # `order_details`). Provider-agnostic.
            bad_tables = unknown_sql_tables(current_sql, self.tables)
            if bad_tables:
                if attempt >= retries:
                    raise RuntimeError(
                        f"Generated SQL referenced tables that do not exist "
                        f"({', '.join(bad_tables)}) and could not be repaired."
                    )
                logging.warning(
                    f"SQL references unknown table(s) {bad_tables} on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below references table(s) that DO NOT EXIST in the schema: {', '.join(bad_tables)}.
You MUST only use the exact table names listed in the schema below.

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query using real tables from the schema.
No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    raise RuntimeError(
                        f"Generated SQL referenced tables that do not exist "
                        f"({', '.join(bad_tables)}) and could not be repaired."
                    )

            # Validation layer: reject hallucinated COLUMN references before a
            # database round-trip (e.g. `orders.quantity` when quantity lives
            # on `order_details`). Deterministic + provider-agnostic.
            bad_columns = unknown_sql_columns(current_sql, columns_by_table)
            if bad_columns:
                repaired, changed = self._repair_unknown_columns(
                    current_sql, columns_by_table, bad_columns
                )
                if changed:
                    logging.warning(
                        f"SQL referenced unknown column(s) {bad_columns}; "
                        f"repaired deterministically (no LLM)."
                    )
                    current_sql = repaired
                    continue
                if attempt >= retries:
                    raise RuntimeError(
                        f"Generated SQL referenced columns that do not exist "
                        f"({', '.join(bad_columns)}) and could not be repaired."
                    )
                logging.warning(
                    f"SQL references unknown column(s) {bad_columns} on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below references column(s) that DO NOT EXIST in the schema: {', '.join(bad_columns)}.
Use only the real columns listed under each table in the schema below.

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query using real columns from the schema.
No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    raise RuntimeError(
                        f"Generated SQL referenced columns that do not exist "
                        f"({', '.join(bad_columns)}) and could not be repaired."
                    )

            # Validation layer: reject SQL that references a table (as `tbl.col`
            # or in a JOIN ON) without that table in FROM/JOIN. This is a plain,
            # database-independent grammar bug ("missing FROM-clause entry"). When
            # the schema has a clear FK edge we inject the correct JOIN ourselves
            # (deterministic, no LLM); otherwise ask the LLM to repair.
            missing = self._find_missing_from_tables(current_sql)
            if missing:
                auto_sql, injected = self._inject_missing_joins(current_sql, missing)
                if injected:
                    logging.warning(
                        f"SQL referenced {missing} without a FROM clause; injected "
                        f"{injected} deterministic FK JOIN(s)."
                    )
                    current_sql = auto_sql
                    continue
                if attempt >= retries:
                    raise RuntimeError(
                        f"Generated SQL referenced table(s) {missing} without including "
                        f"them in FROM/JOIN and could not be repaired."
                    )
                logging.warning(
                    f"SQL references {missing} without a FROM clause on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below references the table(s) {missing} (e.g. `{missing[0]}.column`) but
never includes them in the FROM or JOIN clauses, which crashes with
"missing FROM-clause entry". Add each missing table to the FROM/JOIN using the
correct FOREIGN KEY join from the schema (child.fk_id = parent.id).

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    raise RuntimeError(
                        f"Generated SQL referenced table(s) {missing} without including "
                        f"them in FROM/JOIN and could not be repaired."
                    )

            # Validation layer: reject nested aggregate functions before a
            # database round-trip. AVG(SUM(x)) etc. are invalid in PostgreSQL
            # and this is a frequent LLM mistake that would otherwise waste a
            # full execution + repair cycle.
            nested = self._find_nested_aggregates(current_sql)
            if nested:
                if attempt >= retries:
                    raise RuntimeError(
                        f"Generated SQL nested aggregate function(s) "
                        f"({', '.join(nested)}) and could not be repaired."
                    )
                logging.warning(
                    f"SQL nests aggregate function(s) {nested} on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below nests aggregate functions like {', '.join(nested)} (e.g. AVG(SUM(x))),
which is invalid SQL in PostgreSQL.
To compute an average per group use SUM(x) / COUNT(DISTINCT key), or move the outer
aggregate into a subquery (SELECT AVG(v) FROM (SELECT SUM(x) AS v FROM ... GROUP BY ...) t).

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    raise RuntimeError(
                        f"Generated SQL nested aggregate function(s) "
                        f"({', '.join(nested)}) and could not be repaired."
                    )

            # Validation layer: reject joins that equate the PRIMARY KEYS of two
            # different tables, or that join an FK column to a table it does not
            # actually reference. In a normalized schema both are wrong
            # correlations (e.g. tracks.id = artists.id, tracks.album_id = artists.id);
            # real joins go PK -> FK through the declared relationships.
            bad_pk_joins = self._find_suspicious_pk_joins(current_sql)
            bad_fk_joins = self._find_fk_mismatch_joins(current_sql)
            bad_joins = bad_pk_joins + bad_fk_joins
            if bad_joins:
                if attempt >= retries:
                    raise RuntimeError(
                        f"Generated SQL joins unrelated tables "
                        f"({', '.join(bad_joins)}) and could not be repaired."
                    )
                logging.warning(
                    f"SQL joins unrelated tables {bad_joins} on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below contains broken JOIN conditions: {', '.join(bad_joins)}.
Two different tables having an equal column name does NOT make them related,
and an FK column must be joined to the exact table it references.
Join ONLY through the actual FOREIGN KEY relationships in the schema (e.g. child.fk_id = parent.id).

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query using real FK relationships. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    raise RuntimeError(
                        f"Generated SQL joins unrelated tables "
                        f"({', '.join(bad_joins)}) and could not be repaired."
                    )

            # Validation layer: reject a bare `SELECT *` when the goal asks for
            # an aggregation. `SELECT * FROM plays LIMIT 50` is valid SQL but does
            # not answer "total play count per genre" — better to repair than to
            # chart the raw ID columns. Deterministic + provider-agnostic.
            if self._goal_asks_for_aggregation(user_goal) and self._find_bare_select_star(current_sql):
                if attempt >= retries:
                    raise RuntimeError(
                        "Generated SQL is a plain `SELECT *` that does not aggregate the "
                        "data the goal asks for, and could not be repaired."
                    )
                logging.warning(
                    f"SQL is a bare SELECT * for an aggregating goal on attempt {attempt + 1}. "
                    f"Asking LLM to repair."
                )
                fix_prompt = f"""
The SQL below is a plain `SELECT *` (no aggregate functions, no GROUP BY), but the
user's goal clearly asks for an aggregation or breakdown (total, count, average,
per group, etc.). Rewrite it as an aggregated query that answers the goal using the
real columns in the schema (e.g. SELECT t.col, COUNT(...) ... GROUP BY t.col).

User goal:
{user_goal}

Schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500)
                    current_sql = self._clean_sql(raw)
                    continue
                except Exception:
                    raise RuntimeError(
                        "Generated SQL is a plain `SELECT *` that does not aggregate the "
                        "data the goal asks for, and could not be repaired."
                    )

            # Semantic guard: a compound-extremes goal ("highest and lowest",
            # "most and least", ...) must NOT be answered with a single LIMIT 1
            # row. Strip a single-sided trailing LIMIT so the full ranked set is
            # returned (both extremes present). Deterministic + provider-agnostic.
            current_sql, _changed = self._fix_extremes_sql(user_goal, current_sql)

            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text(current_sql))
                    rows = result.fetchall()
                    cols = list(result.keys())
                df = pd.DataFrame(rows, columns=cols)
                if df.empty:
                    empty_tabs = empty_sql_tables(current_sql, self._row_counts())
                    if empty_tabs:
                        # The data genuinely is not there; repairing is pointless.
                        names = ", ".join(sorted(empty_tabs))
                        return [], current_sql, {
                            "message": (
                                f"The query is valid but returned no rows because "
                                f"the table(s) {names} are empty in this database. "
                                f"Try a goal based on the tables that contain data."
                            )
                        }
                    if attempt >= retries:
                        return [], current_sql, {
                            "message": "The query ran successfully but returned no rows for the current data."
                        }
                    logging.warning(f"SQL returned no rows on attempt {attempt + 1}. Asking LLM to repair for valid broader joins.")
                    fix_prompt = f"""
The SQL below returned zero rows even though the goal is valid.
The query should still return useful data for the business question.
Prefer safe left joins, COALESCE for missing values, and avoid overly strict exact matches.

User goal:
{user_goal}

Relevant schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Return only the corrected SQL query. No explanations, no markdown.
"""
                    try:
                        raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500)
                        current_sql = self._clean_sql(raw)
                        continue
                    except Exception:
                        return [], current_sql, {
                            "message": "The query returned no rows and could not be repaired automatically."
                        }

                df = df.where(pd.notnull(df), None)
                for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
                    df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                for col in df.select_dtypes(include=['float64', 'int64']).columns:
                    df[col] = df[col].astype(float)
                return df.to_dict(orient="records"), current_sql, None
            except Exception as exc:
                if attempt >= retries:
                    # Give up on the LLM SQL, but never fail the request: try a
                    # safe generic query on the join-path tables first.
                    try:
                        safe_sql = self._fallback_sql(join_path)
                        with self.engine.connect() as conn:
                            result = conn.execute(text(safe_sql))
                            rows = result.fetchall()
                            cols = list(result.keys())
                        df = pd.DataFrame(rows, columns=cols)
                        df = df.where(pd.notnull(df), None)
                        for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
                            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                        for col in df.select_dtypes(include=['float64', 'int64']).columns:
                            df[col] = df[col].astype(float)
                        if self._goal_asks_for_aggregation(user_goal):
                            # A raw `SELECT *` fallback would be misleading for an
                            # aggregating goal (it charts IDs, not the breakdown the
                            # user asked for). Explain instead of guessing.
                            return [], safe_sql, {
                                "message": (
                                    "The model could not produce a working aggregate "
                                    "query for this goal. The raw table data is not a "
                                    "valid substitute for the grouped/aggregated answer "
                                    "requested. Please rephrase or try the other model provider."
                                )
                            }
                        logging.warning(f"LLM SQL exhausted retries ({exc}); returning safe fallback.")
                        return df.to_dict(orient="records"), safe_sql, None
                    except Exception as safe_exc:
                        raise RuntimeError(
                            f"Could not build a working query for this goal after retrying "
                            f"({exc}). Last fallback also failed: {safe_exc}"
                        ) from exc
                logging.warning(f"SQL failed on attempt {attempt + 1}: {exc}. Asking LLM to repair.")
                fix_prompt = f"""
The SQL below failed with error:
{exc}

Relevant schema:
{self._build_schema_ddl(join_path, full=False)}

Broken SQL:
{current_sql}

Fix it and return only the corrected SQL query. No explanations, no markdown.
"""
                try:
                    raw = self.llm.complete("sql", fix_prompt, temperature=0.1, num_predict=500)
                    current_sql = self._clean_sql(raw)
                except Exception:
                    break

        return [], current_sql, {
            "message": "Could not produce a working query for this goal after retrying."
        }

    # ------------------------------------------------------------------
    # Missing-value handling / cleaning
    # ------------------------------------------------------------------

    def _clean_data(self, records):
        """Preprocess query results before writing processed_data.json.

        Handles gracefully:
          - fully-empty columns  -> dropped
          - fully-empty rows     -> dropped
          - null / empty-string values -> filled (0 for numeric, '' for text)
          - types               -> coerced (numeric/text/datetime)

        Stores a human-readable report on self.preprocessing_report so the
        payload and UI can explain exactly what was cleaned.
        """
        report = {
            "dropped_empty_columns": [],
            "dropped_empty_rows": 0,
            "nulls_filled": 0,
            "columns_kept": [],
            "notes": [],
        }
        if not records:
            self.preprocessing_report = report
            return []

        df = pd.DataFrame(records)
        df = df.replace({pd.NA: None})

        def _is_empty(series):
            return series.isna() | (series.astype(str).str.strip().isin(["", "nan", "None", "null"]))

        # 1) Drop columns that are entirely empty.
        empty_cols = [c for c in df.columns if len(df[c]) and _is_empty(df[c]).all()]
        if empty_cols:
            df = df.drop(columns=empty_cols)
            report["dropped_empty_columns"] = empty_cols
            report["notes"].append(
                f"Dropped {len(empty_cols)} fully-empty column(s): {', '.join(empty_cols)}."
            )

        if df.empty or len(df.columns) == 0:
            report["notes"].append("No usable columns remain after cleaning.")
            self.preprocessing_report = report
            return []

        # 2) Drop rows that are entirely empty across every remaining column.
        all_empty = df.apply(lambda r: _is_empty(r).all(), axis=1)
        dropped_rows = int(all_empty.sum())
        if dropped_rows:
            df = df.loc[~all_empty].reset_index(drop=True)
            report["dropped_empty_rows"] = dropped_rows
            report["notes"].append(f"Dropped {dropped_rows} entirely-empty row(s).")

        # 3) Fill remaining nulls + coerce types gracefully.
        for col in df.columns:
            col_null = int(df[col].isna().sum())
            kind = df[col].dtype.kind
            if kind in "fc":  # float / complex -> numeric
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            elif kind in "iub":  # int / bool -> numeric (keep float for NaN-free)
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
            elif kind == "M":  # datetime
                df[col] = df[col].astype(object).where(df[col].notna(), None)
            elif kind == "O":  # object / string (includes Decimal, datetime objects)
                # SQLAlchemy returns Postgres NUMERIC as Decimal objects; coerce
                # numeric-looking object columns to float so they are treated as
                # measures downstream (KPIs, charts, predictions), for any database.
                if len(df[col]) and df[col].dropna().map(
                    lambda v: isinstance(v, (datetime, pd.Timestamp))
                ).all():
                    df[col] = df[col].astype(object).where(df[col].notna(), None)
                    continue
                sample = df[col].dropna()
                if len(sample):
                    coerced = pd.to_numeric(sample, errors="coerce")
                    num_ok = int(coerced.notna().sum())
                    if num_ok / len(sample) >= 0.8:
                        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)
                        report["nulls_filled"] += col_null
                        continue
                df[col] = df[col].astype(object).fillna("")
                df[col] = df[col].map(
                    lambda v: "" if isinstance(v, str) and v.strip() == "" else v
                )
            report["nulls_filled"] += col_null

        report["columns_kept"] = list(df.columns)
        if report["nulls_filled"]:
            report["notes"].append(
                f"Filled {report['nulls_filled']} missing value(s) "
                f"(0 for numeric, '' for text)."
            )
        self.preprocessing_report = report

        records = df.where(pd.notnull(df), None).to_dict(orient="records")
        for record in records:
            for key, value in record.items():
                if isinstance(value, datetime):
                    record[key] = value.strftime("%Y-%m-%d %H:%M:%S")
        return records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _schema_summary_text(self):
        """Compact schema description used for LLM-based suggestion generation."""
        lines = []
        for table_name, info in self.tables.items():
            columns = ", ".join(info.get("columns", {}).keys()) or "no columns"
            lines.append(f"- {table_name} ({columns})")
        return "\n".join(lines)

    def get_suggestions(self, limit=8, use_llm=True):
        """Suggest searchable goals. Uses Mistral after analyzing the schema; falls back to template suggestions when the LLM is unavailable."""
        if not self.tables:
            return ["What insights can we derive from the connected data?"]

        template = self._template_suggestions(limit)
        if not use_llm:
            return template

        prompt = f"""
You are the suggestion engine of a BI system. After analysing the database
tables below, suggest {limit} distinct business questions a non-technical user
could search for. Cover sales, totals, trends, comparisons and relationships.

Schema:
{self._schema_summary_text()}

Output a JSON array of strings only. No explanation, no markdown.
"""
        try:
            content = self.llm.chat("suggest", messages=[{"role": "user", "content": prompt}],
                                    temperature=0.5, num_predict=300)
            content = re.sub(r"```json\s*", "", content)
            content = re.sub(r"```\s*", "", content)
            start, end = content.find("["), content.rfind("]")
            if start != -1 and end != -1:
                suggestions = json.loads(content[start:end + 1])
                if isinstance(suggestions, list) and suggestions:
                    return [str(s) for s in suggestions][:limit]
        except Exception as exc:
            logging.warning(f"Suggestion generation failed: {exc}. Using template suggestions.")

        return template

    def _template_suggestions(self, limit=8):
        tables = list(self.tables.keys())
        if not tables:
            return ["What insights can we derive from the connected data?"]

        suggestions = []
        for table_name in tables[:limit]:
            info = self.tables[table_name]
            columns = list(info.get("columns", {}).keys())
            suggestions.append(f"Summarize the main patterns in {table_name}.")
            if columns:
                suggestions.append(f"What are the most important metrics in {table_name}.{columns[0]}?")
            if info.get("foreign_keys"):
                fk = info["foreign_keys"][0]
                suggestions.append(
                    f"How does {table_name} connect to {fk['referenced_table']} through {fk['column']}?"
                )
        seen = set()
        unique = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique.append(suggestion)
        return unique[:limit]

    def _graceful_failure(self, user_goal, output_path, exc):
        """Write a valid processed_data.json explaining WHY the goal could not
        be answered, so the endpoint never returns a 500 for a goal we could
        not answer (provider outage, unfixable SQL, database hiccup)."""
        if self._llm_unavailable(exc) or "429" in str(exc):
            message = (
                "The language model is temporarily unavailable or rate-limited. "
                "Please try again shortly or switch the model provider."
            )
        else:
            message = f"Could not answer this goal: {exc}"
        output = {
            "user_goal": user_goal,
            "kpi_alignment": {},
            "join_path": [],
            "sql_used": None,
            "row_count": 0,
            "data": [],
            "message": message,
            "preprocessing": {"notes": ["No data retrieved; the goal could not be answered."]},
            "missing_values_handled": "filled (0 for numeric, '' for text)",
            "timestamp": datetime.now().isoformat(),
        }
        Path(output_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        logging.error("Goal Agent graceful failure for %r: %s", user_goal, exc)
        return output_path

    def process_goal(self, user_goal, output_path="processed_data.json"):
        try:
            if LANGGRAPH_AVAILABLE and StateGraph is not None and END is not None:
                return self._process_goal_langgraph(user_goal, output_path)
            return self._process_goal_linear(user_goal, output_path)
        except Exception as exc:
            return self._graceful_failure(user_goal, output_path, exc)

    def _process_goal_linear(self, user_goal, output_path="processed_data.json"):
        print(f"Goal Agent: '{user_goal}'")

        kpi_map = self.map_goal_to_kpi(user_goal)
        print(f"Aligned KPIs: {kpi_map['kpis']} | dimensions: {kpi_map['dimensions']}")

        relevant_tables = self._get_relevant_tables(user_goal)
        join_path = self._determine_join_path(relevant_tables)
        print(f"Join path: {join_path}")

        raw_sql = self._generate_sql(user_goal, join_path, kpi_map)
        print(f"Generated SQL: {raw_sql}")

        data_rows, final_sql, note = self._execute_with_retry(user_goal, join_path, kpi_map, raw_sql)
        cleaned_data = self._clean_data(data_rows)

        output = {
            "user_goal": user_goal,
            "kpi_alignment": kpi_map,
            "join_path": join_path,
            "sql_used": final_sql,
            "row_count": len(cleaned_data),
            "data": cleaned_data,
            "message": (note or {}).get("message") if note else None,
            "preprocessing": self.preprocessing_report,
            "missing_values_handled": "filled (0 for numeric, '' for text)",
            "timestamp": datetime.now().isoformat(),
        }

        out_path = Path(output_path)
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        print(f"Goal Agent done! Saved to {output_path}")
        return output_path

    def _process_goal_langgraph(self, user_goal, output_path):
        def load_schema(state):
            state["kpi_map"] = self.map_goal_to_kpi(state["user_goal"])
            state["relevant_tables"] = self._get_relevant_tables(state["user_goal"])
            state["join_path"] = self._determine_join_path(state["relevant_tables"])
            return state

        def generate_sql(state):
            state["raw_sql"] = self._generate_sql(state["user_goal"], state["join_path"], state["kpi_map"])
            return state

        def execute_sql(state):
            rows, final_sql, note = self._execute_with_retry(
                state["user_goal"], state["join_path"], state["kpi_map"], state["raw_sql"]
            )
            state["rows"] = self._clean_data(rows)
            state["final_sql"] = final_sql
            state["message"] = (note or {}).get("message") if note else None
            state["preprocessing"] = self.preprocessing_report
            return state

        def finalize(state):
            output = {
                "user_goal": state["user_goal"],
                "kpi_alignment": state["kpi_map"],
                "join_path": state["join_path"],
                "sql_used": state["final_sql"],
                "row_count": len(state["rows"]),
                "data": state["rows"],
                "message": state.get("message"),
                "preprocessing": state.get("preprocessing"),
                "missing_values_handled": "filled (0 for numeric, '' for text)",
                "timestamp": datetime.now().isoformat(),
            }
            Path(output_path).write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
            state["result_path"] = output_path
            return state

        workflow = StateGraph(dict)
        workflow.add_node("load_schema", load_schema)
        workflow.add_node("generate_sql", generate_sql)
        workflow.add_node("execute_sql", execute_sql)
        workflow.add_node("finalize", finalize)
        workflow.set_entry_point("load_schema")
        workflow.add_edge("load_schema", "generate_sql")
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "finalize")
        workflow.add_edge("finalize", END)

        app = workflow.compile()
        result = app.invoke({
            "user_goal": user_goal, "kpi_map": {}, "relevant_tables": [],
            "join_path": [], "raw_sql": "", "rows": [], "final_sql": "",
        })
        print(f"Goal Agent done! Saved to {result['result_path']}")
        return result["result_path"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python goal_agent.py \"<business goal in plain english>\"")
        sys.exit(1)

    agent = GoalAgent(
        schema_json_path=str(SCHEMA_DIR / "schema_mapping_latest.json"),
        db_uri=DEFAULT_DB_URI,
    )

    print("Sample suggestions:")
    for i, suggestion in enumerate(agent.get_suggestions(5), 1):
        print(f"{i}. {suggestion}")

    agent.process_goal(" ".join(sys.argv[1:]))
