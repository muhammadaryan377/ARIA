"""FK inference benchmark (txt Priority 4).

Takes databases with KNOWN (declared) foreign keys, removes those constraints,
re-runs inference on the stripped clone, and compares the inferred relationships
against the original declared set:

    Precision = correct_inferred / total_inferred
    Recall    = correct_inferred / total_declared
    F1        = 2*P*R / (P+R)
    FPR       = wrong_inferred / total_inferred
    FNR       = missed_declared / total_declared

Run:
    python benchmark_fk.py [--dbs northwind chinook ...]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import schema_agent

ADMIN = "postgresql://postgres:12345@localhost:5432/postgres"

DEFAULT_DBS = ["northwind", "chinook", "olist_ecommerce", "retail_fraud"]

MAX_SOURCE_RATIO = 0.70  # keep at least 30% of the total inferred set


def make_cfg(db):
    return {
        "host": "localhost", "port": 5432, "db": db, "user": "postgres",
        "password": "12345", "db_type": "postgresql", "db_schema": "public",
    }


def list_fks(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT tc.table_name, kcu.column_name, ccu.table_name AS ftable,
               ccu.column_name AS fcol
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
        """,
    )
    return cur.fetchall()


def clone_and_strip(src):
    import psycopg2
    admin = psycopg2.connect(ADMIN)
    admin.autocommit = True
    cur = admin.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (src,))
    if not cur.fetchone():
        admin.close()
        return None
    dst = f"{src}_bench"
    cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (dst,))
    if cur.fetchone():
        cur.execute(f'DROP DATABASE "{dst}"')
    cur.execute(f'CREATE DATABASE "{dst}" TEMPLATE "{src}"')
    admin.close()

    conn = psycopg2.connect(f"postgresql://postgres:12345@localhost:5432/{dst}")
    conn.autocommit = True
    cur = conn.cursor()
    fks = list_fks(conn)
    cur.execute(
        """
        SELECT tc.table_name, tc.constraint_name
        FROM information_schema.table_constraints tc
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
        """,
    )
    for t, name in cur.fetchall():
        try:
            cur.execute(f'ALTER TABLE "{t}" DROP CONSTRAINT "{name}"')
        except Exception:
            pass
    conn.close()
    return dst, fks


def edge_set(mapping):
    out = set()
    for r in mapping.get("inferred_relationships", []):
        col = r.get("column")
        if isinstance(col, list):
            col = ",".join(col)
        ref = r.get("references_column")
        if isinstance(ref, list):
            ref = ",".join(ref)
        out.add((r.get("table"), str(col), r.get("references_table"), str(ref)))
    return out


def run(db):
    cfg = make_cfg(db)
    conn = schema_agent.get_connection(type("C", (), cfg)())
    try:
        mapping = schema_agent.build_schema_mapping(conn, "public", db_type="postgresql")
    finally:
        conn.close()
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dbs", nargs="*", default=DEFAULT_DBS)
    args = ap.parse_args()

    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}

    for db in args.dbs:
        print("=" * 74)
        print(f"DB: {db}", flush=True)
        result = clone_and_strip(db)
        if result is None:
            print("  skipped (not found)")
            continue
        dst, declared = result
        declared_set = {(t, str(c), ft, str(fc)) for t, c, ft, fc in declared}
        t0 = time.time()
        try:
            stripped_map = run(dst)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        elapsed = time.time() - t0

        inferred = edge_set(stripped_map)
        correct = declared_set & inferred
        wrong = inferred - declared_set
        missed = declared_set - inferred
        tp = len(correct)
        fp = len(wrong)
        fn = len(missed)

        precision = tp / len(inferred) if inferred else 0.0
        recall = tp / len(declared_set) if declared_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fpr = fp / len(inferred) if inferred else 0.0
        fnr = fn / len(declared_set) if declared_set else 0.0

        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        print(f"  declared={len(declared_set)} inferred={len(inferred)} ({elapsed:.1f}s)")
        print(f"  Precision={precision:.1%} Recall={recall:.1%} F1={f1:.3f} FPR={fpr:.1%} FNR={fnr:.1%}")
        if correct:
            print("  CORRECT:")
            for e in sorted(correct, key=str):
                print(f"    {'.'.join([e[0], e[1]])} -> {'.'.join([e[2], e[3]])}")
        if wrong:
            print("  FALSE POSITIVES:")
            for e in sorted(wrong, key=str):
                print(f"    {'.'.join([e[0], e[1]])} -> {'.'.join([e[2], e[3]])}")
        if missed:
            print("  FALSE NEGATIVES (missed):")
            for e in sorted(missed, key=str):
                print(f"    {'.'.join([e[0], e[1]])} -> {'.'.join([e[2], e[3]])}")
        rows.append({
            "db": db, "declared": len(declared_set), "inferred": len(inferred),
            "correct": tp, "false_positives": fp, "missed": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "fpr": round(fpr, 4), "fnr": round(fnr, 4),
            "elapsed_s": round(elapsed, 1),
        })

    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    print("\n" + "=" * 74)
    print(f"OVERALL (across {len(rows)} DBs): Precision={p:.1%} Recall={r:.1%} F1={f1:.3f} (TP={tp} FP={fp} FN={fn})")
    out = Path(__file__).resolve().parent / "benchmark_results.json"
    out.write_text(json.dumps({"dbs": rows, "overall": {
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
    }}, indent=2), encoding="utf-8")
    print(f"Saved {out.name}")


if __name__ == "__main__":
    main()