"""User store, JWT auth, and per-user history for ARIA.

Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib only, no bcrypt wheels).
Tokens are HS256 JWTs valid for TOKEN_TTL_DAYS. History is kept per user in a
local SQLite database (data/aria.db).
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "aria.db"
JWT_ALGO = "HS256"
TOKEN_TTL_DAYS = 7
PBKDF2_ITERATIONS = 120_000

_USERNAME_RE = re.compile(r"[A-Za-z0-9_.-]+")


def _load_secret() -> str:
    secret_path = BASE_DIR / "data" / ".jwt_secret"
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    if secret_path.exists():
        return secret_path.read_text(encoding="utf-8").strip()
    secret = secrets.token_hex(32)
    secret_path.write_text(secret, encoding="utf-8")
    try:
        os.chmod(secret_path, 0o600)
    except OSError:
        pass
    return secret


SECRET = _load_secret()


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            goal TEXT NOT NULL,
            sql_used TEXT,
            row_count INTEGER,
            columns TEXT,
            source_type TEXT,
            db_name TEXT,
            dialect TEXT,
            processed_json TEXT,
            insights_json TEXT,
            created_at TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        _algo, iters, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def register(username: str, password: str) -> int:
    username = (username or "").strip()
    if len(username) < 3 or not _USERNAME_RE.fullmatch(username):
        raise ValueError("Username must be at least 3 characters using letters, digits, . _ -.")
    if len(password) < 4:
        raise ValueError("Password must be at least 4 characters.")
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, _hash_password(password), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row["id"]
    except sqlite3.IntegrityError:
        raise ValueError("Username already taken.")
    finally:
        conn.close()


def authenticate(username: str, password: str) -> int | None:
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
    finally:
        conn.close()
    if row and _verify_password(password, row["password_hash"]):
        return row["id"]
    return None


def issue_token(user_id: int, username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + timedelta(days=TOKEN_TTL_DAYS),
    }
    return jwt.encode(payload, SECRET, algorithm=JWT_ALGO)


def verify_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, SECRET, algorithms=[JWT_ALGO])
        return {"user_id": int(payload["sub"]), "username": payload.get("username")}
    except jwt.PyJWTError:
        return None


def add_history(user_id: int, *, goal: str, sql_used=None, row_count=None,
                columns=None, source_type=None, db_name=None, dialect=None,
                processed=None) -> int:
    conn = _get_db()
    try:
        cur = conn.execute(
            """INSERT INTO history
               (user_id, goal, sql_used, row_count, columns, source_type, db_name, dialect, processed_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id, goal, sql_used, row_count,
                json.dumps(columns) if columns else None,
                source_type, db_name, dialect,
                json.dumps(processed) if processed is not None else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def set_history_insights(history_id: int, user_id: int, insights) -> None:
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE history SET insights_json = ? WHERE id = ? AND user_id = ?",
            (json.dumps(insights), history_id, user_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_history(user_id: int, limit: int = 200) -> list[dict]:
    conn = _get_db()
    try:
        rows = conn.execute(
            """SELECT id, goal, sql_used, row_count, columns, source_type, db_name, dialect, created_at, insights_json
               FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    finally:
        conn.close()
    items = []
    for r in rows:
        items.append(
            {
                "id": r["id"],
                "goal": r["goal"],
                "sql_used": r["sql_used"],
                "row_count": r["row_count"],
                "columns": json.loads(r["columns"]) if r["columns"] else [],
                "source_type": r["source_type"],
                "db_name": r["db_name"],
                "dialect": r["dialect"],
                "created_at": r["created_at"],
                "has_insights": bool(r["insights_json"]),
            }
        )
    return items


def get_history(user_id: int, history_id: int) -> dict | None:
    conn = _get_db()
    try:
        r = conn.execute(
            "SELECT * FROM history WHERE id = ? AND user_id = ?",
            (history_id, user_id),
        ).fetchone()
    finally:
        conn.close()
    if not r:
        return None
    return {
        "id": r["id"],
        "goal": r["goal"],
        "sql_used": r["sql_used"],
        "row_count": r["row_count"],
        "columns": json.loads(r["columns"]) if r["columns"] else [],
        "source_type": r["source_type"],
        "db_name": r["db_name"],
        "dialect": r["dialect"],
        "created_at": r["created_at"],
        "processed": json.loads(r["processed_json"]) if r["processed_json"] else None,
        "insights": json.loads(r["insights_json"]) if r["insights_json"] else None,
    }