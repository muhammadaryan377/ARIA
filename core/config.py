"""Shared application configuration and per-user runtime session state.

Environment variables are loaded exactly once here (from the project root's
``.env``) so every other module reads them without calling load_dotenv itself.
"""

import os
import threading
import time
from copy import deepcopy
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Secrets in .env may be stored DPAPI-encrypted (`enc1:<base64>`) so the file
# is unreadable at rest. Decrypt them back into the environment now that dotenv
# has loaded, before any module reads os.getenv for credentials.
from core.secret_box import decrypt, PREFIX  # noqa: E402

for _name, _value in list(os.environ.items()):
    if isinstance(_value, str) and _value.startswith(PREFIX):
        try:
            os.environ[_name] = decrypt(_value)
        except Exception:
            pass

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
# Dedicated folder for all schema mapping JSON files, keeping them out of the
# code tree (everything under data/ is gitignored and regenerable).
SCHEMA_DIR = DATA_DIR / "schemas"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed_data"
INSIGHTS_DIR = ARTIFACTS_DIR / "insights"
FLOW_REPORTS_DIR = ARTIFACTS_DIR / "flow_reports"

for _dir in (ARTIFACTS_DIR, DATA_DIR, SCHEMA_DIR, PROCESSED_DIR, INSIGHTS_DIR, FLOW_REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

SCHEMA_PATH = SCHEMA_DIR / "schema_mapping_latest.json"
PROCESSED_PATH = PROCESSED_DIR / "processed_data.json"
INSIGHTS_PATH = INSIGHTS_DIR / "insights.json"

SOURCE_TYPES = {"relational", "semi-structured"}
PROVIDER_NAMES = {"local", "cloud"}
DB_TYPES = {"postgresql", "mysql"}

DEFAULT_SESSION = {
    "provider": None,       # LLMProvider instance
    "provider_name": None,  # 'local' | 'cloud'
    "source_type": None,    # 'relational' | 'semi-structured'
    "db": None,             # relational DB config
    "db_type": "postgresql",  # 'postgresql' | 'mysql' (relational sources only)
    "db_uri": None,
    "db_name": None,        # name of the connected database / table
    "schema_path": SCHEMA_PATH,   # schema file for the active database
    "processed_path": PROCESSED_PATH,
    "insights_path": INSIGHTS_PATH,
    "dialect": "postgresql",     # 'postgresql'|'mysql' for relational, 'sqlite' for CSV/PDF
    "pending_history_id": None,  # most recent history row awaiting insights
}

SESSION = deepcopy(DEFAULT_SESSION)
SESSION_STORE = {}
SESSION_LOCK = threading.RLock()

# Session store hygiene: cap the number of cached per-user sessions and drop
# sessions that have not been touched for SESSION_TTL_SECONDS. Prevents
# unbounded memory growth in long-running multi-user servers.
SESSION_MAX_USERS = 100
SESSION_TTL_SECONDS = 4 * 60 * 60  # 4 hours


def get_session(user_id=None):
    """Return the current session for a user, or a fresh default copy.

    The default (unauthenticated) session is never shared: every call gets its
    own copy, so no request can leak state into another. All write routes
    require authentication anyway, so the default copy is only a safe fallback.
    """
    if user_id is None:
        return deepcopy(DEFAULT_SESSION)

    now = time.time()
    with SESSION_LOCK:
        # Opportunistic cleanup of expired sessions.
        if len(SESSION_STORE) >= SESSION_MAX_USERS or len(SESSION_STORE) > 0:
            expired = [
                uid for uid, (created, _session) in SESSION_STORE.items()
                if now - created > SESSION_TTL_SECONDS
            ]
            for uid in expired:
                del SESSION_STORE[uid]
            # If still at capacity, drop the oldest entry.
            while len(SESSION_STORE) >= SESSION_MAX_USERS:
                oldest = min(SESSION_STORE, key=lambda uid: SESSION_STORE[uid][0])
                del SESSION_STORE[oldest]

        entry = SESSION_STORE.get(user_id)
        if entry is None:
            entry = (now, deepcopy(DEFAULT_SESSION))
            SESSION_STORE[user_id] = entry
        else:
            SESSION_STORE[user_id] = (now, entry[1])
        return entry[1]


def get_session_for_request(request):
    """Resolve a session using the current request bearer token when available."""
    token = None
    authorization = request.headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif request.query_params.get("token"):
        token = request.query_params.get("token")
    if not token:
        return get_session(None)
    try:
        import auth
        payload = auth.verify_token(token)
    except Exception:
        return get_session(None)
    if not payload:
        return get_session(None)
    return get_session(payload.get("user_id"))
