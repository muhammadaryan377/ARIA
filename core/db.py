"""Small database / identifier helpers."""

import re
from urllib.parse import quote_plus

_URI_SCHEMES = {
    "postgresql": "postgresql",
    "mysql": "mysql+pymysql",
}


def build_db_uri(config, db_type="postgresql"):
    """Build a SQLAlchemy connection URI for the given database type.

    config is a dict with user/password/host/port/db keys. For MySQL the
    scheme is ``mysql+pymysql`` (needs pymysql installed).
    """
    scheme = _URI_SCHEMES.get(db_type, "postgresql")
    return (
        f"{scheme}://{quote_plus(config['user'])}:{quote_plus(config['password'])}"
        f"@{config['host']}:{config['port']}/{quote_plus(config['db'])}"
    )


def safe_db_name(name):
    """Sanitize a filename into a valid SQLite table/database name."""
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip()).strip("_") or "file_upload"