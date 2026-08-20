"""Central startup checks: installed packages, required env vars, and live
health signals (Ollama reachability, default DB reachability).

Run standalone:

    python core/checks.py            # human-readable report (exit 1 if critical)
    python core/checks.py --json     # machine-readable report

The FastAPI app calls :func:`run_checks` at startup and exposes the same data
through ``GET /api/health``.
"""

import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent

# package name -> (import module, pip install argument)
_REQUIRED_PACKAGES = {
    "fastapi": ("fastapi", "fastapi"),
    "uvicorn": ("uvicorn", "uvicorn"),
    "sqlalchemy": ("sqlalchemy", "sqlalchemy"),
    "pandas": ("pandas", "pandas"),
    "numpy": ("numpy", "numpy"),
    "psycopg2": ("psycopg2", "psycopg2-binary"),
    "PyJWT": ("jwt", "PyJWT"),
    "python-multipart": ("multipart", "python-multipart"),
    "pdfplumber": ("pdfplumber", "pdfplumber"),
    "python-dotenv": ("dotenv", "python-dotenv"),
    "httpx": ("httpx", "httpx"),
    "pydantic": ("pydantic", "pydantic"),
}

_OPTIONAL_PACKAGES = {
    "pymysql": ("pymysql", "pymysql"),       # only needed for MySQL sources
    "groq": ("groq", "groq"),                # only needed for the Cloud (Groq) provider
    "ollama": ("ollama", "ollama"),          # only needed for the Local (Ollama) provider
    "langgraph": ("langgraph", "langgraph"), # used by goal_agent; falls back to a linear flow
}


def _pkg_installed(module):
    return importlib.util.find_spec(module) is not None


def check_install():
    """Report which required/optional packages are importable."""
    required = {
        name: {"installed": _pkg_installed(mod), "pip": f"pip install {pip_name}"}
        for name, (mod, pip_name) in _REQUIRED_PACKAGES.items()
    }
    optional = {
        name: {"installed": _pkg_installed(mod), "pip": f"pip install {pip_name}"}
        for name, (mod, pip_name) in _OPTIONAL_PACKAGES.items()
    }
    missing_required = [n for n, v in required.items() if not v["installed"]]
    return {
        "required": required,
        "optional": optional,
        "missing_required": missing_required,
    }


def check_env():
    """Report which env vars are set and note what each gap implies."""
    env = {
        "DB_USER": bool(os.getenv("DB_USER")),
        "DB_PASSWORD": bool(os.getenv("DB_PASSWORD")),
        "DB_HOST": bool(os.getenv("DB_HOST")),
        "DB_NAME": bool(os.getenv("DB_NAME")),
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
    }
    jwt_secret_path = _BASE_DIR / "data" / ".jwt_secret"
    env["jwt_secret_persisted"] = jwt_secret_path.exists()

    notes = []
    if not env["GROQ_API_KEY"]:
        notes.append(
            "Cloud provider will fail until GROQ_API_KEY is set (or a key is "
            "provided in the UI)."
        )
    if not env["DB_NAME"]:
        notes.append(
            "DB_NAME not set in .env: databases are connected per-session through "
            "the UI, so this is only needed for standalone scripts."
        )
    if not env["jwt_secret_persisted"]:
        notes.append(
            "No persisted JWT secret yet: one is generated on first start "
            "(data/.jwt_secret) so tokens survive restarts."
        )
    return {"env": env, "notes": notes}


def check_ollama():
    """Ping the Ollama server (Local provider prerequisite)."""
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    url = host + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return {"ok": resp.status == 200, "detail": f"reachable at {host}"}
    except Exception as exc:
        return {
            "ok": False,
            "detail": f"not reachable at {host}: {type(exc).__name__}. "
            "Start Ollama (ollama serve) and pull the aria-* models to use the Local provider.",
        }


def check_default_db():
    """Best-effort reachability of the .env default database (info only)."""
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    dbname = os.getenv("DB_NAME")
    if not (user and password is not None and dbname):
        return {
            "configured": False,
            "ok": None,
            "detail": "DB_* not set in .env; databases are connected per-session in the UI.",
        }
    try:
        import psycopg2
    except ImportError:
        return {"configured": True, "ok": None, "detail": "psycopg2 not installed."}
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user,
            password=password, connect_timeout=3,
        )
        conn.close()
        return {
            "configured": True,
            "ok": True,
            "detail": f"connected to {dbname}@{host}:{port}",
        }
    except Exception as exc:
        return {
            "configured": True,
            "ok": False,
            "detail": f"could not connect to {dbname}@{host}:{port}: {exc}",
        }


def run_checks():
    """Aggregate all checks into one dict for /api/health and the CLI."""
    install = check_install()
    env = check_env()
    return {
        "python": sys.version.split()[0],
        "install": install,
        "env": env,
        "ollama": check_ollama(),
        "default_db": check_default_db(),
        "critical": install["missing_required"],
    }


def _format_report(report):
    lines = []
    lines.append(f"ARIA health  (python {report['python']})")
    lines.append("")
    lines.append("Packages:")
    for name, info in report["install"]["required"].items():
        mark = "ok  " if info["installed"] else "MISS"
        lines.append(f"  [{mark}] {name:<18} ({info['pip']})")
    lines.append("  Optional:")
    for name, info in report["install"]["optional"].items():
        mark = "ok  " if info["installed"] else "absent"
        lines.append(f"  [{mark}] {name:<18} ({info['pip']})")
    lines.append("")
    lines.append("Environment:")
    for key, value in report["env"]["env"].items():
        shown = value if isinstance(value, str) else ("set" if value else "unset")
        lines.append(f"  {key:<24} {shown}")
    lines.append("")
    lines.append(f"Ollama:  {report['ollama']['detail']}")
    lines.append(f"Default DB: {report['default_db']['detail']}")
    lines.append("")
    if report["env"]["notes"]:
        lines.append("Notes:")
        for note in report["env"]["notes"]:
            lines.append(f"  - {note}")
    lines.append("")
    if report["critical"]:
        lines.append("CRITICAL: missing required packages -> " + ", ".join(report["critical"]))
    else:
        lines.append("All required packages present.")
    return "\n".join(lines)


def main():
    report = run_checks()
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2))
    else:
        print(_format_report(report))
    sys.exit(1 if report["critical"] else 0)


if __name__ == "__main__":
    main()
