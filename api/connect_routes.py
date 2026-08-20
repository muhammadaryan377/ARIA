"""Data source connection routes: relational databases (PostgreSQL / MySQL)."""

import json
import re
import types

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import schema_agent
from core.config import DB_TYPES, PROVIDER_NAMES, SOURCE_TYPES, SCHEMA_DIR, PROCESSED_DIR, INSIGHTS_DIR, get_session
from core.db import build_db_uri
from core.deps import require_writable
from llm_provider import create_provider

router = APIRouter()


class ProviderSwitchRequest(BaseModel):
    provider: str = "local"
    api_key: str | None = None      # optional key override for hosted providers


@router.post("/api/provider/switch")
def switch_provider(request: ProviderSwitchRequest, user: dict = Depends(require_writable)):
    """Explicitly switch the LLM backend for this session."""
    session = get_session(user["user_id"])
    if request.provider not in PROVIDER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider}'. Choose from {sorted(PROVIDER_NAMES)}.")
    current = session.get("provider_name")
    if current == request.provider:
        return {"ok": True, "provider": current, "changed": False}

    try:
        llm = create_provider(
            provider=request.provider,
            api_key=request.api_key,
            base_url=request.base_url,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not initialise '{request.provider}' LLM provider: {exc}")

    session["provider"] = llm
    session["provider_name"] = request.provider
    return {"ok": True, "provider": request.provider, "changed": True}


class ConnectRequest(BaseModel):
    provider: str = "local"
    source_type: str = "relational"
    db_type: str = "postgresql"   # 'postgresql' | 'mysql'
    host: str = "localhost"
    port: int | str = 5432
    db: str | None = None
    user: str | None = None
    password: str | None = None
    db_schema: str = "public"


def _do_connect(request: ConnectRequest, user: dict, db_type: str) -> dict:
    """Shared connect logic. `db_type` is authoritative (set by the calling endpoint)."""
    session = get_session(user["user_id"])
    if request.provider not in PROVIDER_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown provider '{request.provider}'. Choose from {sorted(PROVIDER_NAMES)}.")
    if request.source_type not in SOURCE_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown source type '{request.source_type}'. Choose from {sorted(SOURCE_TYPES)}.")
    if request.source_type == "relational" and db_type not in DB_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown database type '{db_type}'. Choose from {sorted(DB_TYPES)}.")

    try:
        llm = create_provider(provider=request.provider)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not initialise '{request.provider}' LLM provider: {exc}")

    session["provider"] = llm
    session["provider_name"] = request.provider
    session["source_type"] = request.source_type
    session["db"] = None
    session["db_type"] = db_type
    session["db_uri"] = None
    session["db_name"] = None
    uid_dir = f"user_{user['user_id']}"
    session["schema_path"] = SCHEMA_DIR / uid_dir / "schema_mapping_latest.json"
    session["processed_path"] = PROCESSED_DIR / uid_dir / "processed_data.json"
    session["insights_path"] = INSIGHTS_DIR / uid_dir / "insights.json"

    if not (request.db and request.user and request.password is not None):
        raise HTTPException(status_code=400, detail="Relational source requires db, user, and password.")

    config = request.model_dump()
    config["db_type"] = db_type
    config["port"] = int(config["port"])
    try:
        conn = schema_agent.get_connection(types.SimpleNamespace(**config))
        with conn.cursor() as cur:
            if db_type == "mysql":
                cur.execute("SELECT DATABASE();")
                db_name = cur.fetchone()[0]
                current_db_schema = db_name
            else:
                cur.execute("SELECT current_database(), current_schema();")
                db_name, current_db_schema = cur.fetchone()
        conn.close()
    except SystemExit as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        detail = f"Connection failed: {exc}"
        if db_type == "mysql":
            detail += (" Hint: is the MySQL server running, and is pymysql installed "
                       "(pip install pymysql)? Verify host/port, database, user, password.")
        else:
            detail += (" Hint: is the PostgreSQL service running on that host/port? "
                       "Verify the database name, user, and password are correct.")
        raise HTTPException(status_code=400, detail=detail)

    session["db"] = config
    session["db_type"] = db_type
    session["db_uri"] = build_db_uri(config, db_type)
    session["db_name"] = db_name
    session["dialect"] = db_type

    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", db_name)
    uid_dir = f"user_{user['user_id']}"
    session["schema_path"] = SCHEMA_DIR / uid_dir / "schema_mapping_latest.json"
    session["processed_path"] = PROCESSED_DIR / uid_dir / f"processed_data_{safe}.json"
    session["insights_path"] = INSIGHTS_DIR / uid_dir / f"insights_{safe}.json"

    return {
        "ok": True,
        "provider": request.provider,
        "source_type": "relational",
        "database": db_name,
        "schema": current_db_schema,
        "db_type": db_type,
        "schema_file": session["schema_path"].name,
    }


@router.post("/api/connect")
def connect(request: ConnectRequest, user: dict = Depends(require_writable)):
    """Connect to a source; database type taken from the request body."""
    return _do_connect(request, user, request.db_type)


@router.post("/api/connect/postgresql")
def connect_postgresql(request: ConnectRequest, user: dict = Depends(require_writable)):
    """Connect to a PostgreSQL source (explicit route)."""
    return _do_connect(request, user, "postgresql")


@router.post("/api/connect/mysql")
def connect_mysql(request: ConnectRequest, user: dict = Depends(require_writable)):
    """Connect to a MySQL source (explicit route)."""
    return _do_connect(request, user, "mysql")



