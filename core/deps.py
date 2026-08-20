"""Shared FastAPI dependencies: session guards, authentication, mobile read-only."""

import re

from fastapi import Depends, Header, HTTPException, Request

import auth
from core.config import get_session_for_request

_MOBILE_RE = re.compile(
    r"Mobile|Android|iPhone|iPad|iPod|Windows Phone|BlackBerry|IEMobile|Opera Mini|Silk",
    re.IGNORECASE,
)


def is_mobile(request: Request) -> bool:
    """Detect a mobile client via an explicit header (preferred) or the UA string."""
    client = request.headers.get("x-client-type", "")
    if client:
        return client.lower() == "mobile"
    return bool(_MOBILE_RE.search(request.headers.get("user-agent", "")))


def require_provider(request: Request):
    """Fail unless an LLM provider is active in the request-scoped session."""
    session = get_session_for_request(request)
    if not session["provider"]:
        raise HTTPException(status_code=400, detail="Choose an LLM provider and connect a source first (POST /api/connect).")
    return session["provider"]


def require_data(request: Request):
    """Require a provider plus a usable source (relational DB or uploaded CSV/PDF)."""
    session = get_session_for_request(request)
    provider = session["provider"]
    if not provider:
        raise HTTPException(status_code=400, detail="Choose an LLM provider and connect a source first (POST /api/connect).")
    if session["source_type"] == "relational" and not session["db"]:
        raise HTTPException(status_code=400, detail="Connect to a relational database first (POST /api/connect).")
    if session["source_type"] == "semi-structured" and not session["db_uri"]:
        raise HTTPException(status_code=400, detail="Upload a CSV or PDF file first (POST /api/connect_file).")
    return provider


def get_current_user(request: Request, authorization: str | None = Header(default=None)) -> dict:
    """Resolve the Bearer token into a {user_id, username} dict or 401."""
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif request.query_params.get("token"):
        token = request.query_params.get("token")
    user = auth.verify_token(token) if token else None
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated. Provide a valid Bearer token.")
    return user


def require_writable(request: Request, user: dict = Depends(get_current_user)) -> dict:
    """Allow an authenticated user; block write actions on mobile (view history only)."""
    if is_mobile(request):
        raise HTTPException(
            status_code=403,
            detail="Read-only on mobile: you can view your saved history, but connecting sources and running analyses are disabled on this device.",
        )
    return user
