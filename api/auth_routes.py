"""Authentication and per-user history routes."""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

import auth
from core.deps import get_current_user, is_mobile

router = APIRouter()


class AuthRequest(BaseModel):
    username: str
    password: str


@router.post("/api/auth/register")
def register(auth_req: AuthRequest):
    """Create a user account and return a JWT."""
    try:
        user_id = auth.register(auth_req.username, auth_req.password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    username = auth_req.username.strip()
    return {"ok": True, "token": auth.issue_token(user_id, username), "username": username}


@router.post("/api/auth/login")
def login(auth_req: AuthRequest):
    """Exchange username/password for a JWT."""
    user_id = auth.authenticate(auth_req.username, auth_req.password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    username = auth_req.username.strip()
    return {"ok": True, "token": auth.issue_token(user_id, username), "username": username}


@router.get("/api/auth/me")
def me(user: dict = Depends(get_current_user)):
    return {"ok": True, "username": user["username"], "user_id": user["user_id"]}


@router.post("/api/auth/logout")
def logout(user: dict = Depends(get_current_user)):
    """JWT is stateless; the client simply discards the token."""
    return {"ok": True, "username": user["username"]}


@router.get("/api/history")
def history(request: Request, user: dict = Depends(get_current_user), limit: int = 200):
    """List the current user's saved analyses (metadata only, newest first)."""
    items = auth.list_history(user["user_id"], limit=min(limit, 500))
    return {
        "ok": True,
        "mode": "read-only" if is_mobile(request) else "full",
        "items": items,
    }


@router.get("/api/history/{history_id}")
def history_detail(history_id: int, user: dict = Depends(get_current_user)):
    """Return one saved analysis including processed data and insights."""
    record = auth.get_history(user["user_id"], history_id)
    if not record:
        raise HTTPException(status_code=404, detail="History record not found.")
    return {"ok": True, "record": record}
