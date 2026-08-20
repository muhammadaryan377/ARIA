"""Goal Agent + Insight Agent routes: suggestions, ask, and insight analysis."""

import json
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

import auth
from core.config import get_session
from core.deps import require_data, require_writable
from core.validation import validate_goal_text
from goal_agent import GoalAgent
from insight_agent import InsightAgent, sanitize_json

router = APIRouter()


class GoalRequest(BaseModel):
    goal: str


@router.post("/api/suggestions")
def get_suggestions(request: Request, payload: dict | None = None, user: dict = Depends(require_writable)):
    """LLM-driven suggested search goals after analyzing the tables."""
    session = get_session(user["user_id"])
    provider = require_data(request)
    limit = (payload or {}).get("limit", 8)
    try:
        agent = GoalAgent(schema_json_path=str(session["schema_path"]), db_uri=session["db_uri"], provider=provider, dialect=session["dialect"])
        suggestions = agent.get_suggestions(limit=limit, use_llm=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Suggestion generation failed: {exc}")
    return {"ok": True, "provider": session["provider_name"], "suggestions": suggestions}


@router.post("/api/ask")
def ask(request: Request, req: GoalRequest, user: dict = Depends(require_writable)):
    """Run the Goal Agent: goal -> SQL -> processed_data.json."""
    session = get_session(user["user_id"])
    provider = require_data(request)
    goal = validate_goal_text(req.goal)
    try:
        agent = GoalAgent(schema_json_path=str(session["schema_path"]), db_uri=session["db_uri"], provider=provider, dialect=session["dialect"])
        agent.process_goal(goal, output_path=str(session["processed_path"]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Goal Agent failed: {exc}")

    processed = json.loads(session["processed_path"].read_text(encoding="utf-8"))
    processed = sanitize_json(processed)
    columns = list(processed.get("data", [{}])[0].keys()) if processed.get("data") else []
    session["pending_history_id"] = auth.add_history(
        user["user_id"],
        goal=goal,
        sql_used=processed.get("sql_used"),
        row_count=processed.get("row_count"),
        columns=columns,
        source_type=session["source_type"],
        db_name=session["db_name"],
        dialect=session["dialect"],
        processed=processed,
    )
    return {
        "ok": True,
        "provider": session["provider_name"],
        "user_goal": processed.get("user_goal"),
        "kpi_alignment": processed.get("kpi_alignment"),
        "join_path": processed.get("join_path"),
        "sql_used": processed.get("sql_used"),
        "row_count": processed.get("row_count"),
        "columns": columns,
        "message": processed.get("message"),
        "preprocessing": processed.get("preprocessing"),
        "data": processed.get("data", [])[:50],
        "history_id": session["pending_history_id"],
    }


@router.post("/api/insight")
def insight(request: Request, user: dict = Depends(require_writable)):
    """Run the Insight Agent on the last processed data."""
    session = get_session(user["user_id"])
    provider = require_data(request)
    if not session["processed_path"].exists():
        raise HTTPException(status_code=400, detail="No processed data yet. Call POST /api/ask first.")

    try:
        agent = InsightAgent(provider=provider)
        agent.analyze(str(session["processed_path"]), str(session["insights_path"]))
        insights = json.loads(session["insights_path"].read_text(encoding="utf-8"))
    except Exception as exc:
        processed = json.loads(session["processed_path"].read_text(encoding="utf-8")) if session["processed_path"].exists() else {}
        insights = {
            "generated_at": datetime.now().isoformat(),
            "user_goal": processed.get("user_goal", "Unknown goal"),
            "summary": {"rows": processed.get("row_count", 0), "columns": []},
            "kpis": [],
            "trends": [],
            "anomalies": [],
            "hypotheses": [],
            "dashboard": [],
            "business_story": "The insight agent could not complete this analysis, but the dataset was still received and the UI is ready for retry.",
            "warning": str(exc),
        }
        session["insights_path"].write_text(json.dumps(sanitize_json(insights), indent=2), encoding="utf-8")

    insights = sanitize_json(insights)

    if session["pending_history_id"]:
        auth.set_history_insights(session["pending_history_id"], user["user_id"], insights)

    return {
        "ok": True,
        "provider": session["provider_name"],
        "insights": insights,
    }
