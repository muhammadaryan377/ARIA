"""ARIA FastAPI application entry point.

Wires together the route modules under ``api/`` and exposes the public
endpoints (``/``, ``/api/providers``). Run with:

    python app.py
    uvicorn app:app
"""

import os
import socket
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Make the project root importable regardless of the launching directory.
# core.config loads .env centrally on first import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.config import STATIC_DIR  # noqa: E402
from core.checks import run_checks  # noqa: E402
from llm_provider import PROVIDERS  # noqa: E402
from api import auth_routes, connect_routes, query_routes, schema_routes  # noqa: E402

app = FastAPI(title="ARIA Testing UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_routes.router)
app.include_router(connect_routes.router)
app.include_router(schema_routes.router)
app.include_router(query_routes.router)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    """Return a clean JSON 500 with guidance instead of a raw stack trace page."""
    import logging

    logging.getLogger("uvicorn.error").exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal error: {type(exc).__name__}. "
            "Check the server logs for the full traceback."
        },
    )


@app.on_event("startup")
def startup_checks():
    """Log a health report at startup; the app still starts on non-critical gaps."""
    import logging

    from core.checks import run_checks

    report = run_checks()
    critical = report["critical"]
    if critical:
        logging.getLogger("uvicorn.error").error(
            "ARIA startup check: missing required packages -> %s", ", ".join(critical)
        )
    if not report["ollama"]["ok"]:
        logging.getLogger("uvicorn.error").warning(
            "ARIA startup check: %s", report["ollama"]["detail"]
        )
    logging.getLogger("uvicorn.error").info("ARIA startup checks complete.")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    """Live dependency/health status for ops and the UI."""
    return run_checks()


@app.get("/api/providers")
def providers():
    """Return provider metadata (labels + privacy context) for the UI."""
    return {
        "providers": {
            name: {"label": meta["label"], "privacy": meta["privacy"], "models": meta["models"]}
            for name, meta in PROVIDERS.items()
        }
    }


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _resolve_port(start_port: int = 8000, max_tries: int = 20) -> int:
    """Return the first free local port starting from the requested value."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free local port found between {start_port} and {start_port + max_tries - 1}.")


if __name__ == "__main__":
    import uvicorn

    requested_port = int(os.getenv("ARIA_PORT", "8000"))
    port = _resolve_port(requested_port)
    print(f"ARIA Testing UI running at http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
