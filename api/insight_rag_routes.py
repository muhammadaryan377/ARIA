"""PDF-RAG endpoints for the Insight Agent.

The API instantiates the existing InsightAgent and attaches its PDF RAG
capability; RAG is not exposed as a separate autonomous agent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from core.deps import require_writable
from insight_agent import InsightAgent
from insight_rag import InsightPDFRAG
from insight_rag.config import RAG_LLM_MODEL
from llm_provider import create_provider

router = APIRouter(prefix="/api/insight/pdf", tags=["Insight PDF RAG"])


class PDFChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    conversation_id: str | None = None
    document_ids: list[str] | None = None


def _capability(user_id: str | int) -> InsightPDFRAG:
    """Build the Insight Agent with the dedicated fast cloud model for PDF RAG."""
    provider = create_provider(
        "cloud",
        models={
            "rag": RAG_LLM_MODEL,
            "rag_rewrite": RAG_LLM_MODEL,
        },
    )
    insight_agent = InsightAgent(provider=provider)
    return InsightPDFRAG(insight_agent=insight_agent, user_id=user_id)


@router.post("/upload")
def upload_pdf(
    file: UploadFile = File(...),
    user: dict = Depends(require_writable),
):
    filename = Path(file.filename or "document.pdf").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp_path = Path(temp.name)
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                temp.write(chunk)
        result = _capability(user["user_id"]).ingest_pdf(temp_path, original_filename=filename)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PDF ingestion failed: {exc}")
    finally:
        try:
            file.file.close()
        except Exception:
            pass
        if temp_path:
            temp_path.unlink(missing_ok=True)


@router.get("/documents")
def list_pdf_documents(user: dict = Depends(require_writable)):
    try:
        documents = _capability(user["user_id"]).list_documents()
        # Chunk ids are internal implementation details and are intentionally not
        # returned to the browser.
        public_docs = [
            {key: value for key, value in document.items() if key != "chunk_ids"}
            for document in documents
        ]
        return {"ok": True, "documents": public_docs}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.delete("/documents/{document_id}")
def delete_pdf_document(document_id: str, user: dict = Depends(require_writable)):
    try:
        return _capability(user["user_id"]).delete_document(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/chat")
def chat_with_pdfs(request: PDFChatRequest, user: dict = Depends(require_writable)):
    try:
        return _capability(user["user_id"]).chat(
            request.question,
            conversation_id=request.conversation_id,
            document_ids=request.document_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Insight PDF chat failed: {exc}")
