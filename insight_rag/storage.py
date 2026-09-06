"""Persistent document metadata, conversation history, and PGVector access."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document

from .config import RAG_COLLECTION_PREFIX, RAG_DATABASE_URL, user_dir, user_key
from .embeddings import FastBGEEmbeddings


_LOCK = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


class RAGMetadataStore:
    def __init__(self, user_id: str | int):
        self.user_id = str(user_id)
        self.root = user_dir(user_id)
        self.manifest_path = self.root / "manifest.json"

    def load_manifest(self) -> dict:
        with _LOCK:
            return _read_json(self.manifest_path, {"documents": {}})

    def save_manifest(self, manifest: dict) -> None:
        with _LOCK:
            _atomic_write(self.manifest_path, manifest)

    def list_documents(self) -> list[dict]:
        docs = list(self.load_manifest().get("documents", {}).values())
        docs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return docs

    def get_document(self, document_id: str) -> dict | None:
        return self.load_manifest().get("documents", {}).get(document_id)

    def find_by_sha256(self, digest: str) -> dict | None:
        for doc in self.list_documents():
            if doc.get("sha256") == digest:
                return doc
        return None

    def put_document(self, metadata: dict) -> None:
        manifest = self.load_manifest()
        manifest.setdefault("documents", {})[metadata["document_id"]] = metadata
        self.save_manifest(manifest)

    def remove_document(self, document_id: str) -> dict | None:
        manifest = self.load_manifest()
        removed = manifest.setdefault("documents", {}).pop(document_id, None)
        self.save_manifest(manifest)
        return removed

    def chunks_path(self, document_id: str) -> Path:
        return self.root / "chunks" / f"{document_id}.json"

    def save_chunks(self, document_id: str, documents: list[Document]) -> None:
        payload = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in documents
        ]
        _atomic_write(self.chunks_path(document_id), payload)

    def document_file_path(self, document_id: str) -> Path:
        return self.root / "documents" / f"{document_id}.pdf"


class ConversationStore:
    def __init__(self, user_id: str | int):
        self.root = user_dir(user_id) / "conversations"

    @staticmethod
    def _safe_id(conversation_id: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", conversation_id or ""):
            raise ValueError("Invalid conversation id.")
        return conversation_id

    def _path(self, conversation_id: str) -> Path:
        return self.root / f"{self._safe_id(conversation_id)}.json"

    def load(self, conversation_id: str) -> dict:
        return _read_json(
            self._path(conversation_id),
            {"conversation_id": conversation_id, "created_at": _utc_now(), "messages": []},
        )

    def append(self, conversation_id: str, role: str, content: str, sources=None) -> None:
        with _LOCK:
            payload = self.load(conversation_id)
            payload.setdefault("messages", []).append(
                {
                    "role": role,
                    "content": content,
                    "sources": sources or [],
                    "created_at": _utc_now(),
                }
            )
            payload["updated_at"] = _utc_now()
            _atomic_write(self._path(conversation_id), payload)


class UserPGVectorStore:
    """One isolated LangChain PGVector collection per authenticated user."""

    def __init__(self, user_id: str | int):
        if not RAG_DATABASE_URL:
            raise RuntimeError(
                "ARIA_RAG_DATABASE_URL is not configured. Set it to a PostgreSQL database "
                "with the pgvector extension enabled, e.g. "
                "postgresql+psycopg://postgres:password@localhost:5432/aria_rag"
            )

        try:
            from langchain_postgres import PGVector
        except ImportError as exc:
            raise RuntimeError(
                "langchain-postgres is required for PDF RAG. Run: pip install langchain-postgres"
            ) from exc

        connection = RAG_DATABASE_URL
        if connection.startswith("postgresql://"):
            connection = connection.replace("postgresql://", "postgresql+psycopg://", 1)
        elif connection.startswith("postgres://"):
            connection = connection.replace("postgres://", "postgresql+psycopg://", 1)
        elif connection.startswith("postgresql+psycopg2://"):
            connection = connection.replace("postgresql+psycopg2://", "postgresql+psycopg://", 1)

        collection = f"{RAG_COLLECTION_PREFIX}_{user_key(user_id)}"
        self.store = PGVector(
            embeddings=FastBGEEmbeddings(),
            collection_name=collection,
            connection=connection,
            use_jsonb=True,
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        ids = [str(doc.metadata["chunk_id"]) for doc in documents]
        self.store.add_documents(documents=documents, ids=ids)
        return ids

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.store.delete(ids=chunk_ids)

    def search(self, query: str, *, k: int, document_ids: list[str] | None = None):
        metadata_filter = None
        if document_ids:
            if len(document_ids) == 1:
                metadata_filter = {"document_id": document_ids[0]}
            else:
                metadata_filter = {"document_id": {"$in": document_ids}}
        return self.store.similarity_search_with_score(
            query=query,
            k=k,
            filter=metadata_filter,
        )
