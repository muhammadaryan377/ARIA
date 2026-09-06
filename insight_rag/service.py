"""Advanced conversational PDF-RAG capability for ARIA's Insight Agent."""

from __future__ import annotations

import hashlib
import re
import shutil
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from .config import (
    MAX_CONTEXT_CHARS,
    MAX_PDF_MB,
    RECENT_HISTORY_TURNS,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_TOP_K,
    RAG_LLM_MODEL,
)
from .pdf_ingest import extract_pdf_documents
from .storage import ConversationStore, RAGMetadataStore, UserPGVectorStore


_TABLE_INTENT_WORDS = {
    "table", "total", "sum", "average", "avg", "maximum", "minimum", "max", "min",
    "highest", "lowest", "compare", "difference", "percent", "percentage", "amount",
    "revenue", "sales", "profit", "count", "how many", "which product", "which category",
}
_FOLLOWUP_PATTERNS = (
    "what about", "how about", "previous", "former", "latter", "that", "those", "these",
    "it", "them", "same", "above", "earlier", "before", "and in", "and what",
)
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "for", "and",
    "or", "with", "what", "which", "who", "when", "where", "why", "how", "did", "does",
    "do", "me", "show", "tell", "from", "according", "report", "document", "pdf",
}


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_.%-]*", (text or "").lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _to_provider_messages(messages) -> list[dict]:
    roles = {"system": "system", "human": "user", "ai": "assistant"}
    return [
        {"role": roles.get(getattr(message, "type", "human"), "user"), "content": str(message.content)}
        for message in messages
    ]


class InsightPDFRAG:
    """PDF retrieval capability owned by an existing `InsightAgent` instance.

    `insight_agent.llm` must be the cloud provider.  The capability does not
    introduce a fifth agent; it simply extends Insight Agent with document
    understanding and conversational retrieval.
    """

    def __init__(self, *, insight_agent, user_id: str | int):
        self.insight_agent = insight_agent
        self.llm = insight_agent.llm
        if getattr(self.llm, "provider", None) != "cloud":
            raise RuntimeError(
                "Insight PDF RAG is configured for the Cloud LLM. Initialise the Insight Agent "
                "with the cloud Groq provider before using PDF chat."
            )

        # Dedicated low-latency model roles for RAG. Existing SQL/story roles
        # remain untouched so this addition cannot change the current pipeline.
        self.llm.models["rag"] = RAG_LLM_MODEL
        self.llm.models["rag_rewrite"] = RAG_LLM_MODEL

        self.user_id = str(user_id)
        self.metadata = RAGMetadataStore(user_id)
        self.conversations = ConversationStore(user_id)

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def ingest_pdf(self, pdf_path: str | Path, *, original_filename: str) -> dict:
        path = Path(pdf_path)
        if path.suffix.lower() != ".pdf" or not original_filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported in the current Insight RAG version.")
        if not path.exists() or not path.is_file():
            raise ValueError("Uploaded PDF could not be found.")

        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise ValueError("Uploaded PDF is empty.")
        if size_bytes > MAX_PDF_MB * 1024 * 1024:
            raise ValueError(f"PDF is larger than the current {MAX_PDF_MB} MB limit.")
        if path.read_bytes()[:5] != b"%PDF-":
            raise ValueError("The uploaded file does not appear to be a valid PDF.")

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        existing = self.metadata.find_by_sha256(digest)
        if existing:
            return {"ok": True, "duplicate": True, "document": existing}

        document_id = str(uuid.uuid4())
        documents, stats = extract_pdf_documents(
            path,
            document_id=document_id,
            filename=Path(original_filename).name,
        )

        vector_store = UserPGVectorStore(self.user_id)
        chunk_ids = vector_store.add_documents(documents)

        destination = self.metadata.document_file_path(document_id)
        shutil.copyfile(path, destination)
        self.metadata.save_chunks(document_id, documents)

        from datetime import datetime, timezone

        record = {
            "document_id": document_id,
            "filename": Path(original_filename).name,
            "sha256": digest,
            "size_bytes": size_bytes,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pages": stats["pages"],
            "tables": stats["tables"],
            "text_chunks": stats["text_chunks"],
            "table_chunks": stats["table_chunks"],
            "total_chunks": stats["total_chunks"],
            "chunk_ids": chunk_ids,
        }
        self.metadata.put_document(record)
        return {"ok": True, "duplicate": False, "document": record}

    def list_documents(self) -> list[dict]:
        return self.metadata.list_documents()

    def delete_document(self, document_id: str) -> dict:
        record = self.metadata.get_document(document_id)
        if not record:
            raise ValueError("Document not found.")

        UserPGVectorStore(self.user_id).delete_chunks(record.get("chunk_ids", []))
        self.metadata.remove_document(document_id)
        self.metadata.document_file_path(document_id).unlink(missing_ok=True)
        self.metadata.chunks_path(document_id).unlink(missing_ok=True)
        return {"ok": True, "document_id": document_id}

    # ------------------------------------------------------------------
    # Conversation-aware retrieval
    # ------------------------------------------------------------------

    def _needs_rewrite(self, question: str, history: list[dict]) -> bool:
        if not history:
            return False
        lowered = question.lower().strip()
        if len(lowered.split()) <= 8:
            return True
        return any(pattern in lowered for pattern in _FOLLOWUP_PATTERNS)

    def _rewrite_question(self, question: str, history: list[dict]) -> str:
        if not self._needs_rewrite(question, history):
            return question

        history_text = "\n".join(
            f"{item.get('role', 'user').upper()}: {item.get('content', '')}"
            for item in history[-RECENT_HISTORY_TURNS * 2 :]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the latest user message into one standalone PDF search query. "
                    "Resolve pronouns, follow-ups, dates, comparisons and references using the chat history. "
                    "Do not answer the question. Do not add facts. Return only the rewritten query.",
                ),
                ("human", "Chat history:\n{history}\n\nLatest message:\n{question}"),
            ]
        )
        messages = prompt.format_messages(history=history_text, question=question)
        try:
            rewritten = self.llm.chat(
                "rag_rewrite",
                _to_provider_messages(messages),
                temperature=0.0,
                num_predict=120,
                timeout=15,
            ).strip()
            return rewritten or question
        except Exception:
            return question

    @staticmethod
    def _rerank(question: str, candidates: list[tuple[Document, float]]) -> list[Document]:
        """Fast local reranking: dense rank + lexical overlap + table intent boost.

        We deliberately avoid an additional cloud reranker request because the
        project prioritises low conversational latency.
        """
        query_tokens = _tokenize(question)
        lowered = question.lower()
        table_intent = any(word in lowered for word in _TABLE_INTENT_WORDS)
        scored = []
        seen = set()

        for dense_rank, (doc, _distance) in enumerate(candidates, start=1):
            chunk_id = str(doc.metadata.get("chunk_id", ""))
            if chunk_id and chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)

            doc_tokens = _tokenize(doc.page_content)
            overlap = len(query_tokens & doc_tokens) / max(1, len(query_tokens))
            dense_component = 1.0 / dense_rank
            table_boost = 0.18 if table_intent and doc.metadata.get("content_type") == "table" else 0.0
            scored.append((dense_component + (0.75 * overlap) + table_boost, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:RETRIEVAL_TOP_K]]

    def _retrieve(self, question: str, document_ids: list[str] | None) -> list[Document]:
        candidates = UserPGVectorStore(self.user_id).search(
            question,
            k=RETRIEVAL_FETCH_K,
            document_ids=document_ids,
        )
        return self._rerank(question, candidates)

    @staticmethod
    def _source_label(index: int) -> str:
        return f"S{index}"

    def _build_context(self, docs: list[Document]) -> tuple[str, list[dict]]:
        blocks = []
        sources = []
        used_chars = 0

        for index, doc in enumerate(docs, start=1):
            metadata = doc.metadata
            label = self._source_label(index)
            content_type = metadata.get("content_type", "text")
            table_index = metadata.get("table_index")
            title = f"[{label}] {metadata.get('filename')} — page {metadata.get('page')} — {content_type}"
            if content_type == "table" and table_index:
                title += f" {table_index}"
            block = f"{title}\n{doc.page_content.strip()}"
            if used_chars + len(block) > MAX_CONTEXT_CHARS and blocks:
                break
            blocks.append(block)
            used_chars += len(block)
            sources.append(
                {
                    "source_id": label,
                    "document_id": metadata.get("document_id"),
                    "filename": metadata.get("filename"),
                    "page": metadata.get("page"),
                    "content_type": content_type,
                    "table_index": table_index,
                    "chunk_id": metadata.get("chunk_id"),
                    "snippet": doc.page_content[:280].replace("\n", " "),
                }
            )
        return "\n\n".join(blocks), sources

    def _generate_answer(self, *, question: str, context: str, history: list[dict]) -> str:
        history_text = "\n".join(
            f"{item.get('role', 'user').upper()}: {item.get('content', '')}"
            for item in history[-RECENT_HISTORY_TURNS * 2 :]
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are ARIA's Insight Agent answering questions about uploaded PDFs. "
                    "Use ONLY the supplied PDF evidence. Do not use outside knowledge. "
                    "Tables are authoritative structured evidence: preserve row/column relationships and do not invent cells. "
                    "If the evidence does not support the answer, explicitly say the information was not found in the selected PDF evidence. "
                    "Cite factual statements using the supplied source labels such as [S1] or [S2]. "
                    "When comparing values, state the values and their source. Keep the response concise but complete.",
                ),
                (
                    "human",
                    "Recent conversation (for conversational context only):\n{history}\n\n"
                    "PDF evidence:\n{context}\n\n"
                    "Question: {question}\n\nAnswer using only the PDF evidence and source labels:",
                ),
            ]
        )
        messages = prompt.format_messages(
            history=history_text or "(no earlier conversation)",
            context=context,
            question=question,
        )
        return self.llm.chat(
            "rag",
            _to_provider_messages(messages),
            temperature=0.05,
            num_predict=900,
            timeout=30,
        ).strip()

    def chat(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        document_ids: list[str] | None = None,
    ) -> dict:
        question = (question or "").strip()
        if len(question) < 2:
            raise ValueError("Ask a question about the uploaded PDF.")
        if len(question) > 4000:
            raise ValueError("Question is too long.")

        available = {item["document_id"] for item in self.metadata.list_documents()}
        if not available:
            raise ValueError("Upload a text-based PDF before starting document chat.")
        if document_ids:
            unknown = [doc_id for doc_id in document_ids if doc_id not in available]
            if unknown:
                raise ValueError("One or more selected documents do not belong to this user.")
        else:
            document_ids = sorted(available)

        conversation_id = conversation_id or uuid.uuid4().hex
        payload = self.conversations.load(conversation_id)
        history = payload.get("messages", [])
        search_query = self._rewrite_question(question, history)
        docs = self._retrieve(search_query, document_ids)

        if not docs:
            answer = "I could not find supporting information in the selected PDF evidence."
            sources = []
        else:
            context, sources = self._build_context(docs)
            answer = self._generate_answer(question=question, context=context, history=history)

        self.conversations.append(conversation_id, "user", question)
        self.conversations.append(conversation_id, "assistant", answer, sources=sources)

        return {
            "ok": True,
            "conversation_id": conversation_id,
            "question": question,
            "search_query": search_query,
            "answer": answer,
            "sources": sources,
            "document_ids": document_ids,
            "retrieved_chunks": len(sources),
            "model": RAG_LLM_MODEL,
        }
