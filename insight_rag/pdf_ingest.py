"""Text-based PDF ingestion with explicit table preservation.

Scope for v1:
- digitally generated/text PDFs
- normal paragraphs/headings
- tables extractable by pdfplumber

Scanned/image-only PDFs are deliberately rejected for now instead of silently
producing low-quality retrieval data.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import TABLE_ROWS_PER_CHUNK, TEXT_CHUNK_OVERLAP, TEXT_CHUNK_SIZE


_WHITESPACE_RE = re.compile(r"[ \t]+")


def _clean_text(text: str) -> str:
    lines = []
    for raw in (text or "").replace("\x00", " ").splitlines():
        line = _WHITESPACE_RE.sub(" ", raw).strip()
        if line:
            lines.append(line)
    return "\n".join(lines).strip()


def _clean_cell(value) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", str(value).replace("\n", " ")).strip()


def normalise_table(table: list[list]) -> list[list[str]]:
    """Normalize a pdfplumber table and remove fully empty rows/columns."""
    rows = [[_clean_cell(cell) for cell in row] for row in (table or []) if row]
    rows = [row for row in rows if any(cell for cell in row)]
    if not rows:
        return []

    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    keep_cols = [idx for idx in range(width) if any(row[idx] for row in padded)]
    return [[row[idx] for idx in keep_cols] for row in padded]


def table_to_markdown(rows: list[list[str]]) -> str:
    """Convert table rows to a retrieval-friendly Markdown table."""
    if not rows:
        return ""

    def esc(value: str) -> str:
        return value.replace("|", "\\|")

    header = rows[0]
    width = len(header)
    header = [cell or f"Column {i + 1}" for i, cell in enumerate(header)]
    body = [row + [""] * (width - len(row)) for row in rows[1:]]
    lines = [
        "| " + " | ".join(esc(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(esc(cell) for cell in row[:width]) + " |")
    return "\n".join(lines)


def _chunk_table(rows: list[list[str]], rows_per_chunk: int = TABLE_ROWS_PER_CHUNK) -> list[str]:
    if not rows:
        return []
    if len(rows) <= rows_per_chunk + 1:
        return [table_to_markdown(rows)]

    header = rows[0]
    chunks = []
    body = rows[1:]
    for start in range(0, len(body), rows_per_chunk):
        chunks.append(table_to_markdown([header] + body[start : start + rows_per_chunk]))
    return chunks


def _stable_chunk_id(document_id: str, kind: str, page: int, ordinal: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"aria-pdf:{document_id}:{kind}:{page}:{ordinal}"))


def extract_pdf_documents(
    pdf_path: str | Path,
    *,
    document_id: str,
    filename: str,
) -> tuple[list[Document], dict]:
    """Extract LangChain Documents for page text and tables.

    Returns `(documents, stats)`.  Every chunk carries filename, page number,
    content type, document id, and a stable chunk id for deletion/citations.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required. Run: pip install pymupdf") from exc

    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is required. Run: pip install pdfplumber") from exc

    path = Path(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    page_count = 0
    text_chars = 0
    table_count = 0
    text_chunk_count = 0
    table_chunk_count = 0

    with fitz.open(path) as fitz_doc, pdfplumber.open(path) as plumber_doc:
        page_count = len(fitz_doc)
        for page_index in range(page_count):
            page_number = page_index + 1
            page_text = _clean_text(fitz_doc[page_index].get_text("text"))
            text_chars += len(page_text)

            if page_text:
                chunks = splitter.split_text(page_text)
                for ordinal, chunk in enumerate(chunks):
                    chunk_id = _stable_chunk_id(document_id, "text", page_number, ordinal)
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "document_id": document_id,
                                "filename": filename,
                                "page": page_number,
                                "content_type": "text",
                                "chunk_id": chunk_id,
                            },
                        )
                    )
                    text_chunk_count += 1

            plumber_page = plumber_doc.pages[page_index]
            try:
                raw_tables = plumber_page.extract_tables() or []
            except Exception:
                raw_tables = []

            for table_index, raw_table in enumerate(raw_tables, start=1):
                rows = normalise_table(raw_table)
                if len(rows) < 2:
                    continue
                table_count += 1
                for ordinal, table_text in enumerate(_chunk_table(rows)):
                    if not table_text.strip():
                        continue
                    chunk_id = _stable_chunk_id(
                        document_id,
                        f"table-{table_index}",
                        page_number,
                        ordinal,
                    )
                    documents.append(
                        Document(
                            page_content=table_text,
                            metadata={
                                "document_id": document_id,
                                "filename": filename,
                                "page": page_number,
                                "content_type": "table",
                                "table_index": table_index,
                                "chunk_id": chunk_id,
                            },
                        )
                    )
                    table_chunk_count += 1

    # Digital/text PDFs should expose a meaningful amount of text.  Reject
    # scanned/image-only files explicitly so users are not given false answers.
    if text_chars < max(80, page_count * 20):
        raise ValueError(
            "This PDF appears to be scanned/image-only or has too little extractable text. "
            "The current Insight RAG version supports text-based PDFs only."
        )

    if not documents:
        raise ValueError("No usable text or tables could be extracted from this PDF.")

    return documents, {
        "pages": page_count,
        "text_chars": text_chars,
        "tables": table_count,
        "text_chunks": text_chunk_count,
        "table_chunks": table_chunk_count,
        "total_chunks": len(documents),
    }
