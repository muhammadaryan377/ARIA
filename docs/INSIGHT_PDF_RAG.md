# ARIA Insight Agent — Conversational PDF RAG (v1)

This module extends the existing **Insight Agent**. It is not a fifth autonomous agent.

## Current scope

Supported now:

- text-based PDF files
- multi-page PDFs
- paragraphs/headings
- extractable PDF tables
- local embeddings (`BAAI/bge-small-en-v1.5` through FastEmbed/ONNX)
- PostgreSQL + pgvector vector storage through LangChain
- hybrid retrieval: dense semantic search + local lexical search + Reciprocal Rank Fusion (RRF)
- table-aware retrieval and full-table chunk expansion
- conversational follow-up rewriting
- GPT-OSS-20B on Groq for grounded answer generation
- page/table citations
- per-user document isolation and persistent conversation history

Not supported in v1:

- scanned/image-only PDFs
- handwritten PDFs
- chart/image understanding
- DOCX/Excel

## Architecture

```text
PDF upload
   |
   +--> PyMuPDF text extraction
   |
   +--> pdfplumber table extraction
             |
             v
      LangChain Documents
             |
      structure-aware chunks
             |
   FastEmbed BGE-small (local)
             |
       LangChain PGVector
             |
User question + conversation history
             |
     optional query rewrite
             |
   dense retrieval + lexical retrieval
             |
             RRF
             |
      table sibling expansion
             |
       GPT-OSS-20B / Groq
             |
   grounded answer + source pages
```

## 1. Install dependencies

```powershell
pip install -r requirements.txt
```

FastEmbed downloads `BAAI/bge-small-en-v1.5` the first time it is used and caches it locally. It is not downloaded for every query.

## 2. Create the RAG database

Use a separate PostgreSQL database so ARIA never needs to write vector tables into the user's analytics/source database.

From `psql`:

```sql
CREATE DATABASE aria_rag;
```

Then connect to it:

```powershell
psql -U postgres -d aria_rag
```

Enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

If PostgreSQL reports that the `vector` extension is unavailable, install pgvector for the PostgreSQL version on the machine first.

## 3. Configure `.env`

```env
GROQ_API_KEY=your_key_here
ARIA_RAG_DATABASE_URL=postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/aria_rag
ARIA_RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARIA_RAG_LLM_MODEL=openai/gpt-oss-20b
ARIA_RAG_MAX_PDF_MB=50
```

`langchain-postgres` uses psycopg3, so use `postgresql+psycopg://` in the RAG connection URL.

## 4. Run ARIA

```powershell
python app.py
```

Open the normal ARIA UI and sign in first. Then open:

```text
http://127.0.0.1:8000/pdf-chat
```

If port 8000 is already busy, ARIA automatically chooses the next free port. Use the port printed in the terminal.

## PDF RAG API

### Upload/index PDF

`POST /api/insight/pdf/upload`

Multipart form field: `file`

### List indexed PDFs

`GET /api/insight/pdf/documents`

### Delete an indexed PDF

`DELETE /api/insight/pdf/documents/{document_id}`

### Conversational question

`POST /api/insight/pdf/chat`

Example body:

```json
{
  "question": "Which product had the highest revenue?",
  "conversation_id": null,
  "document_ids": ["<document-id>"]
}
```

The response includes:

- `answer`
- `conversation_id`
- rewritten `search_query` when needed
- `sources` with filename/page/table metadata
- retrieval mode (`hybrid_dense_lexical_rrf`)

## Important design decisions

1. **RAG stays inside Insight Agent.** The four-agent FYP architecture is unchanged.
2. **Embeddings stay local.** Only relevant retrieved evidence is sent to the cloud LLM.
3. **User isolation is enforced.** Each authenticated user has a separate PGVector collection and separate local metadata/history folder.
4. **Tables are not flattened into arbitrary prose.** They are preserved as Markdown tables with page/table metadata and large tables are chunked by rows with repeated headers.
5. **Scanned PDFs fail explicitly.** v1 does not pretend to understand image-only documents.
