# Logistics RAG Chatbot

Production-oriented RAG chatbot for a logistics company website.

## Current Scope

This first increment includes:

- Project folder structure.
- Python dependencies.
- Environment variable template.
- OCR-based PDF ingestion pipeline.
- Idempotent ingestion manifest to avoid recreating embeddings for unchanged PDFs.
- Pinecone upsert with page, chunk, document, and source metadata.
- LangGraph chatbot orchestration with intent routing.
- Pinecone retrieval and Groq answer generation modules.
- FastAPI `/chat` and `/health` endpoints.
- Temporary Streamlit testing interface.

## Setup

1. Create and activate a Python 3.12+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR on the host machine and ensure `tesseract` is on `PATH`.
4. Create `.env` from `.env.example` and fill in the private values.

## Ingest Documents

Place company PDFs in `data/`, then run:

```bash
python ingest.py
```

To ingest a specific PDF:

```bash
python ingest.py --pdf "data/paramount company data.pdf"
```

To force regeneration and upsert even if the PDF has not changed:

```bash
python ingest.py --force
```

## Run API

```bash
.venv\Scripts\python.exe app.py
```

The API exposes:

- `GET /health`
- `POST /chat`

Example request:

```json
{
  "message": "What services does the company offer?",
  "session_id": "optional-session-id"
}
```

## Run Streamlit Test UI

```bash
.venv\Scripts\streamlit.exe run frontend/streamlit_app.py
```

or:

```bash
.venv\Scripts\streamlit.exe run streamlit_app.py
```

## Notes

- The embedding model is `BAAI/bge-base-en-v1.5` with 768-dimensional vectors.
- Pinecone index creation is expected to be handled before ingestion so deployment teams can choose cloud, region, and index type deliberately.
- Retrieved chunk text is stored in Pinecone metadata under `text` for the chatbot module.
- Tracking, pricing, and custom logistics requests are handled by deterministic intent routing before retrieval.
- API keys and contact details belong in `.env`, not in source control.
