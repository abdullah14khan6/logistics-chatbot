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

## Current Ingestion Status

The local PDF `data/paramount company data.pdf` has been ingested into Pinecone index
`logistics-company-rag` with 24 vectors. The local ingestion manifest is ignored by Git
and will skip unchanged PDFs on later runs.

## Setup

1. Create and activate a Python 3.12+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR on the host machine and ensure `tesseract` is on `PATH`.
4. Create `.env` from `.env.example` and fill in the private values.

## Preflight Check

```bash
.venv\Scripts\python.exe preflight.py
```

This verifies installed packages, `.env`, PDF inputs, and Tesseract availability before ingestion.

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
- `POST /chat/clear/{session_id}`

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
- If `PINECONE_HOST` is empty, ingestion creates the Pinecone serverless index when missing.
- Pinecone index creation defaults to `PINECONE_CLOUD=aws` and `PINECONE_REGION=us-east-1`.
- Retrieved chunk text is stored in Pinecone metadata under `text` for the chatbot module.
- Tracking, pricing, and custom logistics requests are handled by deterministic intent routing before retrieval.
- API keys and contact details belong in `.env`, not in source control.
- If Tesseract is installed but not on `PATH`, set `TESSERACT_CMD` in `.env`.
- `CORS_ORIGINS` accepts a comma-separated list of allowed website/widget origins.
