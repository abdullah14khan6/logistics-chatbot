# Paramount Logistics Support Assistant

Production-oriented customer-support chatbot with semantic conversation planning,
structured company facts, retrieval-augmented answers, and deterministic contact routing.

## Runtime Architecture

Each non-social message follows this path:

1. Load structured session state and recent meaningful conversation turns.
2. Use the small Groq planner to resolve intent, references, shipment entities, and actions.
3. Read exact facts and authorized contacts from `data/company_profile.json`.
4. Retrieve filtered Pinecone evidence only when descriptive company information is needed.
5. Generate a concise grounded answer and append deterministic tracking or contact details.
6. Store updated shipment context, pending questions, topics, and contact-disclosure recency.

Conversation state uses SQLite by default in the supplied production configuration, with
expiry and session limits. Set `MEMORY_BACKEND=memory` for disposable local tests.

Pure greetings, gratitude, farewells, small talk, and acknowledgements without a pending
question use a local fast path. They are recorded but compacted out of semantic history.

## Setup

Use Python 3.12 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For development and tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and provide the Groq and Pinecone credentials. Install
Tesseract OCR and set `TESSERACT_CMD` when it is not available on `PATH`.

## Company Facts

Business facts that require exact answers live in `data/company_profile.json`, including:

- Office addresses and hours.
- Services and value-added services.
- Payment, restricted-item, and delay policies.
- Public departmental contacts and contact fallback rules.

Contacts are never selected from arbitrary generated text. The controller resolves them from
the profile only after an explicit contact request or a quotation/handoff action.

## Preflight

```powershell
.\.venv\Scripts\python.exe preflight.py
```

This validates dependencies, configuration, Tesseract, the structured company profile,
and source PDFs.

## Ingestion

```powershell
.\.venv\Scripts\python.exe ingest.py
```

Use `--force` after changing extraction, chunking, or metadata:

```powershell
.\.venv\Scripts\python.exe ingest.py --force
```

The ingestion pipeline uses native PDF text when it is usable and OCR only for image-heavy
pages. Chunks include document version, page, section, content type, and extraction method.
Existing vectors for the document are removed before replacement to prevent stale content.

The current source PDF is represented by 23 vectors in the `company-docs` namespace.

## Run the API

```powershell
.\.venv\Scripts\python.exe app.py
```

Endpoints:

- `GET /health` - process health.
- `GET /ready` - embedding/Pinecone warmup readiness.
- `POST /chat` - send a message and optional session ID.
- `POST /chat/clear/{session_id}` - clear conversation state.

Set `PREWARM_ON_STARTUP=true` in production. The API starts loading the local embedding model
immediately, and `/ready` returns `503` until warmup completes. Set `PINECONE_HOST` to the
index data-plane host to avoid an extra control-plane lookup.

## Run the Streamlit Test UI

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

When startup warmup is enabled, Streamlit begins loading retrieval dependencies as soon as
the page opens. Exact social and structured-profile answers remain independent of retrieval.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The suite covers semantic plan validation, multi-turn reference resolution, pending
acknowledgements, contact recency and fallback, email privacy, structured facts, retrieval
filtering and caching, ingestion metadata, API validation, and social fast paths.
