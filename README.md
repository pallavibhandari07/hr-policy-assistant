# HR Policy Assistant

> **Generative AI Mid-Term Project — Enso Group HR Policy Q&A**
>
> An evidence-grounded Retrieval-Augmented Generation (RAG) system that answers questions about the Enso Group HR Policy Manual, cites the relevant page number, and strictly returns **"Not found"** when the document contains insufficient evidence.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                            │
│             (frontend/index.html + style.css + app.js)          │
└────────────────────────┬────────────────────────────────────────┘
                         │  POST /ask  (JSON)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI  (backend/api.py)                     │
│   • Input validation  • Error handling  • Static file serving   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RAG Chain  (backend/chain.py)                   │
│   ┌──────────────────────┐      ┌────────────────────────────┐  │
│   │ Retriever            │      │ Gemini LLM                 │  │
│   │ (backend/retriever)  │─────►│ (gemini-1.5-flash)         │  │
│   │                      │      │ temp=0, strict prompt      │  │
│   │ ChromaDB similarity  │      └────────────────────────────┘  │
│   │ search → top-K chunks│                                      │
│   └──────────────────────┘                                      │
│          ▲                                                       │
│          │ persistent                                            │
│   ┌──────┴──────────────────────────────────────────────────┐   │
│   │  ChromaDB  (vectorstore/)                               │   │
│   │  Embeddings: models/embedding-001 (Google)              │   │
│   └─────────────────────────────────────────────────────────┘   │
│          ▲                                                       │
│          │ one-time ingestion                                    │
│   ┌──────┴──────────────────────────────────────────────────┐   │
│   │  Ingestion Pipeline  (backend/ingest.py)                │   │
│   │  PyMuPDF loader → RecursiveCharacterTextSplitter        │   │
│   │  → deterministic chunk IDs → ChromaDB upsert            │   │
│   └─────────────────────────────────────────────────────────┘   │
│          ▲                                                       │
│          │                                                       │
│   data/hr_policy.pdf  (user-provided, gitignored)               │
└─────────────────────────────────────────────────────────────────┘
```

## Folder Structure

```
hr-policy-assistant/
├── data/                    # HR Policy PDF (gitignored, user-provided)
├── vectorstore/             # ChromaDB persistent storage (gitignored)
│
├── backend/
│   ├── config.py            # Settings from .env, path/key validation
│   ├── ingest.py            # PDF loading, chunking, embedding, ChromaDB write
│   ├── retriever.py         # Similarity search wrapper
│   ├── chain.py             # LangChain LCEL RAG chain + Gemini LLM
│   └── api.py               # FastAPI app (routes, validation, error handling)
│
├── frontend/
│   ├── index.html           # SPA shell with 5 UI states
│   ├── style.css            # Dark-mode design system
│   └── app.js               # State machine, fetch, safe DOM rendering
│
├── tests/
│   ├── conftest.py          # Shared fixtures, API key skip marker
│   ├── test_ingest.py       # Chunk ID, metadata, idempotency tests
│   ├── test_retrieval.py    # DB error handling + integration retrieval
│   ├── test_chain.py        # Context builder, "Not found" logic, integration
│   └── test_api.py          # FastAPI endpoint tests (validation, schemas, errors)
│
├── .env.example             # Template — copy to .env
├── .gitignore
├── requirements.txt
├── ingest_runner.py         # CLI: build the vector store
├── run.py                   # CLI: start the API server
└── README.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.10 or higher
- A Google Gemini API key ([get one here](https://aistudio.google.com/apikey))
- The Enso Group HR Policy Manual PDF

### 1. Clone and enter the repository
```bash
git clone https://github.com/<your-username>/hr-policy-assistant.git
cd hr-policy-assistant
```

### 2. Create a virtual environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Open .env and fill in:
#   GEMINI_API_KEY=your_key_here
#   PDF_PATH=data/hr_policy.pdf
```

### 4. Place the HR Policy PDF
```bash
# Copy the Enso Group HR Policy Manual into the data/ folder
cp /path/to/your/hr_policy.pdf data/hr_policy.pdf
```

### 5. Build the vector store (one-time, re-runnable safely)
```bash
python ingest_runner.py
```
Expected output:
```
✅  Ingestion complete in ~Xs
    Pages processed : N
    Chunks created  : M
    New chunks added: M
    Skipped (dupes) : 0
```

### 6. Start the server
```bash
python run.py
```

### 7. Open the app
Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Running Tests

```bash
# Unit tests only (no API key required)
pytest tests/test_ingest.py tests/test_api.py -v

# All tests including integration (requires GEMINI_API_KEY + populated DB)
pytest -v
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `gemini-1.5-flash` | Fast inference, generous free tier, large context window |
| `models/embedding-001` | Google's embedding model, compatible with Gemini API key |
| `temperature=0` | Deterministic, fact-based answers; prevents hallucination |
| Deterministic chunk IDs | SHA-256 of content prevents duplicate ingestion on re-runs |
| 1-indexed page numbers | Matches what users see in a PDF viewer |
| Strict "Not found" prompt | LLM cannot draw on training data; must cite or refuse |
| No `innerHTML` with server data | Prevents XSS in the frontend |
| `FileNotFoundError` → 503 | Clear user feedback when ingestion hasn't been run |

---

## Assignment Q&A (Verified Answers)

> ⚠️ Answers below will be filled in and verified against the actual PDF after ingestion.
> Do not record answers until cross-checked with source pages.

| # | Question | Answer | Source Page |
|---|---|---|---|
| 1 | How many days of Privilege Leave (PL) does a permanent employee get per year? | **18 days per year** | Page 18 & 60 |
| 2 | What is the probation period for employees up to Senior Manager? | **6 months** from the date of joining | Page 7 |
| 3 | What is the company's health-insurance premium amount? | **Not found** — not present in the HR manual | — |
| 4 | What is the current GST rate in India? | **Not found** — not present in the HR manual | — |
| 5 | How many days of maternity leave are given? | **26 weeks** (as per Maternity Benefit Act) | Page 19 & 60 |

> ✅ Answers 1, 2, and 5 verified against PDF source pages.
> ⚠️ Answers 3 and 4 correctly return **"Not found"** — these topics are not covered in the Enso Group HR Policy Manual.

Screenshots of verified answers are in the `screenshots/` folder.

---

## Technology Stack

- **Python 3.10+**
- **LangChain** (LCEL RAG pipeline)
- **Google Gemini API** (`gemini-1.5-flash` + `embedding-001`)
- **ChromaDB** (persistent local vector store)
- **FastAPI** + **Uvicorn** (API server)
- **HTML / CSS / Vanilla JavaScript** (frontend)
- **PyMuPDF** (PDF loading with page metadata)
- **pytest** + **httpx** (testing)

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | — | Google AI Studio API key |
| `PDF_PATH` | — | `data/hr_policy.pdf` | Path to HR Policy PDF |
| `CHROMA_DIR` | — | `vectorstore` | ChromaDB storage directory |
| `COLLECTION_NAME` | — | `hr_policy` | ChromaDB collection name |
| `CHUNK_SIZE` | — | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | — | `150` | Overlap between chunks |
| `TOP_K` | — | `5` | Chunks retrieved per query |
| `HOST` | — | `127.0.0.1` | Server host |
| `PORT` | — | `8000` | Server port |
