# 🧠 NotesMind — Local RAG Chatbot

> Chat with your own documents — completely offline. No API keys, no data leaving your machine.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Ollama](https://img.shields.io/badge/Ollama-Mistral_7B-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-orange)

## What it does

NotesMind lets you ask natural language questions over any set of documents (.pdf, .md, .txt, .docx) using a fully offline RAG pipeline powered by Mistral 7B running locally via Ollama.

## Features

- **Hybrid retrieval** — BM25 keyword search + dense vector search fused with reciprocal rank fusion
- **Cross-encoder reranking** — precision pass after initial retrieval
- **OCR support** — scanned PDFs are automatically detected and processed with Tesseract
- **Conversation memory** — follow-up questions work naturally
- **Live document ingestion** — upload new files without restarting
- **FastAPI backend** — clean REST API with no timeout issues
- **Source citations** — every answer shows which chunks it came from

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Mistral 7B via Ollama (fully offline) |
| Vector DB | ChromaDB |
| Orchestration | LlamaIndex |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Reranker | cross-encoder/ms-marco-TinyBERT-L-2-v2 |
| Backend | FastAPI + Uvicorn |
| OCR | Tesseract + pdf2image + PyMuPDF |
| Frontend | Vanilla HTML/CSS/JS |

## Architecture

```
Your docs (PDF / MD / TXT / DOCX)
          ↓
  OCR detection + text extraction
          ↓
  SentenceSplitter chunking (256 tokens, 32 overlap)
          ↓
  BGE-small embeddings
          ↓
  ChromaDB vector store
          ↓
  User query
          ↓
  Hybrid retrieval (BM25 + dense vector, top-5 each)
          ↓
  Reciprocal rank fusion
          ↓
  Cross-encoder reranking (top-3)
          ↓
  Mistral 7B via Ollama
          ↓
  Answer + source citations
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract/releases/latest) — for scanned PDFs
- [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/latest) — Windows only, required by pdf2image

## Installation

```bash
git clone https://github.com/DevendraSolanki1508/NotesMind-Local-RAG-Chatbot.git
cd NotesMind-Local-RAG-Chatbot

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
ollama pull mistral
```

## Usage

```bash
# Step 1 — Add your documents
# Drop any .pdf, .txt, .md, or .docx files into the docs/ folder

# Step 2 — Index documents (run once, or after adding new files)
python ingest.py

# Step 3 — Start the server
uvicorn server:app --host 0.0.0.0 --port 8000

# Step 4 — Open in browser
# http://localhost:8000
```

## Project Structure

```
NotesMind/
├── docs/              ← drop your documents here
├── chroma_db/         ← auto-managed by ChromaDB (gitignored)
├── ingest.py          ← OCR + chunking + embedding + indexing pipeline
├── query_engine.py    ← hybrid retrieval + reranking + conversation memory
├── server.py          ← FastAPI backend (REST API)
├── index.html         ← chat UI (vanilla HTML/CSS/JS)
├── evaluate.py        ← RAGAS evaluation pipeline
├── requirements.txt   ← all dependencies
└── README.md
```

## How it works

**Ingestion phase** — documents are loaded and checked for text content. Scanned PDFs with no extractable text are automatically routed through Tesseract OCR. All text is chunked using LlamaIndex's SentenceSplitter with overlap to avoid losing context at boundaries. Chunks are embedded using BGE-small and stored in ChromaDB on disk.

**Query phase** — each user question triggers two parallel retrievals: a BM25 keyword search and a dense vector similarity search. Results are merged using reciprocal rank fusion, then reranked by a cross-encoder model for precision. The top 3 chunks are passed as context to Mistral 7B, which generates a grounded answer.

**Why hybrid search** — dense embeddings miss exact terminology (function names, IDs, technical terms) while BM25 misses semantic similarity. Combining both improves recall significantly over either alone.

**Why reranking** — bi-encoder retrieval (used in vector search) scores query and document independently, which is fast but imprecise. Cross-encoder reranking scores each (query, chunk) pair together, giving much higher precision as a second pass.

## Evaluation

Run the RAGAS evaluation script to measure pipeline quality:

```bash
python evaluate.py
```

Metrics evaluated:
- **Faithfulness** — is the answer grounded in retrieved context?
- **Answer relevancy** — does the answer address the question?
- **Context precision** — are the retrieved chunks relevant?

## Resume / Interview talking points

- Implemented hybrid BM25 + dense vector retrieval because dense embeddings alone miss exact keyword matches — especially important for technical documents
- Used cross-encoder reranking as a precision pass after initial retrieval, since bi-encoder scoring has known precision limitations
- Built OCR fallback detection — system automatically identifies scanned PDFs and routes them through Tesseract
- Chunk size of 256 tokens with 32-token overlap chosen to balance context completeness against retrieval noise
- Evaluated pipeline using RAGAS metrics (faithfulness, answer relevancy, context precision)