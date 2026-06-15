# 🧠 KnowledgeHub AI

> Multi-document RAG (Retrieval-Augmented Generation) assistant.  
> Upload PDFs → ask questions → get answers grounded in your documents, with source citations.

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 PDF Upload | Drag-and-drop via Streamlit UI; up to 50 MB per file |
| 🔍 Text Extraction | PyMuPDF (primary) + pdfplumber (fallback) with per-page tracking |
| ✂️ Smart Chunking | Custom recursive splitter — no LangChain dependency |
| 🧬 Embeddings | `sentence-transformers` — runs **locally**, no embedding API key needed |
| 🗄️ Vector Store | Pure-NumPy cosine similarity — **no C++ build tools, works on Windows** |
| 🤖 LLM Answering | **Anthropic Claude**, **OpenAI GPT**, or **Ollama** (local, free) |
| 📚 Source Citations | Every answer links to the exact chunks it used |
| 💬 Chat History | Multi-turn conversation context passed to the LLM |
| 🐳 Docker | Single-container or docker-compose (separate services) |

---

## 🗂️ Project Structure

```
KnowledgeHub_AI/
├── backend/
│   ├── api/
│   │   ├── chat.py           # POST /chat/ — RAG endpoint
│   │   ├── documents.py      # CRUD for uploaded documents
│   │   └── health.py         # GET /health/
│   ├── core/
│   │   ├── config.py         # Pydantic settings (reads .env)
│   │   └── llm_client.py     # Anthropic / OpenAI / Ollama abstraction
│   ├── db/
│   │   ├── vector_store.py   # NumPy vector store
│   │   └── document_store.py # JSON metadata store
│   ├── models/
│   │   └── schemas.py        # Pydantic request/response models
│   ├── utils/
│   │   ├── pdf_extractor.py  # PDF → per-page text
│   │   └── chunker.py        # Text → overlapping chunks (pure Python)
│   └── main.py               # FastAPI app factory
├── frontend/
│   ├── components/
│   │   ├── document_panel.py # Sidebar upload/manage widget
│   │   └── source_viewer.py  # Citation cards
│   ├── utils/
│   │   └── api_client.py     # HTTP calls to backend
│   └── app.py                # Streamlit main app
├── data/
│   ├── uploads/              # Saved PDFs + doc_metadata.json
│   └── vectors/              # vectors.npy + metadata.json
├── scripts/
│   └── docker_entrypoint.sh
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10 – 3.12** (3.12 fully supported on Windows)
- One of: an Anthropic API key, an OpenAI API key, **or** Ollama installed locally
- No C++ compiler or build tools required

### 2. Clone & set up

```bash
git clone <repo>
cd KnowledgeHub_AI

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3. Configure

```bash
# macOS / Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Edit `.env` and choose your provider (see sections below).

### 4. Run (two terminals)

**Terminal 1 – Backend**
```bash
uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 – Frontend**
```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501** in your browser.

---

## 🤖 LLM Provider Setup

### Option A — Anthropic Claude (default)

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=claude-3-5-sonnet-20241022
```

Get a key at [console.anthropic.com](https://console.anthropic.com).

---

### Option B — OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
```

Get a key at [platform.openai.com](https://platform.openai.com).

---

### Option C — Ollama (local, free, no API key)

Ollama runs models entirely on your machine. No data leaves your computer.

**1. Install Ollama**

Download from [ollama.com](https://ollama.com) (Windows, macOS, Linux).

**2. Pull a model**

```bash
# Recommended default — fast, capable, low RAM
ollama pull qwen3:8b

# Alternative — Meta Llama 3
ollama pull llama3

# Other options
ollama pull mistral
ollama pull phi3
ollama pull gemma2
```

**3. Start the Ollama server** (usually auto-starts after install)

```bash
ollama serve
```

**4. Update `.env`**

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

> **Memory requirements**
> | Model | VRAM / RAM needed |
> |---|---|
> | `qwen3:8b` | ~6 GB |
> | `llama3` (8B) | ~6 GB |
> | `mistral` (7B) | ~5 GB |
> | `phi3:mini` (3.8B) | ~3 GB |
> | `gemma2:2b` | ~2 GB |

Ollama can run on CPU if you don't have a GPU — just expect slower responses.

---

## 🪟 Windows Notes

- **No C++ Build Tools needed** — ChromaDB has been replaced with a pure-NumPy vector store.
- All dependencies install from pre-built wheels via `pip`.
- Ollama has a native Windows installer — fully supported.
- If you see a `torch` warning from `sentence-transformers`, it is non-fatal; CPU inference still works.
- Reset the vector index by deleting `data\vectors\`.

---

## 🐳 Docker

### docker-compose (recommended)

```bash
cp .env.example .env   # fill in your provider settings
docker compose up --build
```

Services:
- **Backend** → http://localhost:8000 (API docs at `/docs`)
- **Frontend** → http://localhost:8501

> **Ollama + Docker**: point `OLLAMA_BASE_URL` at your host machine:
> ```env
> # Linux / Windows WSL2
> OLLAMA_BASE_URL=http://172.17.0.1:11434
> # macOS (Docker Desktop)
> OLLAMA_BASE_URL=http://host.docker.internal:11434
> ```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `ollama` |
| `ANTHROPIC_API_KEY` | — | Anthropic secret key |
| `LLM_MODEL` | `claude-3-5-sonnet-20241022` | Model for Anthropic or OpenAI |
| `OPENAI_API_KEY` | — | OpenAI secret key |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3:8b` | Ollama model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-Transformers model (local) |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `UPLOAD_DIR` | `data/uploads` | PDF storage path |
| `VECTOR_DIR` | `data/vectors` | NumPy vector store path |

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health/` | Liveness check + stats |
| `POST` | `/documents/upload` | Upload and index a PDF |
| `GET` | `/documents/` | List all indexed documents |
| `DELETE` | `/documents/{doc_id}` | Delete a document and its chunks |
| `POST` | `/chat/` | Ask a question (RAG) |

Full interactive docs: **http://localhost:8000/docs**

---

## 📝 License

MIT
