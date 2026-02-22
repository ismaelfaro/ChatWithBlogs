# ChatMyIdeas — Blog Digital Twin

Turn any blog into a conversational digital twin of its author(s).
ChatMyIdeas fetches all articles, indexes them with local embeddings and a vector database, and lets you chat with the author's ideas via a locally-running LLM.

**Everything runs locally — no API keys needed.**

---

## Features

- **RSS / HTML ingestion** — tries the feed first, falls back to HTML scraping
- **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~90 MB)
- **Local LLM via Ollama** — any model you have installed (`llama3.2` default)
- **Pluggable vector store**
  - `chroma` (default) — persistent SQLite, zero extra services
  - `redis` — Redis Stack for higher throughput
- **REST API** — documented at `/docs` (Swagger UI)
- **Web chat UI** — built-in, served at `/`
- **Multi-author support** — chat with a specific author or all at once
- **Conversation history** — multi-turn context sent to the LLM

---

## Quick Start

### 1. Prerequisites

| Dependency | Install |
|------------|---------|
| Python 3.11+ | [python.org](https://www.python.org/downloads/) |
| Ollama | [ollama.com](https://ollama.com) |
| A small LLM | `ollama pull llama3.2` |

### 2. Install Python dependencies

```bash
git clone https://github.com/your-org/chatmyideas.git
cd chatmyideas
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Pull the model

```bash
ollama pull lfm2.5
```

### 4. Configure (optional)

```bash
cp .env.example .env
# Edit .env to change the model, vector store, ports, etc.
```

### 5. Run

```bash
python -m uvicorn app.main:app --reload
```

Open **http://localhost:8000** in your browser.

---

## Usage

### Web UI

1. Paste a blog URL (e.g. `https://simonwillison.net`) into the sidebar.
2. Click **Create Twin** — ChatMyIdeas fetches and indexes all articles.
3. Select an author (if there are multiple).
4. Start chatting.

### REST API

```bash
# Ingest a blog
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://simonwillison.net"}'

# Chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "blog_id": "<id from ingest response>",
    "message": "What is your take on LLMs?",
    "history": []
  }'

# List ingested blogs
curl http://localhost:8000/api/v1/blogs

# Delete a blog
curl -X DELETE http://localhost:8000/api/v1/blogs/<blog_id>

# Health check
curl http://localhost:8000/api/v1/health
```

Full interactive docs: **http://localhost:8000/docs**

---

## Configuration Reference

All settings can be overridden via environment variables or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE` | `chroma` | `chroma` or `redis` |
| `CHROMA_DB_PATH` | `./data/chroma` | Persistence path for ChromaDB |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `REDIS_INDEX_NAME` | `chatmyideas_idx` | Redis search index name |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name (must be pulled first) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBEDDING_DIMENSION` | `384` | Must match the embedding model |
| `CHUNK_SIZE` | `500` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `TOP_K_CHUNKS` | `5` | Chunks retrieved per query |
| `APP_HOST` | `0.0.0.0` | Bind address |
| `APP_PORT` | `8000` | Port |
| `DEBUG` | `false` | Enable debug logging and hot reload |

### Recommended small models

```bash
ollama pull lfm2.5          # Liquid AI LFM 2.5 (default)
ollama pull llama3.2        # 3B — great balance, ~2 GB
ollama pull phi3:mini       # 3.8B — fast, low memory
ollama pull gemma2:2b       # 2B — very small
```

Set your choice in `.env`:
```
OLLAMA_MODEL=lfm2.5
```

### Using Redis instead of ChromaDB

```bash
# Start Redis Stack
docker run -p 6379:6379 redis/redis-stack-server:latest

# Install the redis package
pip install redis

# Configure
echo "VECTOR_STORE=redis" >> .env
echo "REDIS_URL=redis://localhost:6379" >> .env
```

---

## Project Structure

```
chatmyideas/
├── app/
│   ├── api/
│   │   └── routes.py          # REST endpoints
│   ├── core/
│   │   ├── ingestion.py       # RSS + HTML blog fetching
│   │   ├── rag.py             # RAG pipeline (chunk/embed/retrieve/generate)
│   │   └── vectorstore/
│   │       ├── base.py        # Abstract interface
│   │       ├── chroma.py      # ChromaDB (SQLite) backend
│   │       └── redis_store.py # Redis Stack backend
│   ├── models/
│   │   └── schemas.py         # Pydantic request/response models
│   ├── static/
│   │   └── index.html         # Web chat UI
│   └── main.py                # FastAPI app + lifespan
├── tests/
│   ├── conftest.py            # Fixtures (in-memory store, mock embedder)
│   ├── test_api.py            # API endpoint tests
│   ├── test_ingestion.py      # Blog ingestion tests
│   └── test_rag.py            # RAG pipeline tests
├── config.py                  # Pydantic-settings config
├── requirements.txt
├── pyproject.toml
├── .env.example
├── LICENSE                    # Apache 2.0
└── README.md
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run a specific module:

```bash
pytest tests/test_api.py -v
```

Tests use an **in-memory vector store** and a **mock embedder** — Ollama and ChromaDB are not required to run the suite.

---

## How It Works

```
Blog URL
   │
   ▼
 Ingestion (feedparser + trafilatura)
   │  Articles: {title, content, author, url}
   ▼
 Chunking  (overlapping word windows)
   │  Chunks: list[str]
   ▼
 Embedding  (sentence-transformers, local)
   │  Vectors: list[float[384]]
   ▼
 Vector Store  (ChromaDB / Redis)
   │  Persisted with metadata
   ▼
 Chat query
   │
   ├─ Embed query ──► Similarity search ──► Top-K chunks
   │
   └─ Build prompt:
        System: "You are a digital twin of {author}…\n{retrieved chunks}"
        History: prior turns
        User: current message
   │
   ▼
 Ollama (local LLM)
   │
   ▼
 Response + source citations
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
