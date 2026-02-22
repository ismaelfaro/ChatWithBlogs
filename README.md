# ChatMyIdeas — Browser Local

> **Branch:** `browser-local`
> All AI runs inside the browser — no Ollama, no Python vector store, no GPU server needed.

Turn any blog into a conversational digital twin of its author.
Embeddings and LLM inference happen entirely in your browser via **WebGPU** and **WebAssembly**. The Python server only proxies external URLs to bypass CORS.

---

## How it works

```
Blog URL
   │
   ▼  (server-side CORS proxy)
 Fetch HTML / RSS
   │
   ▼  (browser — Readability.js)
 Extract article text
   │
   ▼  (browser — Transformers.js · all-MiniLM-L6-v2)
 Chunk + embed  →  IndexedDB (persisted locally)
   │
 Chat query
   ├─ Embed query  ──► cosine similarity  ──► top-K chunks
   └─ Build prompt ("You are a digital twin of …")
              │
              ▼  (browser — WebLLM · WebGPU)
           LLM response (streamed token by token)
```

**Nothing leaves your machine after the initial model download.**

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11+ | Server (proxy only) |
| Chrome / Edge 113+ | WebGPU support required for the LLM |
| Internet (first run) | Models are downloaded and cached by the browser |

---

## Quick Start

```bash
# Install minimal Python deps
pip install -r requirements.txt

# Start the proxy server
python -m uvicorn app.main:app --reload

# Open in Chrome or Edge
open http://localhost:8000
```

---

## Using the app

1. **Wait** — the embedding model (`all-MiniLM-L6-v2`, ~23 MB) downloads automatically.
2. **Pick an LLM** from the toolbar dropdown and click **Load**.
   - First load downloads the model weights (800 MB – 2 GB, cached by browser).
3. **Paste a blog URL** and click **Create Twin**.
4. **Chat** — responses stream token by token from the in-browser LLM.

### Available LLM models

| Model | Size | Notes |
|---|---|---|
| Llama 3.2 1B | ~800 MB | Fastest, good for quick answers |
| SmolLM2 1.7B | ~1 GB | Compact, surprisingly capable |
| Gemma 2 2B | ~1.5 GB | Google, strong reasoning |
| Llama 3.2 3B | ~2 GB | Best quality in this size range |
| Phi 3.5 Mini | ~2.2 GB | Microsoft, strong at instruction following |

---

## Architecture

```
chatmyideas/
├── app/
│   ├── main.py          # FastAPI — CORS proxy + static file server only
│   └── static/
│       └── index.html   # Full SPA: ingestion, RAG, chat — all in-browser
├── requirements.txt     # fastapi, uvicorn, httpx only
└── README.md
```

### Browser stack

| Layer | Library | Notes |
|---|---|---|
| Embeddings | [Transformers.js](https://github.com/xenova/transformers.js) v2 | ONNX, runs in main thread |
| LLM | [WebLLM](https://github.com/mlc-ai/web-llm) | WebGPU, quantised models |
| Vector search | Pure JS cosine similarity | In-memory, O(n) |
| Persistence | IndexedDB | Chunks + embeddings survive page refresh |
| Article parsing | [Readability.js](https://github.com/mozilla/readability) | Mozilla's article extractor |

---

## Browser compatibility

| Browser | Embeddings | LLM (WebGPU) |
|---|---|---|
| Chrome 113+ | ✓ | ✓ |
| Edge 113+   | ✓ | ✓ |
| Firefox     | ✓ | ✗ (WebGPU behind flag) |
| Safari      | ✓ | ✗ (partial WebGPU) |

---

## Other branches

| Branch | Description |
|---|---|
| `main` | Server-side RAG — Ollama + ChromaDB/Redis + sentence-transformers |
| `browser-local` | **This branch** — everything runs in the browser |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
