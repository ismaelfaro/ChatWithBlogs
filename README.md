# ChatMyIdeas — Browser Local

> **Branch:** `browser-local`
> All AI runs inside the browser — no Ollama, no Python vector store, no GPU server needed.

Turn any blog into a conversational digital twin of its author.
Embeddings and LLM inference happen entirely in your browser via **WebAssembly**. The Python server only proxies external URLs to bypass CORS.

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
              ▼  (browser — wllama · WebAssembly · LFM2-350M-GGUF)
           LLM response (streamed token by token)
```

**Nothing leaves your machine after the initial model download.**

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11+ | Server (proxy only) |
| Any modern browser | Chrome, Firefox, Safari, Edge — no WebGPU needed |
| Internet (first run) | Models are downloaded and cached by the browser |

---

## Quick Start

```bash
# Install minimal Python deps
pip install -r requirements.txt

# Start the proxy server
python -m uvicorn app.main:app --reload

# Open in any modern browser
open http://localhost:8000
```

---

## Using the app

1. **Wait** — the embedding model (`all-MiniLM-L6-v2`, ~23 MB) downloads automatically.
2. **Pick a quantization** from the toolbar dropdown and click **Load LFM2**.
   - First load downloads the model weights (~229–379 MB, cached by browser).
3. **Paste a blog URL** and click **Create Twin**.
4. **Chat** — responses stream token by token from the in-browser LLM.

### Available quantizations (LiquidAI/LFM2-350M-GGUF)

| Quantization | Size | Notes |
|---|---|---|
| Q4_K_M | ~229 MB | Default — good balance of size and quality |
| Q5_K_M | ~260 MB | Slightly better quality |
| Q8_0   | ~379 MB | Highest quality, largest download |

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
| LLM | [wllama](https://github.com/ngxson/wllama) | llama.cpp → WebAssembly, GGUF from HuggingFace Hub |
| LLM model | [LiquidAI/LFM2-350M-GGUF](https://huggingface.co/LiquidAI/LFM2-350M-GGUF) | Hybrid architecture, RAG-optimised, greedy decoding |
| Vector search | Pure JS cosine similarity | In-memory, O(n) |
| Persistence | IndexedDB | Chunks + embeddings survive page refresh |
| Article parsing | [Readability.js](https://github.com/mozilla/readability) | Mozilla's article extractor |

---

## Browser compatibility

| Browser | Embeddings | LLM (WASM) |
|---|---|---|
| Chrome 113+ | ✓ | ✓ |
| Edge 113+   | ✓ | ✓ |
| Firefox     | ✓ | ✓ |
| Safari      | ✓ | ✓ |

---

## Other branches

| Branch | Description |
|---|---|
| `main` | Server-side RAG — Ollama + ChromaDB/Redis + sentence-transformers |
| `browser-local` | **This branch** — everything runs in the browser |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
