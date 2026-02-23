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

### Available models

| Model | Size | Backend | Notes |
|---|---|---|---|
| LFM2-350M · Q4_K_M | 229 MB | WASM | Default — fast RAG, greedy decoding |
| LFM2-350M · Q5_K_M | 260 MB | WASM | Slightly better quality |
| LFM2-350M · Q8_0   | 379 MB | WASM | Highest quality LFM2-350M |
| LFM2-1.2B · Q4_K_M | 731 MB | WASM | Larger LFM2 base |
| LFM2-1.2B-RAG · Q4_K_M | 731 MB | WASM | RAG-tuned LFM2 1.2B |
| Gemma 3 270M · Q4_K_M | 253 MB | WASM | Tiny Gemma 3, surprisingly capable |
| Gemma 2 2B · Q4_K_M | 1.71 GB | WASM + WebGPU | Google, solid reasoning |
| Granite 4.0 1B (MXFP4) | 1.55 GB | WASM | IBM Granite hybrid quant |
| Qwen3-0.6B · Q4_K_M | 397 MB | WASM + WebGPU | Alibaba Qwen3, non-thinking mode |
| Qwen3-1.7B · Q4_K_M | 1.11 GB | WASM + WebGPU | Larger Qwen3, good quality |

Models marked **WASM + WebGPU** can use either backend. On Chrome/Edge with WebGPU, select "Auto" or "WebGPU" in the toolbar for faster inference.

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
| LLM (WASM) | [wllama](https://github.com/ngxson/wllama) | llama.cpp → WebAssembly, GGUF from HuggingFace Hub |
| LLM (WebGPU) | [WebLLM](https://github.com/mlc-ai/web-llm) | MLC quantised models, GPU accelerated |
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
