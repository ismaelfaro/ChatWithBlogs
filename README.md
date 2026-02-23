# ChatWithBlogs — Browser Local

> All AI runs inside the browser — no Ollama, no Python vector store, no GPU server needed.

Turn any blog into a conversational digital twin of its author.
Embeddings and LLM inference happen entirely in your browser via **Transformers.js v3 / ONNX Runtime Web**. The Python server only proxies external URLs to bypass CORS (not needed when using the GitHub Pages hosted version).

---

## How it works

```
Blog URL
   │
   ▼  (CORS proxy — local FastAPI server or allorigins.win on GitHub Pages)
 Fetch HTML / RSS
   │
   ▼  (browser — Readability.js)
 Extract article text
   │
   ▼  (browser — Transformers.js v3 · all-MiniLM-L6-v2 · ONNX)
 Chunk + embed  →  IndexedDB (persisted locally)
   │
 Chat query
   ├─ Embed query  ──► cosine similarity  ──► top-K chunks
   └─ Build prompt ("You are a digital twin of …")
              │
              ▼  (browser — Transformers.js v3 WASM  or  WebLLM WebGPU)
           LLM response (streamed token by token, thinking blocks collapsed)
```

**Nothing leaves your machine after the initial model download.**

---

## Usage options

### Option A — GitHub Pages (no install required)

Open the hosted version directly in your browser. The CORS proxy is handled automatically via [allorigins.win](https://allorigins.win).

> Enable GitHub Pages in your repo: **Settings → Pages → Source → GitHub Actions**

### Option B — Run locally

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
2. **Pick a model** from the toolbar dropdown and click **Load**.
   - First load downloads the ONNX model weights (cached by the browser).
   - WebGPU models are faster on Chrome/Edge; WASM works on all browsers.
3. **Paste a blog URL** and click **Add Blog**.
4. **Chat** — responses stream token by token. Thinking blocks (`<think>…</think>`) are shown live then collapsed into a summary.

---

## Load a blog via URL parameter

Append `?url=<blog-url>` to the app address to fully automate the flow — no clicks needed:

```
https://<your-github-pages-url>/?url=https://example.com/blog
```

```
http://localhost:8000/?url=https://example.com/blog
```

What happens automatically:
1. Waits for the embedding model to finish loading
2. Loads **Qwen3-0.6B** (WASM or WebGPU depending on your browser)
3. Fetches and ingests the blog
4. Sends an initial summary query

This is useful for sharing a direct link that drops someone straight into a conversation about a specific blog.

---

## Available models

| Model | Size | Backend | Notes |
|---|---|---|---|
| Qwen3-0.6B | ~350 MB | WASM + WebGPU | Default · Alibaba Qwen3, thinking mode |
| Qwen3-1.7B | ~950 MB | WASM + WebGPU | Larger Qwen3, better quality |
| Gemma 2 2B | 1.71 GB | WebGPU only | Google, solid reasoning |
| LFM2.5-1.2B Instruct | ~700 MB | WASM | Liquid AI, fast RAG |
| LFM2.5-1.2B Thinking | ~700 MB | WASM | Liquid AI, reasoning variant |
| Gemma 3 270M | ~260 MB | WASM | Tiny, surprisingly capable |
| Granite 4.0 350M | ~350 MB | WASM | IBM Granite 4.0 compact |
| Granite 4.0 1B | ~700 MB | WASM | IBM Granite 4.0 base |

Models with **WASM + WebGPU** use WebGPU automatically when available (Chrome/Edge), falling back to WASM otherwise. Switch manually with the backend selector in the toolbar.

---

## Architecture

```
chatwithblogs/
├── .github/
│   └── workflows/
│       └── pages.yml    # Auto-deploy app/static/ to GitHub Pages on push to main
├── app/
│   ├── main.py          # FastAPI — CORS proxy + static file server (local use only)
│   └── static/
│       ├── index.html   # Full SPA: ingestion, RAG, chat — all in-browser
│       └── .nojekyll    # Disables Jekyll on GitHub Pages
├── requirements.txt     # fastapi, uvicorn, httpx only
└── README.md
```

### Browser stack

| Layer | Library | Notes |
|---|---|---|
| Embeddings | [Transformers.js](https://huggingface.co/docs/transformers.js) v3 | ONNX, `Xenova/all-MiniLM-L6-v2` |
| LLM — WASM | [Transformers.js](https://huggingface.co/docs/transformers.js) v3 | ONNX Runtime Web, all browsers |
| LLM — WebGPU | [WebLLM](https://github.com/mlc-ai/web-llm) | MLC quantised models, GPU accelerated |
| Vector search | Pure JS cosine similarity | In-memory, O(n) |
| Persistence | IndexedDB | Chunks + embeddings survive page refresh |
| Article parsing | [Readability.js](https://github.com/mozilla/readability) | Mozilla's article extractor |
| CORS proxy (local) | FastAPI + httpx | Used when running the Python server |
| CORS proxy (hosted) | [allorigins.win](https://allorigins.win) | Used on GitHub Pages and static hosts |

---

## Browser compatibility

| Browser | Embeddings | LLM (WASM) | LLM (WebGPU) |
|---|---|---|---|
| Chrome 113+ | ✓ | ✓ | ✓ |
| Edge 113+   | ✓ | ✓ | ✓ |
| Firefox     | ✓ | ✓ | — |
| Safari      | ✓ | ✓ | — |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
