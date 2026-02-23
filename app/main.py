"""
ChatWithBlogs — browser-local backend.

The server does exactly two things:
  1. Serve the single-page application (app/static/index.html).
  2. Proxy external HTTP requests so the browser can fetch arbitrary
     URLs without hitting CORS restrictions.

All AI work — embeddings and LLM inference (Transformers.js / ONNX
Runtime Web) — runs entirely inside the user's browser.  No Ollama, no
ChromaDB, no sentence-transformers required on the server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

app = FastAPI(
    title="ChatWithBlogs — browser-local proxy",
    description="CORS proxy + static file server. All AI runs in the browser.",
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class _COOPCOEPMiddleware(BaseHTTPMiddleware):
    """Add COOP/COEP headers so SharedArrayBuffer is available in the browser.

    SharedArrayBuffer is required for wllama's multi-threaded WASM backend.
    COEP uses 'credentialless' instead of 'require-corp' so cross-origin
    resources fetched by the proxy don't need to opt-in with CORP headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        return response


app.add_middleware(_COOPCOEPMiddleware)

_ALLOWED_SCHEMES = {"http", "https"}
_REQUEST_HEADERS = {
    "User-Agent": "ChatWithBlogs/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class ProxyRequest(BaseModel):
    url: str


@app.post("/api/proxy")
async def proxy(body: ProxyRequest) -> dict:
    """Fetch *url* server-side and return its text content."""
    parsed = urlparse(body.url)
    if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.netloc:
        raise HTTPException(status_code=400, detail="Only http/https URLs are allowed.")

    async with httpx.AsyncClient(
        timeout=30,
        follow_redirects=True,
        headers=_REQUEST_HEADERS,
    ) as client:
        try:
            r = await client.get(body.url)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to fetch URL: {exc}")

    return {
        "url": str(r.url),
        "status": r.status_code,
        "content_type": r.headers.get("content-type", ""),
        "text": r.text,
    }


# ── Static / SPA ──────────────────────────────────────────────────────────────

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_static / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
