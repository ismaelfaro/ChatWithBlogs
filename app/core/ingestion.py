"""
Blog ingestion: RSS/Atom feed first, HTML scraping as fallback.

Strategy
--------
1. If the URL itself looks like a feed, parse it directly.
2. Otherwise, fetch the page and look for <link rel="alternate"> feed URLs.
3. If a feed is found, parse every entry and (optionally) fetch the full
   article body via trafilatura for richer content.
4. If no feed is found, discover article links on the page and scrape each
   one individually with trafilatura.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import feedparser
import requests
import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update(
    {"User-Agent": "ChatMyIdeas/1.0 (+https://github.com/chatmyideas/chatmyideas)"}
)
_FETCH_TIMEOUT = 20
_MAX_ARTICLES = 100  # safety cap


@dataclass
class Article:
    url: str
    title: str
    content: str
    author: Optional[str] = None
    published: Optional[str] = None

    @property
    def id(self) -> str:
        return hashlib.md5(self.url.encode()).hexdigest()


# ── Public entry point ────────────────────────────────────────────────────────

def ingest_blog(url: str) -> list[Article]:
    """Return a list of Article objects extracted from *url*."""
    url = url.strip().rstrip("/")

    articles = _try_as_feed(url)
    if articles:
        logger.info("Ingested %d articles via feed from %s", len(articles), url)
        return articles

    feed_url = _discover_feed(url)
    if feed_url:
        articles = _try_as_feed(feed_url)
        if articles:
            logger.info(
                "Ingested %d articles via discovered feed %s", len(articles), feed_url
            )
            return articles

    articles = _scrape_html(url)
    if articles:
        logger.info("Ingested %d articles via HTML scraping from %s", len(articles), url)
        return articles

    # Last resort: treat the URL itself as a single article
    article = _scrape_single(url)
    if article:
        logger.info("Ingested URL as a single article: %s", url)
        return [article]

    logger.warning("No articles found at %s", url)
    return []


def detect_blog_title(url: str) -> str:
    """Best-effort extraction of the blog / site title."""
    try:
        resp = _SESSION.get(url, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception:
        pass
    return urlparse(url).netloc


# ── Feed handling ─────────────────────────────────────────────────────────────

def _try_as_feed(url: str) -> list[Article]:
    """Try to parse *url* as an RSS/Atom feed. Returns [] on failure."""
    feed = feedparser.parse(url)
    if not feed.entries:
        return []
    articles = []
    for entry in feed.entries[:_MAX_ARTICLES]:
        article = _entry_to_article(entry)
        if article and article.content:
            articles.append(article)
    return articles


def _entry_to_article(entry) -> Optional[Article]:
    link = getattr(entry, "link", None)
    title = getattr(entry, "title", "Untitled")
    if not link:
        return None

    # Try to get full content from the feed entry first
    content = ""
    if hasattr(entry, "content") and entry.content:
        content = entry.content[0].get("value", "")
    elif hasattr(entry, "summary"):
        content = entry.summary or ""

    # Strip HTML
    content = BeautifulSoup(content, "html.parser").get_text(separator="\n", strip=True)

    # If the feed only has a summary, fetch the full article
    if len(content) < 200:
        full = _fetch_article(link)
        if full:
            content = full

    if not content:
        return None

    author = _extract_author(entry)
    published = getattr(entry, "published", None) or getattr(entry, "updated", None)

    return Article(url=link, title=title, content=content, author=author, published=published)


def _extract_author(entry) -> Optional[str]:
    if hasattr(entry, "author") and entry.author:
        return str(entry.author).strip()
    if hasattr(entry, "authors") and entry.authors:
        name = entry.authors[0].get("name", "")
        if name:
            return name.strip()
    return None


# ── Feed discovery ────────────────────────────────────────────────────────────

_FEED_SUFFIXES = ["/feed", "/rss", "/rss.xml", "/atom.xml", "/feed.xml", "/index.xml"]


def _discover_feed(url: str) -> Optional[str]:
    """Look for a feed URL in the HTML <head> of *url*."""
    try:
        resp = _SESSION.get(url, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        for link in soup.find_all("link", rel=True):
            rel = " ".join(link.get("rel", []))
            if "alternate" in rel:
                t = link.get("type", "")
                if "rss" in t or "atom" in t:
                    href = link.get("href", "")
                    if href:
                        return urljoin(url, href)
    except Exception:
        pass

    # Try common suffixes
    for suffix in _FEED_SUFFIXES:
        candidate = url + suffix
        try:
            resp = _SESSION.head(candidate, timeout=5, allow_redirects=True)
            if resp.status_code < 400:
                return candidate
        except Exception:
            continue
    return None


# ── HTML scraping fallback ────────────────────────────────────────────────────

_ARTICLE_PATTERNS = re.compile(
    r"(/(?:post|blog|article|entry|p|story|news|writing|essay|note)/[^\"'>\s]+)",
    re.IGNORECASE,
)


def _scrape_html(base_url: str) -> list[Article]:
    """Discover article links from the homepage and scrape each one."""
    try:
        resp = _SESSION.get(base_url, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Could not fetch %s: %s", base_url, exc)
        return []

    soup = BeautifulSoup(resp.content, "html.parser")
    base_domain = "{0.scheme}://{0.netloc}".format(urlparse(base_url))

    seen: set[str] = set()
    links: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].split("#")[0].split("?")[0]
        if not href:
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        # Stay on same domain and match article-like paths
        if parsed.netloc != urlparse(base_url).netloc:
            continue
        if full in seen:
            continue
        if _ARTICLE_PATTERNS.search(parsed.path):
            seen.add(full)
            links.append(full)

    articles: list[Article] = []
    for link in links[:_MAX_ARTICLES]:
        try:
            content = _fetch_article(link)
            if not content:
                continue
            title = _fetch_title(link)
            articles.append(
                Article(url=link, title=title, content=content, author=None)
            )
        except Exception as exc:
            logger.debug("Skipping %s: %s", link, exc)

    return articles


def _scrape_single(url: str) -> Optional[Article]:
    """Treat *url* as a single self-contained article and scrape it directly."""
    try:
        resp = _SESSION.get(url, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        html = resp.text

        content = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        if not content or len(content.strip()) < 50:
            return None

        # Title: og:title > h1 > <title>
        soup = BeautifulSoup(html, "html.parser")
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()
        else:
            title = url

        # Author: og:author > meta[name=author]
        author: Optional[str] = None
        og_author = soup.find("meta", property="og:author") or soup.find("meta", attrs={"name": "author"})
        if og_author and og_author.get("content"):
            author = og_author["content"].strip()

        return Article(url=url, title=title, content=content.strip(), author=author)
    except Exception as exc:
        logger.debug("Single-article scrape failed on %s: %s", url, exc)
        return None


def _fetch_article(url: str) -> Optional[str]:
    """Use trafilatura to extract clean article text from *url*."""
    try:
        resp = _SESSION.get(url, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
        return text.strip() if text else None
    except Exception as exc:
        logger.debug("trafilatura failed on %s: %s", url, exc)
        return None


def _fetch_title(url: str) -> str:
    try:
        resp = _SESSION.get(url, timeout=_FETCH_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        # Prefer og:title or <title>
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"].strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        if soup.title and soup.title.string:
            return soup.title.string.strip()
    except Exception:
        pass
    return url
