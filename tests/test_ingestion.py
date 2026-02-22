"""
Tests for blog ingestion (app/core/ingestion.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.ingestion import (
    Article,
    detect_blog_title,
    ingest_blog,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Blog</title>
    <link>https://blog.example.com</link>
    <item>
      <title>Post One</title>
      <link>https://blog.example.com/post/1</link>
      <author>Jane Doe</author>
      <description>Hello world from post one. This is a longer piece of content that
      gives enough text to be useful in a real RAG scenario. We need at least a couple
      of hundred characters to pass the short-content guard. Let us add more words here.
      </description>
    </item>
    <item>
      <title>Post Two</title>
      <link>https://blog.example.com/post/2</link>
      <author>Jane Doe</author>
      <description>Second post content, also quite detailed and informative.
      This one talks about technology and the future of software engineering.
      There are many interesting topics to cover in this space.
      </description>
    </item>
  </channel>
</rss>"""

HTML_PAGE = """
<html>
  <head>
    <title>My Blog</title>
    <link rel="alternate" type="application/rss+xml" href="/feed.xml" />
  </head>
  <body>
    <h1>Welcome to my blog</h1>
    <a href="/blog/post-one">Post One</a>
    <a href="/blog/post-two">Post Two</a>
  </body>
</html>"""


# ── Article dataclass ─────────────────────────────────────────────────────────

class TestArticle:
    def test_id_is_stable(self):
        a = Article(url="https://example.com", title="T", content="C")
        assert a.id == Article(url="https://example.com", title="T", content="C").id

    def test_id_differs_by_url(self):
        a = Article(url="https://example.com/1", title="T", content="C")
        b = Article(url="https://example.com/2", title="T", content="C")
        assert a.id != b.id


# ── RSS ingestion ─────────────────────────────────────────────────────────────

class TestFeedIngestion:
    def test_parses_rss_feed(self):
        """_try_as_feed should parse RSS entries and return Article objects."""
        import feedparser
        from app.core.ingestion import _try_as_feed

        feed = feedparser.parse(RSS_FEED)
        # Provide a long enough article body so no HTTP fetch is needed
        long_content = "This is article content. " * 20  # > 200 chars

        with patch("app.core.ingestion.feedparser.parse", return_value=feed):
            with patch("app.core.ingestion._fetch_article", return_value=long_content):
                articles = _try_as_feed("https://blog.example.com/feed")

        assert len(articles) >= 1
        assert all(isinstance(a, Article) for a in articles)

    @patch("app.core.ingestion.feedparser.parse")
    def test_returns_empty_for_invalid_feed(self, mock_parse):
        mock_parse.return_value = MagicMock(entries=[])
        # Should fall through to HTML scraping
        with patch("app.core.ingestion._scrape_html", return_value=[]):
            with patch("app.core.ingestion._discover_feed", return_value=None):
                articles = ingest_blog("https://not-a-feed.example.com")
        assert articles == []


# ── Blog title detection ──────────────────────────────────────────────────────

class TestDetectBlogTitle:
    @patch("app.core.ingestion._SESSION")
    def test_returns_title_from_html(self, mock_session):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<html><head><title>My Cool Blog</title></head><body></body></html>"
        mock_session.get.return_value = mock_resp

        title = detect_blog_title("https://example.com")
        assert title == "My Cool Blog"

    @patch("app.core.ingestion._SESSION")
    def test_falls_back_to_domain_on_error(self, mock_session):
        mock_session.get.side_effect = Exception("network error")
        title = detect_blog_title("https://example.com/blog")
        assert "example.com" in title


# ── Author extraction ─────────────────────────────────────────────────────────

class TestAuthorExtraction:
    def test_author_from_feed_entry(self):
        from app.core.ingestion import _extract_author

        entry = MagicMock()
        entry.author = "Alice Smith"
        assert _extract_author(entry) == "Alice Smith"

    def test_author_from_authors_list(self):
        from app.core.ingestion import _extract_author

        entry = MagicMock(spec=[])  # no .author attribute
        entry.authors = [{"name": "Bob Jones"}]
        assert _extract_author(entry) == "Bob Jones"

    def test_no_author_returns_none(self):
        from app.core.ingestion import _extract_author

        entry = MagicMock(spec=[])
        del entry.author
        del entry.authors
        assert _extract_author(entry) is None
