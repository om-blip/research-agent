"""
Unit tests for the RAG pipeline.

Key rule: unit tests should NEVER make real API calls.
We test the logic and plumbing, not the LLM responses.
That's what evals are for.

Why bother testing the chunker and parser?
Because the most common RAG bugs aren't in the LLM —
they're in the plumbing:
- Chunker produces 0 chunks for valid text
- Parser returns empty string for valid HTML
- Vector store creates duplicate IDs

These tests catch those issues in milliseconds, for free.
"""

import pytest
from langchain_core.documents import Document
from rag.chunker import chunk_text, chunk_multiple_sources
from mcp.tools.fetch_tool import _parse_html
from agents.state import ResearchState


class TestChunker:

    def test_basic_text_produces_chunks(self):
        """Valid text should produce at least one chunk."""
        text = "This is about AI research. " * 50
        chunks = chunk_text(text, "https://example.com")
        assert len(chunks) > 0

    def test_every_chunk_is_document(self):
        """All chunks must be LangChain Document objects."""
        chunks = chunk_text("Some content. " * 100, "https://test.com")
        assert all(isinstance(c, Document) for c in chunks)

    def test_source_url_in_metadata(self):
        """Every chunk must carry the source URL."""
        url = "https://mysite.com/article"
        chunks = chunk_text("Content here. " * 100, url)
        for chunk in chunks:
            assert chunk.metadata["source"] == url

    def test_chunk_index_in_metadata(self):
        """Chunks must have chunk_index and chunk_total."""
        chunks = chunk_text("Word. " * 200, "https://test.com")
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_total" in chunk.metadata

    def test_empty_text_returns_empty_list(self):
        """Empty or whitespace text should return [] not crash."""
        assert chunk_text("", "https://test.com") == []
        assert chunk_text("   ", "https://test.com") == []
        assert chunk_text("\n\n\n", "https://test.com") == []

    def test_chunk_size_not_exceeded_by_much(self):
        """No chunk should be massively over the configured size."""
        from config import config
        chunks = chunk_text("word " * 1000, "https://test.com")
        for chunk in chunks:
            # Allow 50% overflow because splitter won't break mid-word
            assert len(chunk.page_content) <= config.CHUNK_SIZE * 1.5

    def test_multiple_sources_combined(self):
        """chunk_multiple_sources should merge chunks from all sources."""
        sources = [
            {"text": "Source one content. " * 30, "url": "https://s1.com"},
            {"text": "Source two content. " * 30, "url": "https://s2.com"},
        ]
        chunks = chunk_multiple_sources(sources)
        urls = {c.metadata["source"] for c in chunks}
        assert "https://s1.com" in urls
        assert "https://s2.com" in urls

    def test_empty_source_skipped(self):
        """Sources with empty text should be skipped gracefully."""
        sources = [
            {"text": "", "url": "https://empty.com"},
            {"text": "Real content here. " * 30, "url": "https://real.com"},
        ]
        chunks = chunk_multiple_sources(sources)
        urls = {c.metadata["source"] for c in chunks}
        assert "https://real.com" in urls
        assert "https://empty.com" not in urls


class TestHTMLParser:

    def test_strips_script_tags(self):
        """JavaScript should not appear in extracted text."""
        html = "<html><body><script>var x=1;</script><p>Real content</p></body></html>"
        result = _parse_html(html, "https://test.com")
        assert "var x=1" not in result["text"]
        assert "Real content" in result["text"]

    def test_strips_nav_and_footer(self):
        """Navigation and footer noise should be removed."""
        html = "<html><body><nav>Menu links</nav><p>Article text</p><footer>Copyright</footer></body></html>"
        result = _parse_html(html, "https://test.com")
        assert "Menu links" not in result["text"]
        assert "Copyright" not in result["text"]
        assert "Article text" in result["text"]

    def test_extracts_title(self):
        """Page title should be extracted correctly."""
        html = "<html><head><title>My Page</title></head><body>Content</body></html>"
        result = _parse_html(html, "https://test.com")
        assert result["title"] == "My Page"
        assert result["success"] is True

    def test_collapses_whitespace(self):
        """Multiple spaces and newlines should become single spaces."""
        html = "<html><body><p>Word1    Word2\n\nWord3</p></body></html>"
        result = _parse_html(html, "https://test.com")
        assert "  " not in result["text"]

    def test_truncates_long_content(self):
        """Content over 50k chars should be truncated."""
        long_text = "word " * 20000
        html = f"<html><body><p>{long_text}</p></body></html>"
        result = _parse_html(html, "https://test.com")
        assert len(result["text"]) <= 50_000

    def test_success_flag_set(self):
        """Valid HTML should return success=True."""
        html = "<html><body><p>Hello</p></body></html>"
        result = _parse_html(html, "https://test.com")
        assert result["success"] is True
        assert result["error"] == ""


class TestResearchState:

    def test_all_required_fields_exist(self):
        """ResearchState must have all expected fields."""
        fields = ResearchState.__annotations__.keys()
        required = [
            "topic", "recipient_email", "sub_questions", "run_id",
            "raw_sources", "errors", "chunks_embedded",
            "report_markdown", "email_sent", "email_error"
        ]
        for field in required:
            assert field in fields, f"Missing field: {field}"

    def test_raw_sources_uses_operator_add(self):
        """
        raw_sources must use Annotated with operator.add.
        This is what makes parallel agent fan-out work.
        If this breaks, parallel agents overwrite each other's results.
        """
        import operator
        from typing import get_args, get_origin, Annotated
        annotation = ResearchState.__annotations__["raw_sources"]
        args = get_args(annotation)
        assert len(args) == 2
        assert args[1] is operator.add