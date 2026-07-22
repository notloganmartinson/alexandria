import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from alexandria.discovery import (
    AsyncCrawler,
    DDGAggregator,
    DiscoveryEngine,
    ResultSifter,
    SearchResult,
    SubQueryCollection,
)


def test_normalize_url_trailing_slashes_and_domains():
    sifter = ResultSifter()

    # Assert trailing slashes are stripped and domain is lowercased
    url = "HTTPS://WWW.EXAMPLE.COM/path/to/resource/"
    expected = "https://example.com/path/to/resource"
    assert sifter.normalize_url(url) == expected

    # Assert tracking parameters are purged
    url_with_tracking = "http://example.com/page?utm_source=twitter&valid=1&fbclid=abc"
    expected_no_tracking = "http://example.com/page?valid=1"
    assert sifter.normalize_url(url_with_tracking) == expected_no_tracking

    # Assert fragment stripping (urlunparse omits fragment if not included)
    url_with_fragment = "https://example.com/doc#section1"
    expected_no_fragment = "https://example.com/doc"
    assert sifter.normalize_url(url_with_fragment) == expected_no_fragment


def test_normalize_url_parameter_sorting():
    sifter = ResultSifter()

    # Assert identical URLs with varying query parameter layouts sort to the same canonical key
    url_a = "https://example.com/api?b=2&a=1&c=3"
    url_b = "https://example.com/api?c=3&b=2&a=1"

    normalized_a = sifter.normalize_url(url_a)
    normalized_b = sifter.normalize_url(url_b)

    expected = "https://example.com/api?a=1&b=2&c=3"
    assert normalized_a == expected
    assert normalized_b == expected
    assert normalized_a == normalized_b


def test_is_valid_authority_logic():
    sifter = ResultSifter()

    # Valid domains
    assert sifter.is_valid_authority("https://en.wikipedia.org/wiki/Python") is True
    assert sifter.is_valid_authority("https://arxiv.org/abs/1234.5678") is True

    # Blacklisted domains
    assert sifter.is_valid_authority("https://pinterest.com/pin/123/") is False
    assert sifter.is_valid_authority("https://www.quora.com/What-is-AI") is False
    assert sifter.is_valid_authority("http://bit.ly/xyz") is False

    # Regex pattern matching
    assert sifter.is_valid_authority("https://example.com/out?url=target") is False
    assert sifter.is_valid_authority("https://example.com/share?id=1") is False
    assert sifter.is_valid_authority("https://ads.network.com/serve") is False
    assert sifter.is_valid_authority("https://blog.com/affiliate-link-click") is False


def test_sift_and_deduplicate_scores():
    sifter = ResultSifter()

    # Multiple SearchResult objects pointing to the same normalized URL
    raw_results = [
        SearchResult(url="https://example.com/article/?utm_source=1", title="Title A", snippet="Snippet A", score=0.5),
        SearchResult(url="https://example.com/article", title="Title B", snippet="Snippet B", score=0.9),
        SearchResult(url="https://EXAMPLE.com/article/", title="Title C", snippet="Snippet C", score=0.7),
        SearchResult(url="https://valid.com/other", title="Other", snippet="Other Snippet", score=0.8),
    ]

    deduplicated = sifter.sift_and_deduplicate(raw_results)

    # We should have exactly two results
    assert len(deduplicated) == 2

    # They should be sorted by score descending (0.9, then 0.8)
    assert deduplicated[0].score == 0.9
    assert deduplicated[0].url == "https://example.com/article"
    assert deduplicated[0].title == "Title B"

    assert deduplicated[1].score == 0.8
    assert deduplicated[1].url == "https://valid.com/other"


@pytest.mark.asyncio
@patch("alexandria.discovery.DDGS")
async def test_ddg_aggregator_error_handling(mock_ddgs, caplog):
    # Setup mock to raise an exception during DDG search to test fault tolerance
    mock_instance = MagicMock()
    mock_instance.text.side_effect = Exception("DDG blocked the request")

    # Synchronous context manager mocking for 'with DDGS() as ddgs:'
    mock_ddgs.return_value.__enter__.return_value = mock_instance

    aggregator = DDGAggregator()

    with caplog.at_level(logging.ERROR):
        results = await aggregator.execute_queries(["test query"])

    # Assert that errors are caught, logged, and an empty list is gracefully returned
    assert results == []
    assert "DuckDuckGo fetch failed" in caplog.text

@pytest.mark.asyncio
@patch("alexandria.discovery.AsyncWebCrawler")
async def test_async_crawler_error_handling(mock_crawler, caplog):
    # Setup mock to raise an exception during URL extraction (e.g., Cloudflare block)
    mock_instance = MagicMock()
    
    # FIX: Use AsyncMock so the exception is raised during 'await', not during task creation
    mock_instance.arun = AsyncMock(side_effect=Exception("Cloudflare blocked crawler"))

    mock_crawler.return_value.__aenter__.return_value = mock_instance

    crawler = AsyncCrawler()

    with caplog.at_level(logging.ERROR):
        results = await crawler.extract_markdown(["https://example.com"])

    assert results == []
    assert "Crawl4AI failed to extract" in caplog.text

@pytest.mark.asyncio
@patch("alexandria.discovery.AsyncCrawler.extract_markdown")
@patch("alexandria.discovery.DDGAggregator.execute_queries")
async def test_discovery_engine_orchestration(mock_execute_queries, mock_extract):
    # Mock Phase 1: URL Discovery
    mock_execute_queries.return_value = [
        SearchResult(url="https://example.com/topic?utm_source=1", title="A", snippet="A", score=1.0),
        SearchResult(url="https://example.com/topic", title="B", snippet="B", score=2.0),
        SearchResult(url="https://pinterest.com/pin/1", title="P", snippet="P", score=5.0),  # Should be sifted out
    ]

    # Mock Phase 2: Markdown Extraction
    mock_extract.return_value = [
        {"url": "https://example.com/topic", "markdown": "# Topic Data"}
    ]

    engine = DiscoveryEngine()
    collection = SubQueryCollection(
        master_topic="Artificial Intelligence",
        sub_queries=["Machine Learning", "Neural Networks", "Artificial Intelligence"],  # Duplicate sub-query
    )

    # Test Phase 1: Expansion
    final_urls = await engine.expand_and_discover(collection)

    # Assert duplicate queries were merged before execution
    mock_execute_queries.assert_called_once_with(["Artificial Intelligence", "Machine Learning", "Neural Networks"])

    # Assert exactly one pristine, deduplicated destination target is exposed
    assert len(final_urls) == 1
    assert final_urls[0] == "https://example.com/topic"

    # Test Phase 2: Extraction
    payloads = await engine.scrape_payloads(final_urls)
    mock_extract.assert_called_once_with(["https://example.com/topic"])
    assert len(payloads) == 1
    assert payloads[0]["markdown"] == "# Topic Data"
