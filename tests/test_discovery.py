import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from alexandria.discovery import (
    DiscoveryEngine,
    ResultSifter,
    SearchResult,
    SearXNGAggregator,
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
        SearchResult(url="https://valid.com/other", title="Other", snippet="Other Snippet", score=0.8)
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
@patch("httpx.AsyncClient.get")
async def test_aggregator_json_decode_error(mock_get, caplog):
    # Setup the mock response to simulate a 502 returning HTML instead of JSON
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    # Force json() to raise JSONDecodeError
    mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "<html>502 Bad Gateway</html>", 0)
    mock_get.return_value = mock_response

    aggregator = SearXNGAggregator(max_concurrent_queries=1)
    
    with caplog.at_level(logging.ERROR):
        results = await aggregator.execute_queries(["test query"])
        
    # Assert that the error is isolated, logged, and returns an empty list gracefully
    assert results == []
    assert "Failed to decode search response payload" in caplog.text
    assert "test query" in caplog.text


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_aggregator_network_errors(mock_get, caplog):
    aggregator = SearXNGAggregator(max_concurrent_queries=2)

    # 1. Mock TimeoutException
    mock_get.side_effect = httpx.TimeoutException("Connection timed out")
    
    with caplog.at_level(logging.ERROR):
        results_timeout = await aggregator.execute_queries(["timeout query"])
    
    assert results_timeout == []
    assert "Network request connection timed out" in caplog.text

    # 2. Mock RequestError
    caplog.clear()
    mock_get.side_effect = httpx.RequestError("Host unreachable", request=MagicMock())
    
    with caplog.at_level(logging.ERROR):
        results_request_err = await aggregator.execute_queries(["request err query"])
        
    assert results_request_err == []
    assert "Transport connectivity mapping error" in caplog.text

    # 3. Mock HTTPStatusError
    caplog.clear()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.side_effect = httpx.HTTPStatusError("500 Server Error", request=MagicMock(), response=mock_response)
    
    with caplog.at_level(logging.ERROR):
        results_status_err = await aggregator.execute_queries(["status err query"])
        
    assert results_status_err == []
    assert "bad status code 500" in caplog.text


@pytest.mark.asyncio
@patch("alexandria.discovery.SearXNGAggregator.execute_queries")
async def test_discovery_engine_orchestration(mock_execute_queries):
    # Setup mock aggregator response with a valid tracking parameter (utm_source)
    mock_execute_queries.return_value = [
        SearchResult(url="https://example.com/topic?utm_source=1", title="A", snippet="A", score=1.0),
        SearchResult(url="https://example.com/topic", title="B", snippet="B", score=2.0),
        SearchResult(url="https://pinterest.com/pin/1", title="P", snippet="P", score=5.0) # Should be sifted out
    ]
    
    engine = DiscoveryEngine()
    collection = SubQueryCollection(
        master_topic="Artificial Intelligence",
        sub_queries=["Machine Learning", "Neural Networks", "Artificial Intelligence"] # Duplicate sub-query
    )
    
    final_urls = await engine.expand_and_discover(collection)
    
    # Assert duplicate queries were merged before execution
    mock_execute_queries.assert_called_once_with(["Artificial Intelligence", "Machine Learning", "Neural Networks"])
    
    # Assert exactly one pristine, deduplicated destination target is exposed
    assert len(final_urls) == 1
    assert final_urls[0] == "https://example.com/topic"
