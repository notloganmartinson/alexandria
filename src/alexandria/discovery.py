# src/alexandria/discovery.py
import asyncio
import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional, Dict

# SOTA Zero-Config Web Scraping & Discovery
from duckduckgo_search import DDGS
from crawl4ai import AsyncWebCrawler

# Conforms to library logging standard (delegates handlers to the top-level application)[cite: 2]
logger = logging.getLogger("alexandria.discovery")

@dataclass(frozen=True)
class SearchResult:
    """Immutable data structural layout representing a verified discovery hit."""
    url: str
    title: str
    snippet: str
    score: float = 0.0

@dataclass(frozen=True)
class SubQueryCollection:
    """Encapsulates a parent topic partitioned into a localized sub-query array."""
    master_topic: str
    sub_queries: List[str]

class ResultSifter:
    """
    Sensory De-duplication & Authority Sifting Layer.
    Responsible for normalizing tracking strings, path canonicalization, query parameter 
    sorting, and executing domain pattern blocklists.
    """
    def __init__(self) -> None:
        self.tracking_params: set[str] = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
            "fbclid", "gclid", "msclkid", "igshid", "mc_eid", "_bta_tid", "_bta_c"
        }
        self.domain_blacklist: set[str] = {
            "pinterest.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
            "tiktok.com", "bit.ly", "t.co", "goo.gl", "tinyurl.com", "ow.ly",
            "ehow.com", "ezinearticles.com", "hubpages.com", "quora.com", "yahoo.answers.com"
        }
        self.low_signal_patterns: List[re.Pattern[str]] = [
            re.compile(r"/share\?", re.IGNORECASE),
            re.compile(r"/click\?", re.IGNORECASE),
            re.compile(r"/redirect\?", re.IGNORECASE),
            re.compile(r"/out\?", re.IGNORECASE),
            re.compile(r"ad[s]?\.", re.IGNORECASE),
            re.compile(r"affiliate", re.IGNORECASE)
        ]

    def normalize_url(self, raw_url: str) -> Optional[str]:
        """
        Transforms messy web links into a strict canonical form. Lowercases domains,
        removes trailing slash permutations, strips fragments, filters out tracking data,
        and alphabetically sorts query parameters to guarantee absolute deduplication.
        """
        try:
            parsed = urllib.parse.urlparse(raw_url)
            if not parsed.scheme or not parsed.netloc:
                return None
            scheme = parsed.scheme.lower()
            if scheme not in ("http", "https"):
                return None
            netloc = parsed.netloc.lower()
            if netloc.startswith("www."):
                netloc = netloc[4:]
            
            # Path Canonicalization: Strip trailing slashes to catch directory structure duplicates[cite: 2]
            path = parsed.path
            if path and path != "/":
                path = path.rstrip("/")
            
            # Extract and sanitize query strings[cite: 2]
            query_dict = urllib.parse.parse_qs(parsed.query)
            filtered_query = {
                k: v for k, v in query_dict.items()
                if k.lower() not in self.tracking_params
            }
            
            # Deterministic Query Parameter Sorting to avoid permutation bypasses[cite: 2]
            sorted_query = sorted(filtered_query.items())
            new_query = urllib.parse.urlencode(sorted_query, doseq=True)
            return urllib.parse.urlunparse((scheme, netloc, path, parsed.params, new_query, ""))
        except ValueError as e:
            logger.warning(f"Value parsing collision encountered for string {raw_url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error running URL normalization processing on {raw_url}: {str(e)}")
            return None

    def is_valid_authority(self, url: str) -> bool:
        """
        Executes strict heuristic validation sweeps against domain authority blacklists
        and algorithmic content diversion regex patterns.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            netloc = parsed.netloc.lower()
            if any(blacklisted in netloc for blacklisted in self.domain_blacklist):
                return False
            for pattern in self.low_signal_patterns:
                if pattern.search(url):
                    return False
            return True
        except Exception as e:
            logger.error(f"Heuristic validation loop crashed on target URL {url}: {str(e)}")
            return False

    def sift_and_deduplicate(self, raw_results: List[SearchResult]) -> List[SearchResult]:
        """
        Processes a raw list of incoming search impressions, maps them to unique canonical 
        topologies, retains the highest-scoring variation on duplicate matches, and 
        delivers an optimized array sorted by search ranking.
        """
        unique_topology: dict[str, SearchResult] = {}
        for result in raw_results:
            normalized_url = self.normalize_url(result.url)
            
            if not normalized_url:
                continue
                
            if not self.is_valid_authority(normalized_url):
                continue

            if normalized_url in unique_topology:
                existing_result = unique_topology[normalized_url]
                if result.score > existing_result.score:
                    unique_topology[normalized_url] = SearchResult(
                        url=normalized_url,
                        title=result.title,
                        snippet=result.snippet,
                        score=result.score
                    )
            else:
                unique_topology[normalized_url] = SearchResult(
                    url=normalized_url,
                    title=result.title,
                    snippet=result.snippet,
                    score=result.score
                )

        return sorted(unique_topology.values(), key=lambda r: r.score, reverse=True)

class DDGAggregator:
    """
    Zero-config Async DuckDuckGo Aggregator.
    Replaces the brittle SearXNG proxy layer with direct, serverless DDG routing.
    """
    def __init__(self, max_results_per_query: int = 8):
        self.max_results_per_query = max_results_per_query

    def _fetch_query_sync(self, query: str) -> List[SearchResult]:
        """Synchronous fetch wrapped in a thread to prevent event loop blocking."""
        results = []
        try:
            with DDGS() as ddgs:
                # The latest DDGS.text returns a list of dictionaries directly
                raw_results = ddgs.text(query, max_results=self.max_results_per_query)
                
                # Check if DDG returned None (rate-limit/empty results)
                if not raw_results:
                    return results
                    
                for r in raw_results:
                    score = 1.0 - (len(results) * 0.05)
                    results.append(SearchResult(
                        url=r.get("href", ""),
                        title=r.get("title", ""),
                        snippet=r.get("body", ""),
                        score=max(score, 0.1)
                    ))
        except Exception as e:
            logger.error(f"DuckDuckGo fetch failed for query '{query}': {str(e)}")
        return results

    async def _fetch_query(self, query: str) -> List[SearchResult]:
        """Offloads the blocking network call to a background thread."""
        return await asyncio.to_thread(self._fetch_query_sync, query)

    async def execute_queries(self, queries: List[str]) -> List[SearchResult]:
        """Maps queries into parallel Async requests."""
        tasks = [self._fetch_query(q) for q in queries]
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)
        
        flattened_results: List[SearchResult] = []
        for res in results_nested:
            if isinstance(res, list):
                flattened_results.extend(res)
            elif isinstance(res, Exception):
                logger.critical(f"Critical execution escape in DDG aggregation: {str(res)}")
        return flattened_results

class AsyncCrawler:
    """
    SOTA Markdown Extraction using Crawl4AI.
    Bypasses Cloudflare, natively renders React/JS, and outputs mathematically clean Markdown.
    """
    async def extract_markdown(self, urls: List[str]) -> List[Dict[str, str]]:
        extracted_data = []
        async with AsyncWebCrawler() as crawler:
            tasks = [crawler.arun(url=url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for url, result in zip(urls, results):
                if isinstance(result, Exception):
                    logger.error(f"Crawl4AI failed to extract {url}: {str(result)}")
                    continue
                
                # Check for successful extraction and pure markdown presence
                if result and hasattr(result, 'markdown') and result.markdown:
                    extracted_data.append({
                        "url": url,
                        "markdown": result.markdown
                    })
        return extracted_data


class DiscoveryEngine:
    """
    Sub-Query Dimensional Expansion Engine.
    Orchestrates DDG discovery, URL deduplication, and Crawl4AI Markdown extraction.
    """
    def __init__(self, max_results_per_query: int = 8) -> None:
        self.aggregator: DDGAggregator = DDGAggregator(max_results_per_query=max_results_per_query)
        self.sifter: ResultSifter = ResultSifter()
        self.crawler: AsyncCrawler = AsyncCrawler()

    async def expand_and_discover(self, collection: SubQueryCollection) -> List[str]:
        """
        Phase 1: Discovery. 
        Executes expanded queries and returns a deduplicated list of target URLs.
        """
        logger.info(f"Initiating dimensional query expansion for target topic: '{collection.master_topic}'")
        
        # Merge master query and expansion queries into a single evaluation set[cite: 2]
        all_queries = [collection.master_topic] + collection.sub_queries
        unique_queries = list(dict.fromkeys(all_queries))
        
        raw_results = await self.aggregator.execute_queries(unique_queries)
        logger.info(f"Discovery aggregation complete. Captured {len(raw_results)} total raw impression hits.")
        
        cleaned_results = self.sifter.sift_and_deduplicate(raw_results)
        logger.info(f"Sifting sequence finalized. Retained {len(cleaned_results)} unique, high-authority links.")
        
        return [result.url for result in cleaned_results]

    async def scrape_payloads(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Phase 2: Extraction.
        Consumes URLs and returns structured dictionaries of source URLs and clean Markdown.
        """
        logger.info(f"Executing Crawl4AI sequence against {len(urls)} validated targets.")
        payloads = await self.crawler.extract_markdown(urls)
        logger.info(f"Extraction sequence complete. Successfully rendered {len(payloads)} Markdown payloads.")
        return payloads
