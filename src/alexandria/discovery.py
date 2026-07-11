# src/alexandria/discovery.py
import asyncio
import json
import logging
import re
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional

import httpx

# Conforms to library logging standard (delegates handlers to the top-level application)
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

            # Path Canonicalization: Strip trailing slashes to catch directory structure duplicates
            path = parsed.path
            if path and path != "/":
                path = path.rstrip("/")

            # Extract and sanitize query strings
            query_dict = urllib.parse.parse_qs(parsed.query)
            filtered_query = {
                k: v for k, v in query_dict.items()
                if k.lower() not in self.tracking_params
            }
            
            # Deterministic Query Parameter Sorting to avoid permutation bypasses
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


class SearXNGAggregator:
    """
    Resilient Concurrent SearXNG Proxy Aggregator.
    Handles highly parallelized network connections bounded by async semaphores and 
    strict request parameters to protect host system limits.
    """

    def __init__(
        self, 
        endpoint: str = "http://127.0.0.1:8081/search", 
        timeout_seconds: float = 12.0,
        max_concurrent_queries: int = 8
    ) -> None:
        self.endpoint: str = endpoint
        self.timeout_seconds: float = timeout_seconds
        # Enforces a strict operational boundary to prevent overloading local proxy sockets
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_queries)

    async def _fetch_query(self, client: httpx.AsyncClient, query: str) -> List[SearchResult]:
        """
        Executes a localized search network transaction wrapped inside tight exception safety nets.
        Isolates JSON serialization faults to explicitly identify proxy service failures.
        """
        params: dict[str, str] = {
            "q": query,
            "format": "json"
        }
        
        async with self._semaphore:
            try:
                response = await client.get(self.endpoint, params=params)
                response.raise_for_status()
                
                # Isolated decoding boundary to catch HTML proxy faults (e.g., 502/504 text maps)
                try:
                    data = response.json()
                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"Failed to decode search response payload for query '{query}'. "
                        f"The SearXNG proxy service likely experienced an upstream deadlock and returned HTML text. "
                        f"Details: {str(json_err)}"
                    )
                    return []

                results: List[SearchResult] = []
                for item in data.get("results", []):
                    url = item.get("url")
                    title = item.get("title", "")
                    snippet = item.get("content", "")
                    
                    try:
                        score = float(item.get("score", 0.0))
                    except (TypeError, ValueError):
                        score = 0.0

                    if url:
                        results.append(SearchResult(url=url, title=title, snippet=snippet, score=score))
                        
                return results

            except httpx.TimeoutException:
                logger.error(f"Network request connection timed out against SearXNG for query string: '{query}'")
                return []
            except httpx.HTTPStatusError as e:
                logger.error(f"Local SearXNG instance returned bad status code {e.response.status_code} for query: '{query}'")
                return []
            except httpx.RequestError as e:
                logger.error(f"Transport connectivity mapping error connecting to search endpoint for query '{query}': {str(e)}")
                return []
            except Exception as e:
                logger.error(f"Unexpected operational fault inside network task lane for query '{query}': {str(e)}")
                return []

    async def execute_queries(self, queries: List[str]) -> List[SearchResult]:
        """
        Maps a collection of search vectors into parallel processing lanes, safely captures 
        individual task yields, and aggregates them into a flat discovery record array.
        """
        timeout = httpx.Timeout(self.timeout_seconds)
        limits = httpx.Limits(max_connections=120, max_keepalive_connections=40)
        
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            tasks = [self._fetch_query(client, q) for q in queries]
            # return_exceptions=True guards the outer orchestrator event loop from a catastrophic failure
            results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        flattened_results: List[SearchResult] = []
        for res in results_nested:
            if isinstance(res, list):
                flattened_results.extend(res)
            elif isinstance(res, Exception):
                logger.critical(f"Critical execution escape caught at parallel worker aggregation boundary: {str(res)}")

        return flattened_results


class DiscoveryEngine:
    """
    Sub-Query Dimensional Expansion Engine.
    The primary orchestration entry point for Phase 2. Accepts multi-vector taxonomy 
    targets, commands concurrent network proxy lookups, sweeps files via the sifter,
    and exposes a collection of pristine destination targets.
    """

    def __init__(self, search_endpoint: str = "http://127.0.0.1:8081/search", timeout_seconds: float = 12.0) -> None:
        self.aggregator: SearXNGAggregator = SearXNGAggregator(endpoint=search_endpoint, timeout_seconds=timeout_seconds)
        self.sifter: ResultSifter = ResultSifter()

    async def expand_and_discover(self, collection: SubQueryCollection) -> List[str]:
        """
        Main orchestration loop executing the full Phase 2 discovery pipeline.
        Returns a deduplicated list of optimized target URLs ready for extraction.
        """
        logger.info(f"Initiating dimensional query expansion for target topic: '{collection.master_topic}'")
        
        # Merge master query and expansion queries into a single evaluation set
        all_queries = [collection.master_topic] + collection.sub_queries
        # Cleanly deduplicate source search queries prior to launch to save proxy cycles
        unique_queries = list(dict.fromkeys(all_queries))
        
        raw_results = await self.aggregator.execute_queries(unique_queries)
        logger.info(f"Discovery aggregation complete. Captured {len(raw_results)} total raw impression hits.")
        
        cleaned_results = self.sifter.sift_and_deduplicate(raw_results)
        logger.info(f"Sifting sequence finalized. Retained {len(cleaned_results)} unique, high-authority data link frameworks.")
        
        return [result.url for result in cleaned_results]
