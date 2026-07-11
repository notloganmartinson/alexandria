# src/alexandria/extraction.py
import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Set

import httpx
import trafilatura

# Conforms to library logging standard
logger = logging.getLogger("alexandria.extraction")


@dataclass(frozen=True)
class ProvenanceManifest:
    """
    Data layout strictly aligning with Phase 1 LanceDB source_registry schema requirements.
    Establishes cryptographically verifiable provenance for extracted knowledge nodes.
    """
    source_id: str
    source_uri: str
    raw_text_hash: str
    harvest_timestamp: int  # Fixed: Aligns perfectly with pa.int64() Unix epoch milestones
    total_nodes_generated: int = 0
    raptor_compilation_state: bool = False  # Fixed: Aligns perfectly with pa.bool_() layouts


@dataclass(frozen=True)
class ExtractedDocument:
    """Immutable data structure representing a successfully harvested and processed web document."""
    url: str
    markdown_content: str
    manifest: ProvenanceManifest


@dataclass(frozen=True)
class EvaluationResult:
    """Data structure encapsulating the algorithmic quality score of extracted text."""
    is_valid: bool
    signal_score: float
    rejection_reason: Optional[str]


class DensityEvaluator:
    """
    Signal-to-Noise Lexical Density Evaluator.
    Executes heavy string manipulation and regex targeting to compute an internal technical 
    signal score. Discards CAPTCHAs, cookie banners, and low-density shell pages.
    """

    def __init__(self) -> None:
        self.minimum_word_count: int = 75
        self.minimum_lexical_variety: float = 0.35
        self.rejection_patterns: List[re.Pattern[str]] = [
            re.compile(r"(?i)enable javascript to view"),
            re.compile(r"(?i)please verify you are human"),
            re.compile(r"(?i)checking your browser before accessing"),
            re.compile(r"(?i)access denied \- security check"),
            re.compile(r"(?i)accept all cookies"),
            re.compile(r"(?i)captcha challenge"),
        ]
        self.code_block_pattern = re.compile(r"```[\s\S]*?```")

    def _sync_evaluate(self, text: str) -> EvaluationResult:
        """Synchronous evaluation core designed to run inside an isolated thread pool."""
        if not text or not text.strip():
            return EvaluationResult(False, 0.0, "Empty content payload")

        for pattern in self.rejection_patterns:
            if pattern.search(text):
                return EvaluationResult(False, 0.0, "Matched malicious or blocking pattern (CAPTCHA/JS-gate)")

        words = text.split()
        word_count = len(words)
        if word_count < self.minimum_word_count:
            return EvaluationResult(False, 0.0, f"Low word count ({word_count} < {self.minimum_word_count})")

        unique_words = set(w.lower() for w in words)
        lexical_variety = len(unique_words) / word_count
        if lexical_variety < self.minimum_lexical_variety:
            return EvaluationResult(False, lexical_variety, f"Low lexical variety ({lexical_variety:.2f})")

        code_blocks = self.code_block_pattern.findall(text)
        code_char_count = sum(len(block) for block in code_blocks)
        total_char_count = len(text)
        
        # Documents with high code density get a signal boost in a technical library
        code_ratio = code_char_count / total_char_count if total_char_count > 0 else 0
        signal_score = min(1.0, lexical_variety + (code_ratio * 0.5))

        return EvaluationResult(True, signal_score, None)

    async def evaluate(self, text: str) -> EvaluationResult:
        """Asynchronous wrapper to offload CPU-bound string analysis to a background thread."""
        return await asyncio.to_thread(self._sync_evaluate, text)


class HeuristicExtractor:
    """
    Thread-Isolated Layout Stripper.
    Executes DOM layout stripping and markdown generation using trafilatura logic.
    """

    def _sync_extract(self, html_payload: bytes) -> Optional[str]:
        """Synchronous CPU-bound DOM traversal and boilerplate removal."""
        try:
            extracted_text = trafilatura.extract(
                html_payload,
                include_comments=False,
                include_tables=True,
                include_links=True,
                format="markdown",
                no_fallback=False
            )
            return extracted_text
        except Exception as e:
            logger.error(f"Catastrophic failure during heuristic DOM extraction: {str(e)}")
            return None

    async def extract_text(self, html_payload: bytes) -> Optional[str]:
        """Asynchronous wrapper protecting the main event loop from CPU-bound parsing pauses."""
        return await asyncio.to_thread(self._sync_extract, html_payload)


class WebHarvester:
    """
    Asynchronous Protected Content Harvester.
    Implements malicious asset shielding, strict limits, and parallel stream execution.
    """

    def __init__(self, max_concurrent_streams: int = 10, max_content_length: int = 10 * 1024 * 1024) -> None:
        self.max_content_length: int = max_content_length
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_streams)
        
        self.allowed_content_types: Set[str] = {
            "text/html",
            "text/plain",
            "text/markdown",
            "application/xhtml+xml"
        }
        
        self.user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (Alexandria-Vault/1.0)"
        )
        self.timeout = httpx.Timeout(15.0, connect=5.0)
        self.limits = httpx.Limits(max_connections=50, max_keepalive_connections=20)

    def _is_valid_content_type(self, content_type_header: str) -> bool:
        """Evaluates HTTP Content-Type headers against the approved safelist."""
        if not content_type_header:
            return False
        base_type = content_type_header.split(";")[0].strip().lower()
        return base_type in self.allowed_content_types

    async def _fetch_single_url(self, client: httpx.AsyncClient, url: str) -> Optional[bytes]:
        """
        Executes an isolated stream request, intercepting headers to prevent zip bombs
        and unreadable binary streams from allocating into system memory.
        """
        async with self._semaphore:
            try:
                # Stream the response to intercept headers before pulling the payload body
                async with client.stream("GET", url, follow_redirects=True) as response:
                    response.raise_for_status()

                    content_type = response.headers.get("Content-Type", "")
                    if not self._is_valid_content_type(content_type):
                        logger.warning(f"Rejected {url} due to invalid Content-Type: {content_type}")
                        return None

                    content_length_str = response.headers.get("Content-Length")
                    if content_length_str and content_length_str.isdigit():
                        if int(content_length_str) > self.max_content_length:
                            logger.warning(f"Rejected {url} due to oversized Content-Length: {content_length_str} bytes")
                            return None

                    payload = bytearray()
                    async for chunk in response.aiter_bytes():
                        payload.extend(chunk)
                        if len(payload) > self.max_content_length:
                            logger.warning(f"Rejected {url}: Stream exceeded dynamic allocation limit.")
                            return None
                    
                    return bytes(payload)

            except httpx.TooManyRedirects:
                logger.error(f"Redirect loop detected for {url}. Connection dropped.")
                return None
            except httpx.TimeoutException:
                logger.error(f"Connection timeout while streaming {url}.")
                return None
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP {e.response.status_code} returned for {url}.")
                return None
            except httpx.RequestError as e:
                logger.error(f"Network transport error for {url}: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected operational fault harvesting {url}: {str(e)}")
                return None


class ExtractionPipeline:
    """
    Master orchestrator for Phase 3.
    Accepts pristine target URLs, marshals concurrent harvesting, executes heuristic 
    extraction, calculates lexical density, and structures the final provenanced data layouts.
    """

    def __init__(self) -> None:
        self.harvester = WebHarvester()
        self.extractor = HeuristicExtractor()
        self.evaluator = DensityEvaluator()

    def _generate_manifest(self, url: str, text: str) -> ProvenanceManifest:
        """
        Constructs the strict cryptographic manifest linking Phase 3 extraction back to Phase 1 storage.
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Fixed: Generates a deterministic hash from the URL layout to protect system idempotency
        source_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
        
        # Fixed: Converts current time into millisecond-accurate Unix epoch integer values
        epoch_timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)

        return ProvenanceManifest(
            source_id=source_id,
            source_uri=url,
            raw_text_hash=text_hash,
            harvest_timestamp=epoch_timestamp,
            total_nodes_generated=0,
            raptor_compilation_state=False
        )

    async def _process_single_target(self, client: httpx.AsyncClient, url: str) -> Optional[ExtractedDocument]:
        """Pipeline lane for a single URL: Fetch -> Extract -> Evaluate -> Manifest."""
        raw_html = await self.harvester._fetch_single_url(client, url)
        if not raw_html:
            return None

        extracted_text = await self.extractor.extract_text(raw_html)
        if not extracted_text:
            logger.warning(f"Heuristic extraction yielded no usable layout for {url}")
            return None

        evaluation = await self.evaluator.evaluate(extracted_text)
        if not evaluation.is_valid:
            logger.warning(f"Signal density evaluation failed for {url}. Reason: {evaluation.rejection_reason}")
            return None

        logger.info(f"Successfully processed {url} with signal score: {evaluation.signal_score:.2f}")
        manifest = self._generate_manifest(url, extracted_text)
        
        return ExtractedDocument(
            url=url,
            markdown_content=extracted_text,
            manifest=manifest
        )

    async def execute_extraction(self, urls: List[str]) -> List[ExtractedDocument]:
        """
        Consumes an array of target URLs and processes them concurrently through the 
        protected extraction boundaries. Returns validated ExtractedDocument objects.
        """
        logger.info(f"Initializing extraction pipeline for {len(urls)} targets.")
        headers = {"User-Agent": self.harvester.user_agent}
        
        async with httpx.AsyncClient(
            headers=headers,
            timeout=self.harvester.timeout,
            limits=self.harvester.limits,
            max_redirects=4
        ) as client:
            tasks = [self._process_single_target(client, url) for url in urls]
            results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        valid_documents: List[ExtractedDocument] = []
        for res in results_nested:
            if isinstance(res, Exception):
                logger.critical(f"Critical execution escape caught at harvester parallel boundary: {str(res)}")
            elif res is not None:
                valid_documents.append(res)

        logger.info(f"Extraction pipeline finalized. Yielded {len(valid_documents)} validated documents.")
        return valid_documents
