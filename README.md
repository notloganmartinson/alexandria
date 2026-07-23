# Alexandria

Alexandria is an enterprise-grade, privacy-first knowledge vault and advanced GraphRAG engine. It is designed to act as a local intelligence backend for researchers and Model Context Protocol (MCP) AI agents. Moving beyond traditional RAG constraints, Alexandria implements a dual-level architecture inspired by HippoRAG 2 and LightRAG, fusing multi-hop matrix graph traversals with high-level community abstraction routing.

The core philosophy of Alexandria is maximum privacy and local execution, ensuring personal vault data remains strictly on the user's hardware while providing state-of-the-art (SOTA) retrieval fidelity.

## Core Architecture

The current engine is fully modularized and mathematically verified via automated testing. It consists of four distinct phases:

*   **Discovery Engine (`discovery.py`):** An asynchronous web harvester that canonicalizes URLs and aggressively strips DOM clutter to extract clean, dense Markdown from research targets.
*   **Graph Builder (`extraction.py` / `graph_builder.py`):** Utilizes `instructor` and structured LLM outputs to parse unstructured Markdown into strict Entity-Relation triples (Source -> Target -> Relation), safely hydrating heavily enforced PyArrow schemas to prevent downstream ingestion panics.
*   **Alexandria Vault (`vault.py`):** The decentralized storage layer. Replacing legacy in-memory NetworkX pickling, the vault uses a multi-table **LanceDB** deployment (records, edges, entities, communities). It executes vector-based $O(1)$ entity canonicalization during ingestion and instantly hydrates a decoupled **SciPy Bipartite Matrix** (`Chunk <-> Entity`) for highly optimized memory graph operations.
*   **Hybrid Retriever (`retriever.py`):** A dynamic intent-routing query engine. It intelligently directs broad conceptual queries to fetch LLM-generated community summaries (LightRAG abstraction), while routing specific queries to a Reciprocal Rank Fusion (RRF) pipeline that mathematically blends dense vector ANN search with SciPy-accelerated Personalized PageRank (PPR) matrix traversals.

## Development & Testing

The system relies on `uv` for lightning-fast dependency management and `pytest-asyncio` for executing the robust test suite. Strict schema validation is enforced to interface seamlessly with the Rust/Arrow backend. The hybrid RAG logic is continuously benchmarked using `ragas` (LLM-as-a-judge) to ensure multi-hop factual context is never dropped and to validate recall improvements against standard vector baselines.

To run the integration tests and the RRF benchmark:

```bash
# Load environment variables (e.g., OPENROUTER_API_KEY) and run the suite
uv run pytest -v -s
