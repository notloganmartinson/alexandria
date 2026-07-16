# Alexandria

Alexandria is an enterprise-grade, privacy-first knowledge graph and retrieval-augmented generation (RAG) engine. It is designed to act as a local knowledge vault for researchers and AI agents, fusing semantic vector search with multi-hop graph routing to retrieve highly dense context that traditional RAG systems fail to surface. 

The core philosophy of Alexandria is maximum privacy and local execution, ensuring personal vault data remains strictly on the user's hardware.

## Core Architecture

The current engine is fully modularized and mathematically verified via automated testing. It consists of four distinct phases:

* **Discovery Engine (`discovery.py`):** An asynchronous web harvester that canonicalizes URLs and aggressively strips DOM clutter to extract clean, dense Markdown from research targets.
* **Graph Builder (`extraction.py` / `graph_builder.py`):** Utilizes `instructor` and structured LLM outputs to parse unstructured Markdown into strict Entity-Relation triples (Source -> Target -> Relation) without hallucinating schema.
* **Alexandria Vault (`vault.py`):** The local storage backend. It uses **LanceDB** for high-speed, zero-copy vector storage and **NetworkX** to map the multi-hop topological relationships between extracted entities.
* **Hybrid Retriever (`retriever.py`):** When queried, this module executes an isolated vector search and a graph traversal, mathematically fusing the two distinct data streams via Reciprocal Rank Fusion (RRF).

## Development & Testing

The system relies on `uv` for lightning-fast dependency management and `pytest-asyncio` for executing the test suite. The hybrid RAG logic is continuously benchmarked using `ragas` (LLM-as-a-judge) to ensure multi-hop factual context is never dropped.

To run the integration tests and the RRF benchmark:

```bash
# Load environment variables (e.g., OPENROUTER_API_KEY) and run the suite
uv run pytest -v -s
