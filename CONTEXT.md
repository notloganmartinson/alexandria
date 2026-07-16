# Alexandria: AI Developer Context & System Architecture

This file provides context for LLMs and AI coding agents operating on this repository. It defines the current verified engine state, the open-source distribution goals, and future milestones.

## Core Philosophy & Distribution
* **Open Source & Portable:** Alexandria is an open-source tool built to be easily accessible, highly portable, and trivial to configure for other users.
* **Local-First Privacy:** Designed to act as an offline-first knowledge vault for researchers and AI agents, keeping user data strictly on their own local hardware.
* **Environment Agnostic:** Code must never use hardcoded absolute system paths. Storage paths and environment variables must resolve relatively or dynamically to ensure seamless installation across different machines.

## Core Architecture State (Verified)
Alexandria is a local, privacy-first Hybrid RAG and GraphRAG engine combining semantic vector lookup with topological graph routing.
* **Storage Backend (`vault.py`):** Multi-modal storage using LanceDB (768-dimension vectors) and NetworkX (directed entity-relation graphs).
* **Ingestion (`graph_builder.py`):** Uses `instructor` to extract clean JSON Graph Triples via OpenRouter.
* **Retrieval (`retriever.py`):** Fuses standard vector hits and multi-hop graph nodes using Reciprocal Rank Fusion (RRF).

## Implementation Directives

### 1. SOTA Local Embeddings (`embedder.py`)
Transition the engine from mock vectors to completely offline embeddings using `nomic-ai/nomic-embed-text-v1.5` (768 dimensions).
* **Requirement:** Must prepend `"search_document: "` for data ingestion and `"search_query: "` for retrieval questions.
* **Concurrency:** Embedding inference must be wrapped in `asyncio.to_thread` to prevent blocking the async scraping/ingestion loops.

### 2. Standalone CLI & Unified Entrypoint (`main.py`)
Expose a native terminal interface for foreground operations.
* **Command `/ingest <url>`:** Executes discovery engine, generates local Nomic document embeddings, triggers graph extraction, and commits to storage.
* **Command `/ask <query>`:** Embeds the question as a query vector, runs `HybridRetriever`, and feeds the context to an LLM for a conversational response.

### 3. Model Context Protocol Server (`mcp_server.py`)
Expose the engine capabilities as tool call parameters for agents using `FastMCP`.
* Wrap the discovery, extraction, local embedding, and hybrid retrieval loops into fast tools exposed over standard I/O (`stdio`).
* This enables external coding agents (Goose, Cursor, Claude Desktop) to autonomously read from and write to the local vault.

### 4. Background Incremental RAPTOR Daemon (`raptor_daemon.py`)
Implement a two-speed database approach via an asynchronous background daemon to enable global macro-level summarization.
* **Trunk/Leaf Architecture:** Foreground pipeline handles rapid flat-RAG and GraphRAG insertion. The background daemon handles tree-building.
* **Incremental Tree Building:** The daemon must scan for unsummarized leaf nodes, calculate clusters (UMAP/GMM), generate cluster summaries via LLM, and incrementally update only the affected branches of the summary tree to prevent total tree reconstruction overhead.

## Guidelines for AI Generation
* **User-Facing Design:** Write clean, informative console output (logs/progress indicators) so other users can see exactly what the engine is executing.
* **Dependency Discipline:** Use explicit `uv` dependency management rules. Avoid polluting the global python namespace.
* Never use outdated embedding formats or drop Nomic task prefixes.
* Keep async and heavy synchronous tasks strictly isolated.
