Phase 1: Bipartite & Community Ingestion (Offline)

    Extract: OpenRouter extracts entities and relations from your chunks.

    Bipartite Link (HippoRAG 2): We inject the actual Chunk IDs into NetworkX as nodes. We draw edges from Chunk Node -> Entity Node.

    Community Abstract (LightRAG): We run community detection (e.g., Louvain) on the entity graph to find thematic clusters. We ask OpenRouter to summarize each cluster into a single "High-Level Abstract" vector.

    Vectorize: Everything (Chunks, Entities, Community Abstracts) gets stored in LanceDB.

Phase 2: Tri-Route Retrieval (Online)
When a user asks a question, we use instructor to extract both Local Entities and Global Themes from the query.

    The Hippo Route: We vector-match the Local Entities, seed the NetworkX Personalized PageRank algorithm, and retrieve the highest-probability raw chunks.

    The Light Route: We vector-match the Global Themes against our pre-computed Community Abstracts to pull high-level conceptual context.

    The FTS Route: We fallback to Tantivy BM25 lexical search just in case the graph missed a specific keyword.

    Fusion: We run variadic Reciprocal Rank Fusion (RRF) across all outputs and hand the pristine context to the LLM.
