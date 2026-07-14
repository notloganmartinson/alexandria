# src/alexandria/retriever.py
from typing import List, Dict, Any
from alexandria.storage.vault import AlexandriaVault

class HybridRetriever:
    """
    Production Reciprocal Rank Fusion Query Engine.
    Blends dense vectors and structured graphs onto a unified document context domain.
    """
    def __init__(self, vault: AlexandriaVault, rrf_constant: int = 60):
        self.vault = vault
        self.k = rrf_constant

    def compute_rrf(self, vector_results: List[Dict[str, Any]], graph_chunk_ids: List[str]) -> List[str]:
        """Calculates Reciprocal Rank Fusion sorting maps across independent retrieval arrays."""
        scores: Dict[str, float] = {}

        # Parse Vector array ranking order
        for rank, doc in enumerate(vector_results):
            cid = doc["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rank + self.k))

        # Parse Graph array ranking order
        for rank, cid in enumerate(graph_chunk_ids):
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (rank + self.k))

        # Re-rank elements matching combined mathematical distributions
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sorted_docs]

    async def retrieve_context(self, query_vector: List[float], seed_entities: List[str], limit: int = 5) -> str:
        """Executes parallel searches and constructs a highly dense text context string block."""
        # 1. Dispatch queries across systems concurrently
        vector_hits = await self.vault.hybrid_vector_search(query_vector, top_k=20)
        graph_hits = await self.vault.graph_traverse_search(seed_entities, max_depth=2)

        # 2. Intersect systems using standard RRF matrix sorting
        fused_ids = self.compute_rrf(vector_hits, graph_hits)[:limit]

        # 3. Retrieve raw text values from the active database
        hydrated_records = await self.vault.get_chunks_by_ids(fused_ids)
        chunk_map = {rec["chunk_id"]: rec["text"] for rec in hydrated_records}

        # 4. Synthesize final clean Markdown text blocks
        output = "### System Discovery Context\n\n"
        for cid in fused_ids:
            if cid in chunk_map:
                output += f"* {chunk_map[cid]}\n"
        return output
