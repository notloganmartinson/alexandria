# src/alexandria/embedder.py
import asyncio
from typing import List, Literal
from sentence_transformers import SentenceTransformer

class LocalEmbedder:
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        print(f"[*] Loading local embedding model: {model_name}...")
        # trust_remote_code=True is strictly required for Nomic's custom architecture
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        print("[+] Embedding model loaded into memory.")
        
    def _sync_embed(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Synchronous CPU/NPU math execution."""
        # Nomic SOTA requirement: Explicitly separate the latent space of documents and queries
        prefix = "search_document: " if task_type == "document" else "search_query: "
        prefixed_texts = [f"{prefix}{text}" for text in texts]
        
        # normalize_embeddings=True is required for Cosine Similarity in LanceDB
        embeddings = self.model.encode(prefixed_texts, normalize_embeddings=True)
        
        # LanceDB expects native Python float lists, not numpy arrays
        return embeddings.tolist()

    async def embed_batch(self, texts: List[str], task_type: Literal["document", "query"] = "document") -> List[List[float]]:
        """Asynchronously embeds a batch of chunks without blocking the main event loop."""
        if not texts:
            return []
        # Offload the heavy tensor math to a background CPU thread
        return await asyncio.to_thread(self._sync_embed, texts, task_type)

    async def embed_text(self, text: str, task_type: Literal["document", "query"] = "document") -> List[float]:
        """Convenience method for a single string (like a user query)."""
        embeddings = await self.embed_batch([text], task_type)
        return embeddings[0]
