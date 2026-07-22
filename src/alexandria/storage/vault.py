# src/alexandria/storage/vault.py
import os
import asyncio
from typing import List, Dict, Any

import lancedb
import pandas as pd

from alexandria.storage.schemas import (
    create_document_schema, 
    create_entity_schema, 
    create_community_schema, 
    GraphEdgeRecord
)
from alexandria.graph.matrix import PPRMatrixEngine
from alexandria.embedder import LocalEmbedder

class AlexandriaVault:
    """
    Decoupled Production Storage Layer.
    Executes O(1) canonicalization via Vector ANN and manages LanceDB I/O.
    """
    def __init__(self, embedder: LocalEmbedder, storage_path: str = "./.alexandria_vault", vector_dim: int = 768):
        self.storage_path = storage_path
        self.vector_dim = vector_dim
        self.embedder = embedder
        os.makedirs(self.storage_path, exist_ok=True)
        
        self._lock = asyncio.Lock()
        
        self.db = lancedb.connect(os.path.join(self.storage_path, "lancedb"))
        self._init_tables()
        
        self.matrix_engine = PPRMatrixEngine()
        self._hydrate_matrix()

    def _init_tables(self):
        doc_schema = create_document_schema(self.vector_dim)
        ent_schema = create_entity_schema(self.vector_dim)
        comm_schema = create_community_schema(self.vector_dim)
        
        self.table = self.db.create_table("vault_records", schema=doc_schema, exist_ok=True)
        self.edges_table = self.db.create_table("vault_edges", schema=GraphEdgeRecord, exist_ok=True)
        self.entities_table = self.db.create_table("vault_entities", schema=ent_schema, exist_ok=True)
        self.communities_table = self.db.create_table("vault_communities", schema=comm_schema, exist_ok=True)

    def _hydrate_matrix(self):
        """Builds the Scipy Sparse Matrix into RAM on boot."""
        df = self.edges_table.to_pandas()
        self.matrix_engine.rebuild(df)

    async def _canonicalize_entities(self, raw_names: List[str], distance_threshold: float = 0.15) -> Dict[str, str]:
        """
        Replaces RapidFuzz with Sub-Millisecond Dense Vector ANN.
        Distance threshold 0.15 roughly corresponds to 0.85 Cosine Similarity.
        """
        unique_names = list(set([n.strip() for n in raw_names if n.strip()]))
        if not unique_names:
            return {}
            
        embeddings = await self.embedder.embed_batch(unique_names, task_type="query")
        mapping = {}
        new_entities = []
        
        # We query LanceDB for each entity to find the nearest canonical match
        for name, vec in zip(unique_names, embeddings):
            try:
                hits = self.entities_table.search(vec).limit(1).to_list()
                if hits and hits[0]["_distance"] < distance_threshold:
                    mapping[name] = hits[0]["name"]
                else:
                    mapping[name] = name
                    new_entities.append({"name": name, "vector": vec})
            except Exception:
                # Triggers if the table is empty
                mapping[name] = name
                new_entities.append({"name": name, "vector": vec})
                
        # Batch insert newly discovered entities
        if new_entities:
            self.entities_table.add(new_entities)
            
        return mapping

    async def write_records(self, records: List[Dict[str, Any]]) -> bool:
        """Thread-safe fast ingestion. Deliberately skips FTS Indexing to prevent I/O blocking."""
        async with self._lock:
            try:
                if not records:
                    return True
                
                # Extract all unique entity names for batch canonicalization
                all_raw_ents = []
                for r in records:
                    for rel in r.get("relations", []):
                        all_raw_ents.extend([
                            rel["source"] if isinstance(rel, dict) else rel.source,
                            rel["target"] if isinstance(rel, dict) else rel.target
                        ])
                        
                canonical_map = await self._canonicalize_entities(all_raw_ents)
                
                edge_records = []
                for record in records:
                    chunk_id = record["chunk_id"]
                    for rel in record.get("relations", []):
                        raw_src = rel["source"] if isinstance(rel, dict) else rel.source
                        raw_tgt = rel["target"] if isinstance(rel, dict) else rel.target
                        edge_type = rel["relation"] if isinstance(rel, dict) else rel.relation
                        
                        edge_records.append({
                            "source": canonical_map[raw_src],
                            "target": canonical_map[raw_tgt],
                            "relation": edge_type,
                            "chunk_id": chunk_id
                        })
                        
                # Batch Insertions
                self.table.add(records)
                if edge_records:
                    self.edges_table.add(edge_records)
                    # Non-blocking trigger to update matrix mapping
                    asyncio.create_task(asyncio.to_thread(self._hydrate_matrix))
                    
                return True
            except Exception as e:
                print(f"Ingestion Error: {e}")
                return False

    async def hybrid_vector_search(self, query_vector: List[float], top_k: int = 15) -> List[str]:
        async with self._lock:
            results = self.table.search(query_vector).limit(top_k).to_list()
            return [res["chunk_id"] for res in results]

    async def fts_search(self, query: str, top_k: int = 15) -> List[str]:
        async with self._lock:
            try:
                results = self.table.search(query, query_type="fts").limit(top_k).to_list()
                return [res["chunk_id"] for res in results]
            except Exception:
                return []

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        async with self._lock:
            if not chunk_ids:
                return []
            sanitized = [c.replace("'", "''") for c in chunk_ids if isinstance(c, str)]
            ids_str = ", ".join([f"'{c}'" for c in sanitized])
            return self.table.search().where(f"chunk_id IN ({ids_str})").to_list()
