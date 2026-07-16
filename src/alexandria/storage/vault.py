# src/alexandria/storage/vault.py
import os
import pickle
import asyncio
import lancedb
import pyarrow as pa
import networkx as nx
from typing import List, Dict, Any, Optional
from lancedb.pydantic import LanceModel, pydantic_to_schema
from alexandria.storage.schemas import IngestionDocumentRecord

class AlexandriaVault:
    """
    Frontier-grade Hybrid RAG Vault.
    Maintains concurrent async transaction boundaries for LanceDB vector spaces 
    and a local NetworkX Directed Graph.
    """
    def __init__(self, storage_path: str = "./.alexandria_vault"):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Async synchronization primitives to protect concurrent operations
        self._lock = asyncio.Lock()
        
        # Initialize Vector Store
        self.db = lancedb.connect(os.path.join(self.storage_path, "lancedb"))
        self.table_name = "vault_records"
        
        # Derive schema directly from our production Pydantic layout
        self.arrow_schema = pydantic_to_schema(IngestionDocumentRecord)
        
        # Embeddings configurations (Vector array must map exactly to dimension bounds)
        # We extend the schema with a 384-dimension vector field for embeddings models
        self.extended_schema = pa.schema(
            list(self.arrow_schema) + [pa.field("vector", pa.list_(pa.float32(), 768))]
        )
        
        if self.table_name not in self.db.list_tables():
            self.table = self.db.create_table(self.table_name, schema=self.extended_schema)
        else:
            self.table = self.db.open_table(self.table_name)
            
        # Initialize Directed Knowledge Graph
        self.graph_path = os.path.join(self.storage_path, "knowledge_graph.gpickle")
        self._load_graph()

    def _load_graph(self):
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
        else:
            self.graph = nx.DiGraph()

    def _sync_save_graph(self):
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)

    async def write_records(self, records: List[Dict[str, Any]]) -> bool:
        """
        Thread-safe ingestion pipeline. 
        Accepts records matching the schema extended with a 'vector' array.
        """
        async with self._lock:
            try:
                if not records:
                    return True
                    
                # 1. Update LanceDB Vector Storage
                self.table.add(records)
                
                # 2. Extract and link graph nodes
                for record in records:
                    chunk_id = record["chunk_id"]
                    relations = record.get("relations", [])
                    
                    for rel in relations:
                        # Handle both dict configurations and raw Pydantic schemas dynamically
                        src = rel["source"] if isinstance(rel, dict) else rel.source
                        tgt = rel["target"] if isinstance(rel, dict) else rel.target
                        edge_type = rel["relation"] if isinstance(rel, dict) else rel.relation
                        
                        if not self.graph.has_node(src):
                            self.graph.add_node(src, source_chunks=set())
                        self.graph.nodes[src]["source_chunks"].add(chunk_id)
                        
                        if not self.graph.has_node(tgt):
                            self.graph.add_node(tgt, source_chunks=set())
                        self.graph.nodes[tgt]["source_chunks"].add(chunk_id)
                        
                        self.graph.add_edge(src, tgt, relation=edge_type)
                        
                await asyncio.to_thread(self._sync_save_graph) 
                return True
            except Exception as e:
                print(f"Ingestion Error: {e}")
                return False

    async def hybrid_vector_search(self, query_vector: List[float], top_k: int = 15) -> List[Dict[str, Any]]:
        """Executes a thread-safe dense vector semantic distance check."""
        async with self._lock:
            return self.table.search(query_vector).limit(top_k).to_list()

    async def graph_traverse_search(self, seed_entities: List[str], max_depth: int = 2) -> List[str]:
        """Executes multi-hop entity traversal to find highly connected chunk IDs."""
        async with self._lock:
            chunk_hits: Dict[str, float] = {}
            
            for entity in seed_entities:
                if entity in self.graph:
                    # Mentions explicitly present in query get immediate weight boost
                    for cid in self.graph.nodes[entity].get("source_chunks", []):
                        chunk_hits[cid] = chunk_hits.get(cid, 0.0) + 3.0
                        
                    # Graph structural hop neighborhood expansion traversal loop
                    edges = nx.bfs_edges(self.graph, source=entity, depth_limit=max_depth)
                    for u, v in edges:
                        for cid in self.graph.nodes[v].get("source_chunks", []):
                            chunk_hits[cid] = chunk_hits.get(cid, 0.0) + 1.0
                            
            sorted_chunks = sorted(chunk_hits.items(), key=lambda x: x[1], reverse=True)
            return [chunk_id for chunk_id, _ in sorted_chunks]

    async def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Hydrates full text payloads matching high-ranked target IDs."""
        async with self._lock:
            if not chunk_ids:
                return []
            # Sanitization mapping for SQL statement formatting injection protection
            ids_str = ", ".join([f"'{c}'" for c in chunk_ids])
            return self.table.search().where(f"chunk_id IN ({ids_str})").to_list()
