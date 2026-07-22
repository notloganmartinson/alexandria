# src/alexandria/graph/matrix.py
import asyncio
import numpy as np
import scipy.sparse as sp
import pandas as pd
from typing import List, Dict, Tuple

class PPRMatrixEngine:
    """
    SOTA Bipartite Matrix Engine for HippoRAG 2 Personalized PageRank.
    Offloads graph traversal to highly optimized C/Fortran BLAS routines via SciPy.
    """
    def __init__(self):
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: List[str] = []
        self.is_chunk: np.ndarray = np.array([])
        self.P_T: sp.csr_matrix = None

    def rebuild(self, edges_df: pd.DataFrame) -> None:
        """Constructs a row-normalized bipartite adjacency matrix from raw edges."""
        if edges_df.empty:
            return

        # 1. Extract Unique Nodes
        chunks = edges_df['chunk_id'].unique()
        src_ents = edges_df['source'].unique()
        tgt_ents = edges_df['target'].unique()
        
        entities = np.unique(np.concatenate([src_ents, tgt_ents]))
        all_nodes = np.concatenate([chunks, entities])
        
        self.idx_to_node = list(all_nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.idx_to_node)}
        
        n_nodes = len(self.idx_to_node)
        self.is_chunk = np.zeros(n_nodes, dtype=bool)
        self.is_chunk[:len(chunks)] = True

        # 2. Build Bipartite Adjacency Matrix
        row_idx, col_idx, data = [], [], []

        def add_edge(u: str, v: str, weight: float = 1.0):
            row_idx.append(self.node_to_idx[u])
            col_idx.append(self.node_to_idx[v])
            data.append(weight)

        for _, row in edges_df.iterrows():
            src, tgt, cid = row['source'], row['target'], row['chunk_id']
            # Bipartite flow: Chunk <-> Entity
            add_edge(cid, src)
            add_edge(src, cid)
            add_edge(cid, tgt)
            add_edge(tgt, cid)
            # Entity <-> Entity flow
            add_edge(src, tgt)
            add_edge(tgt, src)

        A = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n_nodes, n_nodes))
        
        # 3. Transition Matrix Normalization (P = D^-1 * A)
        row_sums = np.array(A.sum(axis=1)).flatten()
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0 
        D_inv = sp.diags(1.0 / row_sums)
        
        P = D_inv @ A
        # Transpose for the Power Iteration algorithm
        self.P_T = P.T.tocsr()

    def _ppr_sync(self, seed_nodes: List[str], alpha: float = 0.85, max_iter: int = 50, tol: float = 1e-6) -> List[Tuple[str, float]]:
        """Synchronous power iteration calculating $x = \\alpha P^T x + (1-\\alpha) v$."""
        if self.P_T is None:
            return []

        n_nodes = len(self.idx_to_node)
        v = np.zeros(n_nodes)
        
        seed_indices = [self.node_to_idx[n] for n in seed_nodes if n in self.node_to_idx]
        if not seed_indices:
            return []
            
        v[seed_indices] = 1.0 / len(seed_indices)
        x = v.copy()

        # Power Iteration
        for _ in range(max_iter):
            x_new = alpha * (self.P_T @ x) + (1 - alpha) * v
            if np.linalg.norm(x_new - x, 1) < tol:
                break
            x = x_new

        # Filter strictly for chunk nodes
        chunk_scores = x[self.is_chunk]
        chunk_indices = np.where(self.is_chunk)[0]
        
        results = [
            (self.idx_to_node[idx], float(score)) 
            for idx, score in zip(chunk_indices, chunk_scores) 
            if score > 0
        ]
        
        return sorted(results, key=lambda i: i[1], reverse=True)

    async def execute_ppr(self, seed_nodes: List[str], top_k: int = 15) -> List[str]:
        """Async wrapper to offload matrix operations to a background thread."""
        results = await asyncio.to_thread(self._ppr_sync, seed_nodes)
        return [cid for cid, _ in results[:top_k]]
