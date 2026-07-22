# tests/test_hybrid_rag.py
import pytest
import os
import shutil
import warnings
from unittest.mock import AsyncMock, MagicMock

from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import ContextPrecision, ContextRecall
from ragas.llms import llm_factory 
from openai import OpenAI
from dotenv import load_dotenv

from alexandria.storage.schemas import ContentChunkMetadata
from alexandria.storage.vault import AlexandriaVault
from alexandria.retriever import HybridRetriever

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

TEST_VAULT_DIR = "./.test_system_hybrid_vault"

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.dimension = 3
    
    def make_vec(text):
        val = float(sum(ord(c) for c in text) % 100) / 100.0
        return [val, val, val]
        
    embedder.embed_batch = AsyncMock(side_effect=lambda texts, **kwargs: [make_vec(t) for t in texts])
    embedder.embed_text = AsyncMock(side_effect=lambda text, **kwargs: make_vec(text))
    return embedder

@pytest.fixture()
async def operational_vault(mock_embedder):
    if os.path.exists(TEST_VAULT_DIR):
        shutil.rmtree(TEST_VAULT_DIR)
        
    vault = AlexandriaVault(embedder=mock_embedder, storage_path=TEST_VAULT_DIR, vector_dim=3)
    dummy_vector = [0.0, 0.0, 0.0]
    
    # Generate a mathematically perfect schema struct to satisfy the Rust backend
    safe_metadata = ContentChunkMetadata(
        chunk_type="test_record",
        url="", domain="", timestamp_epoch=0, lexical_density_score=0.0,
        topic_cluster="", file_path="", node_type="", start_line=0,
        end_line=0, language="en", header=""
    ).model_dump()
    
    mock_records = [
        {
            "chunk_id": "doc_chunk_alpha",
            "text": "Project Omega is explicitly managed by Principal Engineer Sarah.",
            "vector": dummy_vector,
            "metadata": safe_metadata, 
            "relations": [{"source": "Project Omega", "target": "Sarah", "relation": "managed_by"}]
        }
    ]
    
    success = await vault.write_records(mock_records)
    assert success is True, "Vault failed to write test records."
    
    vault._hydrate_matrix() 
    yield vault
    
    if os.path.exists(TEST_VAULT_DIR):
        shutil.rmtree(TEST_VAULT_DIR)

@pytest.mark.asyncio
async def test_multi_hop_fusion_retrieval(operational_vault):
    retriever = HybridRetriever(vault=operational_vault)
    
    query = "Is Project Omega impacted by any corporate spending limits?"
    seed_entities = ["Project Omega"]
    blind_vector = [0.0, 0.0, 0.0]
    
    context_output = await retriever.retrieve_context(
        query_vector=blind_vector, seed_entities=seed_entities, limit=3
    )
    
    assert "Sarah" in context_output, "RRF failed to surface the PPR matrix node."
