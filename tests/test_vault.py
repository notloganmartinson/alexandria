# tests/test_vault.py
import pytest
import os
import shutil
from unittest.mock import AsyncMock, MagicMock
from alexandria.storage.vault import AlexandriaVault

@pytest.fixture
def mock_embedder_factory():
    """Provides a lightweight mock embedder that adapts to custom dimensions for O(1) testing."""
    def _create_mock(dim=768):
        embedder = MagicMock()
        embedder.dimension = dim
        embedder.embed_batch = AsyncMock(side_effect=lambda texts, **kwargs: [[0.1] * dim for _ in texts])
        embedder.embed_text = AsyncMock(side_effect=lambda text, **kwargs: [0.1] * dim)
        return embedder
    return _create_mock

@pytest.fixture
def temp_vault_dir(tmp_path):
    vault_dir = str(tmp_path / "test_vault")
    yield vault_dir
    if os.path.exists(vault_dir):
        shutil.rmtree(vault_dir)

@pytest.mark.asyncio
async def test_dynamic_vector_dimensions(temp_vault_dir, mock_embedder_factory):
    custom_dim = 384
    embedder = mock_embedder_factory(dim=custom_dim)
    vault = AlexandriaVault(embedder=embedder, storage_path=temp_vault_dir, vector_dim=384)
    
    record = {
        "chunk_id": "test_dynamic_001",
        "text": "Testing dimension scaling.",
        "vector": [0.5] * custom_dim,
        "metadata": {"chunk_type": "fallback_character"},
        "relations": []
    }
    
    success = await vault.write_records([record])
    assert success is True
    assert vault.table.schema.field("vector").type.list_size == custom_dim

@pytest.mark.asyncio
async def test_vault_write_and_bipartite_hydration(temp_vault_dir, mock_embedder_factory):
    dim = 3
    embedder = mock_embedder_factory(dim=dim)
    vault = AlexandriaVault(embedder=embedder, storage_path=temp_vault_dir, vector_dim=dim)
    
    records = [
        {
            "chunk_id": "chunk_alpha",
            "text": "OpenAI partnered with Anthropic.",
            "vector": [0.1, 0.2, 0.3],
            "metadata": {"topic_cluster": "AI"},
            "relations": [{"source": "OpenAI", "target": "Anthropic", "relation": "partnered_with"}]
        }
    ]
    
    success = await vault.write_records(records)
    assert success is True
    
    assert vault.table.count_rows() == 1
    assert vault.edges_table.count_rows() == 1
    assert vault.entities_table.count_rows() == 2  
    
    # Verify matrix engine loaded the edges
    df = vault.edges_table.to_pandas()
    assert "chunk_alpha" in df["chunk_id"].values
    # FIX: Assert exact case matched from ingestion schema
    assert "OpenAI" in df["source"].values 

@pytest.mark.asyncio
async def test_ppr_search_execution(temp_vault_dir, mock_embedder_factory):
    """Verifies HippoRAG 2 Matrix PPR execution returns chunk nodes."""
    dim = 3
    embedder = mock_embedder_factory(dim=dim)
    vault = AlexandriaVault(embedder=embedder, storage_path=temp_vault_dir, vector_dim=dim)
    
    records = [
        {
            "chunk_id": "hop_1", 
            "text": "A targets B", 
            "vector": [0.0, 0.0, 0.0], 
            "metadata": {"chunk_type": "test_record"}, 
            "relations": [{"source": "EntityA", "target": "EntityB", "relation": "rel1"}]
        }
    ]
    await vault.write_records(records)
    vault._hydrate_matrix()
    
    # FIX: Match the exact case loaded into the bipartite matrix
    chunks = await vault.matrix_engine.execute_ppr(["EntityA"], top_k=2)
    assert "hop_1" in chunks
