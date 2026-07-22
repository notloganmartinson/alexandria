# tests/test_graph_builder.py
import os
import pytest
from dotenv import load_dotenv
from unittest.mock import AsyncMock, MagicMock

from alexandria.storage.vault import AlexandriaVault
from alexandria.graph_builder import OpenRouterExtractor

load_dotenv()

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

@pytest.mark.asyncio
async def test_openrouter_extraction_and_vault_insert(tmp_path, mock_embedder):
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None, "OPENROUTER_API_KEY is missing from the .env file!"
    
    test_vault_dir = str(tmp_path / "integration_vault")
    vault = AlexandriaVault(embedder=mock_embedder, storage_path=test_vault_dir, vector_dim=3)
    extractor = OpenRouterExtractor()
    
    sample_text = (
        "Nvidia recently announced the Blackwell architecture, which significantly "
        "accelerates Large Language Models. Blackwell was designed by Jensen Huang's "
        "engineering team."
    )
    
    print("\n[+] Sending text to OpenRouter for extraction...")
    relations = await extractor.extract_relations(sample_text)
    assert len(relations) > 0, "The extractor returned an empty list!"
    
    record = {
        "chunk_id": "doc_001",
        "text": sample_text,
        "vector": [0.1, 0.2, 0.3],
        "metadata": {"chunk_type": "markdown"},
        "relations": [r.model_dump() for r in relations]
    }
    
    success = await vault.write_records([record])
    
    assert success is True, "Vault failed to write the records!"
    assert vault.table.count_rows() == 1, "LanceDB table should have exactly 1 row."
    assert vault.edges_table.count_rows() > 0, "LanceDB edges table should have extracted nodes."
