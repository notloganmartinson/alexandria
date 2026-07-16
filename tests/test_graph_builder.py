# tests/test_graph_builder.py
import os
import pytest
from dotenv import load_dotenv
from alexandria.storage.vault import AlexandriaVault
from alexandria.graph_builder import OpenRouterExtractor

# Automatically load the .env file from the root directory so the API key is active
load_dotenv()

@pytest.mark.asyncio
async def test_openrouter_extraction_and_vault_insert(tmp_path):
    """
    Integration test: Proves we can call OpenRouter to extract strict JSON edges
    and successfully save them into the LanceDB/NetworkX vault.
    """
    # 1. Verify the key actually loaded
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key is not None, "OPENROUTER_API_KEY is missing from the .env file!"

    # 2. Spin up a temporary database vault for this test
    test_vault_dir = str(tmp_path / "integration_vault")
    vault = AlexandriaVault(storage_path=test_vault_dir)
    extractor = OpenRouterExtractor()

    # 3. The raw markdown we want the LLM to process
    sample_text = (
        "Nvidia recently announced the Blackwell architecture, which significantly "
        "accelerates Large Language Models. Blackwell was designed by Jensen Huang's "
        "engineering team."
    )

    # 4. Execute the API Call
    print("\n[+] Sending text to OpenRouter for extraction...")
    relations = await extractor.extract_relations(sample_text)

    # 5. Assert the LLM followed instructions and returned data
    assert len(relations) > 0, "The extractor returned an empty list!"
    print(f"[+] Successfully extracted {len(relations)} relations:")
    for r in relations:
        print(f"    [{r.source}] --({r.relation})--> [{r.target}]")

    # 6. Build the record (mocking the 768-dim Nomic vector for now)
    mock_vector = [0.1] * 768
    record = {
        "chunk_id": "doc_001",
        "text": sample_text,
        "vector": mock_vector,
        "metadata": {
            "url": "https://tech-news.com/blackwell",
            "domain": "tech-news.com",
            "timestamp_epoch": 1774212000,
            "lexical_density_score": 0.85,
            "topic_cluster": "AI Hardware"
        },
        "relations": [r.model_dump() for r in relations]
    }

    # 7. Assert that LanceDB and NetworkX successfully saved the LLM's output
    print("\n[+] Saving LLM outputs to the hybrid vault...")
    success = await vault.write_records([record])
    
    assert success is True, "Vault failed to write the records!"
    assert vault.table.count_rows() == 1, "LanceDB table should have exactly 1 row."
    assert len(vault.graph.nodes) > 0, "NetworkX graph should have extracted nodes."
    print("[+] Save verified successfully.")
