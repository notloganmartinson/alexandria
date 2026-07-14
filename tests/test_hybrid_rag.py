# tests/test_hybrid_rag.py
import pytest
import os
import shutil
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import ContextPrecision, ContextRecall
from ragas.llms import llm_factory 
from openai import OpenAI
from alexandria.storage.vault import AlexandriaVault
from alexandria.retriever import HybridRetriever

TEST_VAULT_DIR = "./.test_system_hybrid_vault"

@pytest.fixture()
async def operational_vault():
    """Initializes and tears down temporary database clusters matching the core production system schema."""
    if os.path.exists(TEST_VAULT_DIR):
        shutil.rmtree(TEST_VAULT_DIR)
        
    vault = AlexandriaVault(storage_path=TEST_VAULT_DIR)
    dummy_vector = [0.0] * 384
    
    # Ingestion dataset strictly adhering to ContentChunkMetadata and IngestionDocumentRecord structures
    mock_records = [
        {
            "chunk_id": "doc_chunk_alpha",
            "text": "The engineering framework named Project Omega is explicitly managed by Principal Engineer Sarah.",
            "vector": dummy_vector,
            "metadata": {
                "url": "https://internal.dev/omega",
                "domain": "internal.dev",
                "timestamp_epoch": 1774212000,
                "lexical_density_score": 0.74,
                "topic_cluster": "engineering_ops"
            },
            "relations": [{"source": "Project Omega", "target": "Sarah", "relation": "managed_by"}]
        },
        {
            "chunk_id": "doc_chunk_beta",
            "text": "Sarah directly maintains all escalation pipelines routing to Director Marcus.",
            "vector": dummy_vector,
            "metadata": {
                "url": "https://internal.dev/teams",
                "domain": "internal.dev",
                "timestamp_epoch": 1774212100,
                "lexical_density_score": 0.65,
                "topic_cluster": "corporate_hierarchy"
            },
            "relations": [{"source": "Sarah", "target": "Marcus", "relation": "reports_to"}]
        },
        {
            "chunk_id": "doc_chunk_gamma",
            "text": "Director Marcus issued a mandatory engineering structural spending freeze today.",
            "vector": dummy_vector,
            "metadata": {
                "url": "https://internal.dev/fiscal",
                "domain": "internal.dev",
                "timestamp_epoch": 1774212200,
                "lexical_density_score": 0.82,
                "topic_cluster": "finance_announcements"
            },
            "relations": [{"source": "Marcus", "target": "Spending Freeze", "relation": "issued"}]
        }
    ]
    
    success = await vault.write_records(mock_records)
    assert success is True, "Ingestion loop failed to write standard schema records."
    
    yield vault
    
    # Clean up database resources post-run
    if os.path.exists(TEST_VAULT_DIR):
        shutil.rmtree(TEST_VAULT_DIR)

@pytest.mark.asyncio
async def test_multi_hop_fusion_retrieval(operational_vault):
    """Verifies that multi-hop graph routing correctly captures linked nodes via RRF."""
    retriever = HybridRetriever(vault=operational_vault)
    
    query = "Is Project Omega impacted by any corporate spending limits?"
    seed_entities = ["Project Omega"]
    blind_vector = [0.0] * 384
    
    context_output = await retriever.retrieve_context(
        query_vector=blind_vector, seed_entities=seed_entities, limit=3
    )
    
    # Verify that RRF successfully pulled the un-indexed multi-hop chunk via graph connections
    assert "spending freeze" in context_output.lower(), "RRF failed to surface the multi-hop graph connection."

    # 1. Point the native OpenAI client to your local llama.cpp server
    openai_client = OpenAI(
        api_key="not-needed",
        base_url="http://localhost:8082/v1"
    )
    
    # 2. Use the new Ragas llm_factory to create the InstructorLLM
    evaluator_llm = llm_factory(
        "qwen2.5-coder-7b-instruct", 
        client=openai_client
    )

        # 3. Pass the wrapped Judge to the Metrics
    precision_metric = ContextPrecision(llm=evaluator_llm)
    recall_metric = ContextRecall(llm=evaluator_llm)

    # Execute LLM-as-a-judge precision tracking sweeps via Ragas
    eval_dataset = Dataset.from_dict({
        "question": [query],
        "ground_truth": ["Yes. Project Omega is run by Sarah, who reports to Marcus, and Marcus issued a spending freeze."],
        "contexts": [[context_output]],
        "answer": [""] # Isolated strictly to retrieval profiling
    })
    
    eval_metrics = evaluate(
        dataset=eval_dataset, 
        metrics=[precision_metric, recall_metric]
    )
    
    print("\n==========================================")
    print("      ALEXANDRIA HYBRID RAG BENCHMARK      ")
    print("==========================================")
    print(f"Context Precision Score : {eval_metrics['context_precision']:.4f}")
    print(f"Context Recall Score    : {eval_metrics['context_recall']:.4f}")
    print("==========================================\n")
    
    assert eval_metrics["context_recall"] == 1.0, "System context expansion dropped crucial factual tracks."
