import os
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv

# Use standard OpenAI client for OpenRouter
from openai import AsyncOpenAI
from datasets import Dataset

# Modern Ragas imports
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.embeddings.base import BaseRagasEmbedding
from ragas import aevaluate
# Alexandria imports
from alexandria.embedder import LocalEmbedder
from alexandria.storage.vault import AlexandriaVault
from alexandria.storage.paths import resolve_vault_path

load_dotenv()

class RagasEmbedderAdapter(BaseRagasEmbedding):
    """Modern Ragas Interface wrapping our existing Alexandria LocalEmbedder."""
    def __init__(self, embedder: LocalEmbedder):
        self.embedder = embedder
        super().__init__()

    # --- Modern Ragas Interface ---
    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self.embedder._sync_embed([text], "query")[0]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await self.embedder.embed_text(text, task_type="query")

    def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        return self.embedder._sync_embed(texts, "query")

    async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
        return await self.embedder.embed_batch(texts, task_type="query")
        
    # --- Legacy LangChain Fallbacks (For buggy v0.4 metrics) ---
    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)
        
    async def aembed_query(self, text: str) -> list[float]:
        return await self.aembed_text(text)
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)
        
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.aembed_texts(texts)

async def run_baseline_evaluation(dataset_path: str = "data/eval_data.json"):
    print("[*] Initializing Alexandria Baseline Evaluator...")
    
    # 1. Load Evaluation Data
    with open(dataset_path, "r") as f:
        eval_data = json.load(f)
        
    print(f"[*] Loaded {len(eval_data)} questions from testset.")

    # 2. Setup your local RAG components for answering
    print("[*] Loading local embedding model...")
    embedder = LocalEmbedder()
    vault = AlexandriaVault(
        embedder=embedder,
        storage_path=str(resolve_vault_path())
    )
    
    # 3. Setup OpenRouter LLM using Ragas' native factory (The Fix)
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    # Create a raw async OpenAI client pointed at OpenRouter
    openai_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=10
    )
    
    # Use llm_factory instead of LangchainLLMWrapper
    ragas_llm = llm_factory(
        model="gpt-4o-mini", 
        client=openai_client
    )
    
    # 4. Setup Local Embedder for Grading
    ragas_embeddings = RagasEmbedderAdapter(embedder)
    
    # 5. Generate Answers for the testset
    print("[*] Generating RAG answers to evaluate...")
    results = []
    
    for item in eval_data:
        question = item["question"]
        
        query_vector = await embedder.embed_text(question)
        search_results = vault.table.search(query_vector).limit(3).to_list()
        contexts = [res["text"] for res in search_results]
        
        # Manually invoke OpenRouter for the answer generation
        prompt = f"Answer the question based ONLY on the context.\n\nContext: {contexts}\n\nQuestion: {question}"
        response = await openai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"]
        })
        
    eval_dataset = Dataset.from_pandas(pd.DataFrame(results))
    
    # 6. Run Evaluation Metrics
    print("[*] Running Ragas evaluation metrics...")
    metrics = [
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    ]
    
    evaluation = await aevaluate(
        dataset=eval_dataset,
        metrics=metrics,
        run_config=RunConfig(max_workers=2, max_retries=10)
    )
    
    # 7. Save results
    output_path = "data/baseline_results.json"
    evaluation.to_pandas().to_json(output_path, orient="records", indent=4)
    print(f"\n[+] Baseline Evaluation Complete! Saved to {output_path}")
    print("\n=== Baseline Metrics ===")
    
    print(evaluation)

if __name__ == "__main__":
    asyncio.run(run_baseline_evaluation())

