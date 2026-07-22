import os
import json
from dotenv import load_dotenv

# Ragas & LangChain (v0.4.x native compliant)
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Modern Ragas imports (avoids deprecation warnings)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.run_config import RunConfig
# Alexandria imports
from alexandria.storage.vault import AlexandriaVault
from alexandria.storage.paths import resolve_vault_path

load_dotenv()

def build_testset(output_path: str = "data/eval_data.json", testset_size: int = 50):
    print("[*] Booting Alexandria Vault to extract chunk payloads...")
    
    # 1. Connect to LanceDB and extract all chunks
    embedder = LocalEmbedder()
    vault = AlexandriaVault(
        embedder=embedder,
        storage_path=str(resolve_vault_path()))
    
    # Read entire chunk table into Pandas memory
    df = vault.table.to_pandas()
    
    if df.empty:
        print("[!] Vault is empty. Ingest documents before generating testsets.")
        return

    # 2. Format LanceDB rows into LangChain Document objects for Ragas
    docs = []
    for _, row in df.iterrows():
        docs.append(Document(
            page_content=row["text"],
            metadata={"filename": str(row["chunk_id"])}  
        ))
        
    print(f"[+] Extracted {len(docs)} chunks from LanceDB.")
    
    # 3. Setup OpenRouter Generator LLM
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("ALEXANDRIA_EXTRACTION_MODEL", "openai/gpt-4o-mini")
    
    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=10,
    )
    generator_llm = LangchainLLMWrapper(llm)
    
    # 4. Setup Local Nomic Embeddings
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1.5",
        model_kwargs={"trust_remote_code": True}
    )
    generator_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    
    # 5. Initialize TestsetGenerator
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    
    print(f"[*] Generating {testset_size} synthetic evaluation questions...")
    
    # 6. Generate adversarial dataset
    # Force Ragas to process a maximum of 2 requests at a time
    dataset = generator.generate_with_langchain_docs(
        docs, 
        testset_size=testset_size,
        run_config=RunConfig(max_workers=2, max_retries=10)
    )
    # 7. Format to match baseline output mapping
    formatted_dataset = []
    for row in dataset.to_pandas().to_dict(orient="records"):
        formatted_dataset.append({
            "question": row["user_input"],
            "ground_truth": row["reference"],
            "query_type": row.get("synthesizer_name", "unknown")
        })
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(formatted_dataset, f, indent=4)
        
    print(f"[+] Testset generation complete. Saved to {output_path}")

if __name__ == "__main__":
    build_testset()
