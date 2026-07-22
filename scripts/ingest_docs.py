import os
import uuid
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Ensure the src folder is in the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alexandria.extraction import RoutingChunker
from alexandria.embedder import LocalEmbedder
from alexandria.graph_builder import OpenRouterExtractor
from alexandria.storage.vault import AlexandriaVault
from alexandria.storage.paths import resolve_vault_path
from alexandria.storage.schemas import ContentChunkMetadata

load_dotenv()

async def ingest_directory(docs_dir: str = "data/docs"):
    print(f"[*] Starting Production Ingestion Pipeline from {docs_dir}...")
    
    # 1. Initialize Alexandria Components
    chunker = RoutingChunker()
    embedder = LocalEmbedder()
    extractor = OpenRouterExtractor()
    vault = AlexandriaVault(
        embedder=embedder,
        storage_path=str(resolve_vault_path()),
        vector_dim=embedder.dimension
    )
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"[!] Directory {docs_dir} not found. Please create it and add .md files.")
        return

    md_files = list(docs_path.glob("**/*.md"))
    print(f"[*] Found {len(md_files)} markdown files.")
    
    master_records = []
    
    for filepath in md_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        print(f"[*] Processing {filepath.name}...")
        
        # 2. Route through your semantic chunker
        chunks = chunker.process_file(str(filepath), content)
        
        for chunk_data in chunks:
            text = chunk_data["content"]
            metadata_dict = chunk_data["metadata"]
            
            # 3. Generate dense vectors
            vector = await embedder.embed_text(text, task_type="document")
            
            # 4. Extract strict graph relations via OpenRouter
            relations = await extractor.extract_relations(text)
            
            # 5. Construct the final LanceDB record
            record = {
                "chunk_id": str(uuid.uuid4()),
                "text": text,
                "vector": vector,
                "metadata": ContentChunkMetadata(**metadata_dict).model_dump(),
                "relations": [rel.model_dump() for rel in relations]
            }
            master_records.append(record)
            
    # 6. Commit to LanceDB
    print(f"[*] Writing {len(master_records)} records to LanceDB...")
    success = await vault.write_records(master_records)
    
    if success:
        print(f"[+] Successfully ingested {len(master_records)} chunks into the vault.")
    else:
        print("[!] Ingestion failed.")

if __name__ == "__main__":
    asyncio.run(ingest_directory())
