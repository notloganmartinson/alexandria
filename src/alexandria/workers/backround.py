# src/alexandria/workers/background.py
import asyncio
from alexandria.storage.vault import AlexandriaVault

class BackgroundWorker:
    """Handles heavy IO/CPU operations decoupled from the main ingestion event loop."""
    def __init__(self, vault: AlexandriaVault):
        self.vault = vault

    def _sync_rebuild_fts(self):
        """Forces Tantivy FTS reconstruction over the local store."""
        print("[*] Rebuilding Tantivy FTS Index...")
        self.vault.table.create_fts_index("text", replace=True)
        print("[+] FTS Indexing Complete.")
        
    async def trigger_fts_rebuild(self):
        """Non-blocking execution of index reconstruction."""
        await asyncio.to_thread(self._sync_rebuild_fts)
