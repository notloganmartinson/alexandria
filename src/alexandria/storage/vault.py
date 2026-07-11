# src/alexandria/storage/vault.py
import asyncio
import inspect
import logging
import platform
import subprocess
from pathlib import Path
from typing import Optional

import lancedb
from lancedb import AsyncConnection

from .paths import resolve_vault_path
from .schemas import KNOWLEDGE_NODES_SCHEMA, SOURCE_REGISTRY_SCHEMA


class AlexandriaStorageVault:
    def __init__(self) -> None:
        self.vault_path: Path = resolve_vault_path()
        self._write_lock: asyncio.Lock = asyncio.Lock()
        self._db: Optional[AsyncConnection] = None
        
        # Conforms to library logging standard (delegates formatting/handlers to the top-level main application)
        self.logger: logging.Logger = logging.getLogger("alexandria.storage.vault")

    async def initialize(self) -> None:
        """
        Initializes the file-system directories, enforces runtime schema validations,
        and provisions tables with strict PyArrow layouts. This method is entirely 
        idempotent and safe to run across multiple app cycles.
        """
        async with self._write_lock:
            # 1. Idempotency Check: Short-circuit if a connection pool is already active
            if self._db is not None:
                self.logger.debug("Storage vault initialization called on an already active connection.")
                return

            try:
                # 2. Directory Scaffolding
                self.vault_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Storage path resolved and scaffolded: {self.vault_path}")

                # 3. Arch Linux / Btrfs Fragmentation Mitigation
                if platform.system().lower() == "linux":
                    try:
                        # Sets the +C attribute (No Copy-on-Write) to eliminate database write fragmentation
                        result = subprocess.run(
                            ["chattr", "+C", str(self.vault_path)], 
                            capture_output=True, 
                            text=True, 
                            check=False
                        )
                        if result.returncode == 0:
                            self.logger.info("Successfully applied No-CoW filesystem attribute to storage directory.")
                        else:
                            self.logger.debug(f"chattr statement skipped or non-applicable: {result.stderr.strip()}")
                    except Exception as fs_err:
                        self.logger.debug(f"Fs-attribute adjustment unviable for host filesystem environment: {str(fs_err)}")

                # 4. Connection Establishment
                self._db = await lancedb.connect_async(str(self.vault_path))
                self.logger.info("Asynchronous LanceDB connection established.")

                # 5. Schema Validation & Enforcements
                _tables_response = await self._db.list_tables()
                existing_tables = _tables_response.tables if hasattr(_tables_response, "tables") else list(_tables_response)

                # Table A: knowledge_nodes
                if "knowledge_nodes" not in existing_tables:
                    self.logger.info("Provisioning 'knowledge_nodes' table with defined schema.")
                    await self._db.create_table(
                        "knowledge_nodes", schema=KNOWLEDGE_NODES_SCHEMA, mode="create"
                    )
                else:
                    self.logger.info("'knowledge_nodes' table found. Verifying integrity...")
                    table = await self._db.open_table("knowledge_nodes")
                    rows = await table.count_rows()
                    self.logger.info(f"'knowledge_nodes' integrity verified. Row count: {rows}")

                # Table B: source_registry
                if "source_registry" not in existing_tables:
                    self.logger.info("Provisioning 'source_registry' table with defined schema.")
                    await self._db.create_table(
                        "source_registry", schema=SOURCE_REGISTRY_SCHEMA, mode="create"
                    )
                else:
                    self.logger.info("'source_registry' table found. Verifying integrity...")
                    table = await self._db.open_table("source_registry")
                    rows = await table.count_rows()
                    self.logger.info(f"'source_registry' integrity verified. Row count: {rows}")

                self.logger.info("Vault initialization and schema enforcement complete.")

            except Exception as e:
                self.logger.critical(
                    f"Failed to initialize Alexandria Storage Vault: {str(e)}",
                    exc_info=True,
                )
                # Nullify broken connection traces if step 5 fails to complete perfectly
                self._db = None
                raise RuntimeError(f"Vault Initialization Error: {str(e)}") from e

    async def get_connection(self) -> AsyncConnection:
        """
        Retrieves the active asynchronous database connection pool.
        Raises a RuntimeError if invoked out of sequence.
        """
        if self._db is None:
            raise RuntimeError(
                "Database connection not initialized. Execute 'await initialize()' first."
            )
        return self._db

    async def close(self) -> None:
        """
        Gracefully terminates active connection handles and flushes internal transaction logs 
        to disk to guarantee absolute database consistency on shutdown.
        """
        async with self._write_lock:
            if self._db is None:
                self.logger.debug("Close method executed on an uninitialized vault connection.")
                return

            try:
                # Dynamically verifies client-level close routes across LanceDB engine versions
                if hasattr(self._db, "close"):
                    if inspect.iscoroutinefunction(self._db.close):
                        await self._db.close()
                    else:
                        self._db.close()
                
                self._db = None
                self.logger.info("LanceDB connection pool closed cleanly.")
            except Exception as close_err:
                self.logger.error(f"Error encountered during connection pool teardown: {str(close_err)}")
                self._db = None
                raise RuntimeError(f"Vault Teardown Error: {str(close_err)}") from close_err
