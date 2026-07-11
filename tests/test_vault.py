# tests/test_vault.py
import asyncio
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from lancedb import AsyncConnection

from alexandria.storage.vault import AlexandriaStorageVault


@pytest.fixture
def temp_vault_dir() -> Generator[Path, None, None]:
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Normal teardown; handles cleanup after file locks are closed
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


@pytest.mark.asyncio
async def test_vault_initialization_and_schema_enforcement(
    temp_vault_dir: Path,
) -> None:
    with patch(
        "alexandria.storage.vault.resolve_vault_path", return_value=temp_vault_dir
    ):
        vault = AlexandriaStorageVault()

        await vault.initialize()

        connection = await vault.get_connection()
        assert isinstance(connection, AsyncConnection)

        _tables_res_1 = await connection.list_tables()
        tables = _tables_res_1.tables if hasattr(_tables_res_1, "tables") else list(_tables_res_1)

        assert "knowledge_nodes" in tables
        assert "source_registry" in tables

        kn_table = await connection.open_table("knowledge_nodes")
        kn_schema = await kn_table.schema()
        assert "node_id" in kn_schema.names
        assert "vector" in kn_schema.names

        sr_table = await connection.open_table("source_registry")
        sr_schema = await sr_table.schema()
        assert "source_id" in sr_schema.names
        assert "raw_text_hash" in sr_schema.names

        # Tests idempotency pathway (re-init should not leak connections or drop state)
        await vault.initialize()
        _tables_res_2 = await connection.list_tables()
        tables_after_second_init = _tables_res_2.tables if hasattr(_tables_res_2, "tables") else list(_tables_res_2)
        assert len(tables_after_second_init) == 2
        
        # Graceful cleanup to release file tracking handles
        await vault.close()


@pytest.mark.asyncio
async def test_vault_uninitialized_connection_error() -> None:
    vault = AlexandriaStorageVault()
    with pytest.raises(RuntimeError, match="Database connection not initialized"):
        await vault.get_connection()


@pytest.mark.asyncio
async def test_vault_concurrent_initialization(temp_vault_dir: Path) -> None:
    with patch("alexandria.storage.vault.resolve_vault_path", return_value=temp_vault_dir):
        vault = AlexandriaStorageVault()
        
        tasks = [vault.initialize() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        connection = await vault.get_connection()
        _tables_res_3 = await connection.list_tables()
        tables = _tables_res_3.tables if hasattr(_tables_res_3, "tables") else list(_tables_res_3)
        
        assert "knowledge_nodes" in tables
        assert "source_registry" in tables
        
        await vault.close()


@pytest.mark.asyncio
async def test_vault_graceful_close_lifecycle(temp_vault_dir: Path) -> None:
    """Verifies that closing resets internal connection state and severs access handles cleanly."""
    with patch("alexandria.storage.vault.resolve_vault_path", return_value=temp_vault_dir):
        vault = AlexandriaStorageVault()
        await vault.initialize()
        
        # Ensure it works while hot
        conn = await vault.get_connection()
        assert conn is not None
        
        # Trigger close primitive
        await vault.close()
        
        # Assert state is locked down and subsequent retrievals throw proper sequentially errors
        with pytest.raises(RuntimeError, match="Database connection not initialized"):
            await vault.get_connection()


@pytest.mark.asyncio
async def test_vault_close_uninitialized_idempotency() -> None:
    """Verifies that calling close on a blank instance handles gracefully without throwing exceptions."""
    vault = AlexandriaStorageVault()
    try:
        await vault.close()
    except Exception as e:
        pytest.fail(f"Closing an uninitialized vault threw an unexpected error: {str(e)}")
