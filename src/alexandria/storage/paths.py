# this module dynamically determines the appropriate host operating system and scopes the data directory accordinlgy falling back to POSIX-compliant standard data homes
import os
import platform
from pathlib import Path


def resolve_vault_path() -> Path:
    system = platform.system().lower()

    if system == "windows":
        base_dir = Path(
            os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
        )
        return base_dir / "alexandria" / "lancedb-store"

    elif system == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "alexandria"
            / "lancedb-store"
        )

    else:
        base_dir = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
        return base_dir / "alexandria" / "lancedb-store"
