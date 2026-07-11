import os
from pathlib import Path
from unittest.mock import patch

from alexandria.storage.paths import resolve_vault_path


def test_resolve_vault_path_windows() -> None:
    with (
        patch("platform.system", return_value="Windows"),
        patch.dict(
            os.environ,
            {"LOCALAPPDATA": "C:\\\\Users\\\\Test\\\\AppData\\\\Local"},
            clear=True,
        ),
    ):
        expected = (
            Path("C:\\\\Users\\\\Test\\\\AppData\\\\Local")
            / "alexandria"
            / "lancedb-store"
        )
        assert resolve_vault_path() == expected


def test_resolve_vault_path_darwin() -> None:
    with (
        patch("platform.system", return_value="Darwin"),
        patch("pathlib.Path.home", return_value=Path("/Users/Test")),
    ):
        expected = Path(
            "/Users/Test/Library/Application Support/alexandria/lancedb-store"
        )
        assert resolve_vault_path() == expected


def test_resolve_vault_path_linux_xdg() -> None:
    with (
        patch("platform.system", return_value="Linux"),
        patch.dict(
            os.environ, {"XDG_DATA_HOME": "/home/test/.local/share"}, clear=True
        ),
    ):
        expected = Path("/home/test/.local/share") / "alexandria" / "lancedb-store"
        assert resolve_vault_path() == expected


def test_resolve_vault_path_linux_fallback() -> None:
    with (
        patch("platform.system", return_value="Linux"),
        patch.dict(os.environ, {}, clear=True),
        patch("pathlib.Path.home", return_value=Path("/home/test")),
    ):
        expected = Path("/home/test/.local/share") / "alexandria" / "lancedb-store"
        assert resolve_vault_path() == expected
