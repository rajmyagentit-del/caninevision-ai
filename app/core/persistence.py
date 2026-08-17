from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


DEFAULT_PERSISTENT_ROOT = (
    "/content/drive/MyDrive/CanineVisionAI"
)


def get_persistent_root() -> Path | None:
    """
    Resolve the persistent storage root.

    Priority:
    1. CANINEVISION_PERSISTENT_ROOT environment variable
    2. Google Drive default path if mounted
    3. None if no persistent storage is available
    """

    configured_root = os.getenv(
        "CANINEVISION_PERSISTENT_ROOT"
    )

    if configured_root:
        root = Path(configured_root)

        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        return root

    default_root = Path(
        DEFAULT_PERSISTENT_ROOT
    )

    if default_root.parent.exists():
        default_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        return default_root

    return None


def ensure_subdirectory(
    root: Path,
    name: str,
) -> Path:
    """
    Create and return a persistent subdirectory.
    """

    path = root / name

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def persist_file(
    source: Path,
    category: str,
    destination_name: str | None = None,
) -> Path | None:
    """
    Copy one file to persistent storage.

    Returns:
        Persistent destination path if successful.

        None if persistent storage is unavailable.
    """

    if not source.exists():
        raise FileNotFoundError(
            f"Cannot persist missing file: {source}"
        )

    persistent_root = (
        get_persistent_root()
    )

    if persistent_root is None:
        return None

    destination_directory = (
        ensure_subdirectory(
            persistent_root,
            category,
        )
    )

    destination = (
        destination_directory
        / (
            destination_name
            if destination_name
            else source.name
        )
    )

    shutil.copy2(
        source,
        destination,
    )

    return destination


def persist_checkpoint(
    source: Path,
    destination_name: str | None = None,
) -> Path | None:
    """
    Persist a model checkpoint.
    """

    return persist_file(
        source=source,
        category="checkpoints",
        destination_name=destination_name,
    )


def persist_metrics(
    source: Path,
    destination_name: str | None = None,
) -> Path | None:
    """
    Persist metrics or training-history files.
    """

    return persist_file(
        source=source,
        category="metrics",
        destination_name=destination_name,
    )


def save_json(
    data: dict[str, Any] | list[Any],
    path: Path,
) -> None:
    """
    Save JSON safely with readable formatting.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")

    temporary_path.replace(
        path
    )


def persistence_status() -> dict[str, Any]:
    """
    Return a simple status object describing
    whether persistent storage is available.
    """

    root = get_persistent_root()

    return {
        "available": root is not None,
        "root": (
            str(root)
            if root is not None
            else None
        ),
    }