from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


def chmod_tree(root: Path | str, mode: int) -> None:
    """Apply one exact mode to a filesystem tree without following child symlinks."""
    root = Path(root)
    if not root.exists():
        return

    def raise_walk_error(error: OSError) -> None:
        raise error

    for current, directories, files in os.walk(
        root,
        topdown=False,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        current_path = Path(current)
        for name in files:
            path = current_path / name
            if not path.is_symlink():
                path.chmod(mode)
        for name in directories:
            path = current_path / name
            if not path.is_symlink():
                path.chmod(mode)
        current_path.chmod(mode)


def chmod_trees(roots: Iterable[Path | str], mode: int) -> None:
    for root in roots:
        chmod_tree(root, mode)
