"""Record what a saved model was built from.

A model file on its own cannot answer "which code and which data made this?".
Six months later that is the first question you have, so the answer is stamped
into the bundle: the git commit, whether the tree was dirty, and a checksum of
the dataset.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import subprocess
from pathlib import Path

from football import paths

log = logging.getLogger(__name__)

#: Single seed for everything that can be seeded.
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Seed Python, NumPy and the hash randomisation used by some sklearn paths."""
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def git_revision() -> dict[str, str | bool | None]:
    """Current commit and whether the working tree is dirty.

    Returns ``None`` values when git is unavailable or this is not a checkout,
    which is a normal situation, not an error.
    """

    def _run(*args: str) -> str | None:
        try:
            # Fixed command with fixed arguments and no shell -- `args` comes
            # only from the literal call sites below, never from user input.
            result = subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607  - resolved from PATH by design
                cwd=paths.ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
    }


def file_checksum(path: Path, chunk_size: int = 1 << 20) -> str | None:
    """SHA-256 of *path*, or ``None`` if it is missing."""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect(dataset: Path | None = None) -> dict:
    """Everything worth stamping into a saved model."""
    dataset = dataset or paths.MATCHES
    return {
        "git": git_revision(),
        "dataset": str(dataset.relative_to(paths.ROOT))
        if dataset.is_relative_to(paths.ROOT)
        else str(dataset),
        "dataset_sha256": file_checksum(dataset),
        "seed": SEED,
    }
