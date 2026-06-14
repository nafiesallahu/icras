"""File hashing utilities for the ICRAS intake pipeline.

The intake gate must record a tamper-evident fingerprint for every document it
admits so downstream agents (and human auditors) can prove that the artifact
they are reasoning about is byte-for-byte identical to what was ingested. A
SHA256 digest is the canonical fingerprint used throughout ICRAS.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Read files in fixed-size chunks so hashing never loads a large PDF fully into
# memory. 64 KiB is a good balance between syscall overhead and footprint.
_CHUNK_SIZE = 64 * 1024


def sha256_file(path: str | Path) -> str:
    """Compute the SHA256 hex digest of a file's contents.

    Args:
        path: Filesystem path to the file to hash.

    Returns:
        The lowercase 64-character hexadecimal SHA256 digest.

    Raises:
        FileNotFoundError: If the path does not exist or is not a regular file.
        OSError: If the file cannot be read (for example, due to permissions).
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {file_path}")

    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
