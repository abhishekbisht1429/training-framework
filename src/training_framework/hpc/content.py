"""Content keys of local data (`upload` sources), with a per-file hash cache.

A key is the hash of a deterministic content manifest: every file's
relative path, size, and SHA-256. Hashing a large dataset is slow, so
per-file hashes are cached, keyed by everything that changes when a file's
content may have changed: path, device, inode, size, `mtime_ns`, and
`ctime_ns` (a content write changes ctime even if size and mtime are put
back). The cache is SQLite, so every write is atomic and concurrent writers
stay valid.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from training_framework.hpc.names import hash_value

_CHUNK = 1 << 20


def default_cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache",
    )
    return Path(base) / "training-framework" / "hashes.sqlite3"


@dataclass(frozen=True)
class ContentSummary:
    key: str
    files: int
    size: int


class HashCache:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS hashes ("
                " path TEXT PRIMARY KEY, device INTEGER, inode INTEGER,"
                " size INTEGER, mtime_ns INTEGER, ctime_ns INTEGER,"
                " sha256 TEXT)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60)

    def file_hash(self, path: str) -> tuple[str, int]:
        stat = os.stat(path)
        fingerprint = (stat.st_dev, stat.st_ino, stat.st_size,
                       stat.st_mtime_ns, stat.st_ctime_ns)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT device, inode, size, mtime_ns, ctime_ns, sha256 "
                "FROM hashes WHERE path = ?", (path,),
            ).fetchone()
        if row is not None and tuple(row[:5]) == fingerprint:
            return row[5], stat.st_size
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO hashes VALUES (?, ?, ?, ?, ?, ?, ?)",
                (path, *fingerprint, sha),
            )
        return sha, stat.st_size


def content_summary(path: str, cache: HashCache) -> ContentSummary:
    """The content key of a file or directory. Symbolic links to directories
    are refused (they make the manifest depend on what lies outside);
    symbolic links to files are hashed by content."""
    entries: list[list] = []
    total = 0
    root = os.path.abspath(path)
    if os.path.isfile(root):
        sha, size = cache.file_hash(root)
        entries.append([os.path.basename(root), size, sha])
        total = size
    else:
        for directory, subdirectories, files in os.walk(root):
            subdirectories.sort()
            for name in subdirectories:
                if os.path.islink(os.path.join(directory, name)):
                    raise ValueError(
                        f"{os.path.join(directory, name)} is a symbolic link "
                        "to a directory; upload sources must not contain them"
                    )
            for name in sorted(files):
                full = os.path.join(directory, name)
                sha, size = cache.file_hash(os.path.realpath(full))
                entries.append([os.path.relpath(full, root), size, sha])
                total += size
    entries.sort()
    return ContentSummary(hash_value(entries), len(entries), total)
