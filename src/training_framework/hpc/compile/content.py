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
from collections.abc import Callable
from contextlib import closing, contextmanager
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
    """Per-file hashes. Within `session()` one connection serves every lookup
    and new hashes are committed in batches (a transaction per file made
    large datasets several times slower than hashing them); leaving the
    session, even through an interrupt, commits what was hashed, so the next
    run resumes from there."""

    BATCH = 1000

    def __init__(self, path: Path):
        self.path = path
        self._connection = None
        self._pending: list[tuple] = []
        #: Files actually read and hashed (not served from the cache).
        self.hashed = 0
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

    @contextmanager
    def session(self):
        if self._connection is not None:      # already inside one
            yield self
            return
        self._connection = self._connect()
        try:
            yield self
        finally:
            try:
                self._flush()
            finally:
                self._connection.close()
                self._connection = None

    def _flush(self) -> None:
        if self._pending:
            with self._connection:
                self._connection.executemany(
                    "INSERT OR REPLACE INTO hashes VALUES (?, ?, ?, ?, ?, ?, ?)",
                    self._pending,
                )
            self._pending = []

    def file_hash(self, path: str) -> tuple[str, int]:
        with self.session():
            stat = os.stat(path)
            fingerprint = (stat.st_dev, stat.st_ino, stat.st_size,
                           stat.st_mtime_ns, stat.st_ctime_ns)
            row = self._connection.execute(
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
            self.hashed += 1
            self._pending.append((path, *fingerprint, sha))
            if len(self._pending) >= self.BATCH:
                self._flush()
            return sha, stat.st_size


#: progress(path, files done, total files, bytes done)
Progress = Callable[[str, int, int, int], None]


def content_summary(path: str, cache: HashCache, progress: Progress | None = None) -> ContentSummary:
    """The content key of a file or directory. Symbolic links to directories
    are refused (they make the manifest depend on what lies outside);
    symbolic links to files are hashed by content."""
    root = os.path.abspath(path)
    if os.path.isfile(root):
        names = [(os.path.basename(root), root)]
    else:
        names = []
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
                names.append((os.path.relpath(full, root), full))
    entries: list[list] = []
    total = 0
    with cache.session():
        for done, (relative, full) in enumerate(names, start=1):
            sha, size = cache.file_hash(os.path.realpath(full))
            entries.append([relative, size, sha])
            total += size
            if progress is not None:
                progress(root, done, len(names), total)
    entries.sort()
    return ContentSummary(hash_value(entries), len(entries), total)
