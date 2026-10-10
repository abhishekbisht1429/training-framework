"""`RepositoryReader`: everything the HPC layer reads from Git.

Files are read at the resolved commit, never from the worktree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol

from training_framework.hpc.errors import HpcError


class RepositoryReader(Protocol):
    root: Path

    def resolve(self, ref: str) -> str: ...

    def is_clean(self) -> bool: ...

    def read(self, sha: str, path: str) -> bytes | None: ...

    def exists(self, sha: str, path: str) -> bool: ...

    def object_id(self, sha: str, path: str) -> str | None: ...


class GitRepository:
    """`RepositoryReader` over the `git` executable."""

    def __init__(self, start: Path):
        result = self._git(["rev-parse", "--show-toplevel"], cwd=start)
        if result.returncode != 0:
            raise HpcError(f"{start} is not inside a Git repository")
        self.root = Path(result.stdout.decode().strip())

    @staticmethod
    def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, check=False,
        )

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        return self._git(args, self.root)

    def resolve(self, ref: str) -> str:
        result = self._run(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"])
        if result.returncode != 0:
            raise HpcError(f"cannot resolve '{ref}' to a commit")
        return result.stdout.decode().strip()

    def is_clean(self) -> bool:
        result = self._run(["status", "--porcelain", "--untracked-files=normal"])
        if result.returncode != 0:
            raise HpcError(f"git status failed: {result.stderr.decode().strip()}")
        return not result.stdout.strip()

    def read(self, sha: str, path: str) -> bytes | None:
        result = self._run(["show", f"{sha}:{path}"])
        return result.stdout if result.returncode == 0 else None

    @staticmethod
    def _spec(sha: str, path: str) -> str:
        # Git names the root tree `<sha>:`, not `<sha>:.`.
        path = path.strip("/")
        return f"{sha}:" if path in ("", ".") else f"{sha}:{path}"

    def exists(self, sha: str, path: str) -> bool:
        return self._run(["cat-file", "-e", self._spec(sha, path)]).returncode == 0

    def object_id(self, sha: str, path: str) -> str | None:
        """Git's content hash of a file or directory at `sha`."""
        result = self._run(["rev-parse", "--verify", "--quiet", self._spec(sha, path)])
        return result.stdout.decode().strip() if result.returncode == 0 else None

    @staticmethod
    def bundle_ref(sha: str) -> str:
        return f"refs/hpc/stage-{sha}"

    def create_bundle(self, sha: str, path: str) -> None:
        """A bundle holding `sha` (`git bundle` needs a ref, so a temporary
        one is created and always deleted)."""
        ref = self.bundle_ref(sha)
        result = self._run(["update-ref", ref, sha])
        if result.returncode != 0:
            raise HpcError(f"git update-ref failed: {result.stderr.decode().strip()}")
        try:
            result = self._run(["bundle", "create", "--quiet", path, ref])
            if result.returncode != 0:
                raise HpcError(f"git bundle failed: {result.stderr.decode().strip()}")
        finally:
            self._run(["update-ref", "-d", ref])

