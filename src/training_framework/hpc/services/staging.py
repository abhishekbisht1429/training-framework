"""`SourceStager`: puts the exact commit on the cluster, detached.

`bundle` (default) needs no Git credentials on the cluster; `fetch` clones a
configured URL there and verifies the commit is reachable. Dirty changes are
never included; nothing is committed or pushed.
"""

from __future__ import annotations

import os
import tempfile

from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports import FileTransfer, RemoteExecutor


class SourceStager:
    def __init__(self, repository, executor: RemoteExecutor, transfer: FileTransfer,
                 strategy: str = "bundle", url: str | None = None):
        if strategy not in ("bundle", "fetch") or (strategy == "fetch") != (url is not None):
            raise ValueError("strategy 'fetch' needs a url; 'bundle' takes none")
        self.repository = repository
        self.executor = executor
        self.transfer = transfer
        self.strategy = strategy
        self.url = url

    def _git(self, argv: list[str], what: str) -> None:
        result = self.executor.run(["git", *argv])
        if result.status != 0:
            raise HpcError(f"{what} failed on the cluster: {result.stderr.strip()}")

    def stage(self, sha: str, target: str) -> None:
        if self.strategy == "bundle":
            remote_bundle = f"{target}.bundle"
            with tempfile.TemporaryDirectory() as directory:
                local_bundle = os.path.join(directory, "source.bundle")
                self.repository.create_bundle(sha, local_bundle)
                self.transfer.sync(local_bundle, remote_bundle)
            # A clone only fetches branches and tags; fetch the bundle's one
            # temporary ref explicitly.
            origin = remote_bundle
            self._git(["init", "--quiet", target], "git init")
            self._git(["-C", target, "fetch", "--quiet", origin,
                       self.repository.bundle_ref(sha)], "git fetch")
        else:
            origin = self.url
            self._git(["clone", "--quiet", "--no-checkout", origin, target], "git clone")
        verify = self.executor.run(["git", "-C", target, "cat-file", "-e", f"{sha}^{{commit}}"])
        if verify.status != 0:
            raise HpcError(f"commit {sha} is not reachable from {origin}")
        self._git(["-C", target, "checkout", "--quiet", "--detach", sha], "git checkout")
