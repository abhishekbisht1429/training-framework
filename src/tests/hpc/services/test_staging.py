import subprocess

import pytest

from training_framework.hpc.errors import HpcError
from training_framework.hpc.ports.repository import GitRepository
from training_framework.hpc.ports.ssh import RsyncTransfer, SshExecutor
from training_framework.hpc.services.staging import SourceStager
from tests.hpc.batch_repo import make_repo
from tests.hpc.fake_cluster import fake_cluster  # noqa: F401 (fixture)


def head(root):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=True).stdout.strip()


def refs(root):
    return subprocess.run(["git", "for-each-ref", "--format=%(refname)"], cwd=root,
                          capture_output=True, text=True, check=True).stdout.split()


def test_a_bundle_stages_the_exact_commit_detached(fake_cluster, tmp_path):
    root = make_repo(tmp_path)
    sha = head(root)
    before = refs(root)
    (root / "batch.yaml").write_text("dirty: true\n")          # never staged
    target = str(tmp_path / "cluster" / "source")
    stager = SourceStager(GitRepository(root), SshExecutor("cluster"), RsyncTransfer("cluster"))
    stager.stage(sha, target)
    assert head(target) == sha
    assert "dirty" not in (tmp_path / "cluster" / "source" / "batch.yaml").read_text()
    assert subprocess.run(["git", "-C", target, "symbolic-ref", "-q", "HEAD"]).returncode != 0
    assert refs(root) == before                                 # temporary ref deleted


def test_fetch_clones_a_url_and_verifies_the_commit(fake_cluster, tmp_path):
    root = make_repo(tmp_path)
    stager = SourceStager(GitRepository(root), SshExecutor("cluster"), RsyncTransfer("cluster"),
                          strategy="fetch", url=str(root))
    stager.stage(head(root), str(tmp_path / "a"))
    assert head(tmp_path / "a") == head(root)
    with pytest.raises(HpcError, match="not reachable"):
        stager.stage("0" * 40, str(tmp_path / "b"))
