"""Environment launcher, Slurm option policy, scheduler composition, and
bundle cleanup (from the branch review)."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import pytest

from training_framework.hpc.compile.plan import ExecutionContext, JobNode
from training_framework.hpc.config.schema import Resources, parse_profiles
from training_framework.hpc.errors import PlanError
from training_framework.hpc.ports.local import LocalExecutor
from training_framework.hpc.scheduler import SCHEDULERS, SchedulerBundle
from training_framework.hpc.scheduler.slurm import SlurmDialect, SlurmSettings
from tests.hpc.batch_repo import MANIFEST, PROFILE, compile_repo, make_repo, profile


# Environment launcher ----------------------------------------------------------


def test_every_context_puts_the_environment_first_on_path(tmp_path):
    plan = compile_repo(make_repo(tmp_path)).plan
    env_bin = f"{plan.environment['location']}/bin"
    for node_id in ("group-probes",):
        context = plan.node(node_id).context
        assert context.path_prepend == (env_bin,)
        assert f"export PATH={env_bin}:\"$PATH\"" in plan.node(node_id).script
    assert plan.environment["preflight_script"].count(f"export PATH={env_bin}") == 1


def test_a_prepare_command_finds_the_environments_python(tmp_path):
    env = tmp_path / "env" / "bin"
    env.mkdir(parents=True)
    (env / "python").write_text("#!/bin/sh\necho environment-python\n")
    (env / "python").chmod(0o755)
    context = ExecutionContext("prepare", str(env / "python"), str(tmp_path / "env"),
                               str(tmp_path), (), (), (), (), path_prepend=(str(env),))
    assert LocalExecutor().run(["python"], context).stdout.strip() == "environment-python"


def test_conda_run_wraps_every_command_in_the_launcher(tmp_path):
    data = profile()
    data["profiles"]["cluster"]["env"] = {"backend": "conda", "python_source": "conda",
                                         "conda_run": True}
    root = make_repo(tmp_path, manifest=copy.deepcopy(MANIFEST) | {
        "environment": {"requirements": "requirements/hpc.txt", "python": "3.12"}})
    plan = compile_repo(root, profile_data=data).plan
    location = plan.environment["location"]
    launcher = ("conda", "run", "--no-capture-output", "-p", location)
    assert plan.node("group-probes").context.launcher == launcher
    assert f"conda run --no-capture-output -p {location} {location}/bin/python -m" in \
        plan.node("group-probes").script
    assert f"exec conda run --no-capture-output -p {location} \"$@\"" in plan.environment["preflight_script"]
    assert plan.node("setup").context.launcher == ()     # the environment does not exist yet


def test_conda_run_with_a_venv_is_refused(tmp_path):
    data = profile()
    data["profiles"]["cluster"]["env"]["conda_run"] = True
    with pytest.raises(PlanError, match="conda_run needs a conda environment"):
        compile_repo(make_repo(tmp_path), profile_data=data)


def test_the_launcher_is_applied_by_the_executor(tmp_path):
    context = ExecutionContext("task", "x", None, str(tmp_path), (), (), (), (),
                               launcher=("env", "LAUNCHED=yes"))
    result = LocalExecutor().run(["sh", "-c", "echo $LAUNCHED"], context)
    assert result.stdout.strip() == "yes"


# Slurm option policy -------------------------------------------------------------


def resources(options):
    return Resources("01:00:00", 3600, 1, 4, "8G", options)


SETTINGS = SlurmSettings({"account": "lab", "qos": "normal"}, {"gpu": "g", "cpu": "c"}, 10)


def test_qos_is_site_policy_not_a_group_option():
    dialect = SlurmDialect(SETTINGS)
    assert dialect.check_profile() == []
    problems = dialect.check_resources(resources({"slurm": {"qos": "high"}}), "g")
    assert any("qos is set by the HPC layer" in problem for problem in problems)


@pytest.mark.parametrize("value", ["a100\n#SBATCH --account=other", "x\ry", "x\x00"])
def test_control_characters_cannot_inject_directives(value):
    dialect = SlurmDialect(SETTINGS)
    assert any("control character" in p
               for p in dialect.check_resources(resources({"slurm": {"constraint": value}}), "g"))
    bad_profile = SlurmDialect(SlurmSettings({"comment": value}, {"cpu": "c"}, 10))
    assert any("control character" in p for p in bad_profile.check_profile())
    node = JobNode(id="setup", kind="setup", location="job", depends_on=(),
                   context=ExecutionContext("setup", "python3", None, "/b", (), (), (), ()),
                   resources=resources({"slurm": {"constraint": value}}))
    with pytest.raises(ValueError, match="control characters"):
        dialect.render(node, "batch", "/b", ["python3"], None)


# Scheduler composition ------------------------------------------------------------


class FakeDialect:
    name = "fake"

    def __init__(self, settings):
        self.settings = settings
        self.max_task_set_size = settings["max"]

    def check_profile(self):
        return []

    def check_resources(self, resources, where):
        return []

    def signal_lead(self, timeout, grace):
        return int(timeout) + grace

    def render(self, node, batch_id, batch_dir, command, lead):
        return f"# fake script for {batch_id}.{node.id}: {' '.join(command)}\n"


def test_a_scheduler_is_added_at_one_composition_point(tmp_path, monkeypatch):
    bundle = SchedulerBundle(
        name="fake",
        parse_settings=lambda raw, where, problems, reader: {"max": raw.get("max", 5)},
        dialect=FakeDialect, submitter=lambda executor: None, owner_probe=lambda executor: None,
    )
    monkeypatch.setitem(SCHEDULERS, "fake", bundle)
    data = copy.deepcopy(PROFILE)
    cluster = data["profiles"]["cluster"]
    cluster["scheduler"] = "fake"
    cluster["fake"] = {"max": 50}
    del cluster["slurm"]
    plan = compile_repo(make_repo(tmp_path), profile_data=data).plan
    assert plan.scheduler == "fake"
    assert plan.node("group-probes").script.startswith(f"# fake script for {plan.batch_id}.group-probes")
    monkeypatch.delitem(SCHEDULERS, "fake")


def test_settings_for_an_unselected_scheduler_are_refused(monkeypatch):
    monkeypatch.setitem(SCHEDULERS, "fake", SchedulerBundle(
        "fake", lambda *a: {}, FakeDialect, lambda e: None, lambda e: None))
    data = copy.deepcopy(PROFILE)
    data["profiles"]["cluster"]["fake"] = {}
    with pytest.raises(PlanError, match="settings for scheduler 'fake'"):
        parse_profiles(data, None, "p")
    monkeypatch.delitem(SCHEDULERS, "fake")


# Bundle cleanup ---------------------------------------------------------------


def test_staging_removes_the_bundle_after_checkout(tmp_path):
    from training_framework.hpc.ports.repository import GitRepository
    from training_framework.hpc.services.staging import SourceStager

    root = make_repo(tmp_path)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                         text=True).stdout.strip()
    target = tmp_path / "cluster" / "source"

    class LocalTransfer:
        def sync(self, local, remote):
            os.makedirs(os.path.dirname(remote), exist_ok=True)
            Path(remote).write_bytes(Path(local).read_bytes())
            return 0

    SourceStager(GitRepository(root), LocalExecutor(), LocalTransfer()).stage(sha, str(target))
    assert (target / "batch.yaml").exists()
    assert not Path(f"{target}.bundle").exists()
