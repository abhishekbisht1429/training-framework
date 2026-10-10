"""End to end on this machine: compile a real batch, stage it, preflight it
in a real environment, and run its job scripts (task wrapper, the task entry
point, the engine) through a local stand-in for the scheduler."""

from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import pytest
import torch
import yaml

import training_framework
from training_framework.hpc.ports.local import LocalExecutor, LocalFilesystem, LocalSourceStager
from training_framework.hpc.services.submit import submit
from tests.hpc.batch_repo import PROFILE, compile_repo, make_repo
from tests.hpc.fakes import LocalScheduler

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="the end-to-end task runs a one-rank gloo DDP session",
)

COMPONENTS = Path(__file__).parents[1] / "integration" / "integration_training_components.py"
FRAMEWORK_SRC = Path(training_framework.__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def environment(tmp_path_factory):
    """A user-owned environment that sees this checkout's framework: a venv
    over the current interpreter's packages (torch), with `src` on its path
    and a training-framework distribution record of its own."""
    root = tmp_path_factory.mktemp("env") / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                    "--without-pip", str(root)], check=True)
    purelib = Path(sysconfig.get_path("purelib", vars={"base": str(root), "platbase": str(root)}))
    (purelib / "training_framework_src.pth").write_text(f"{FRAMEWORK_SRC}\n")
    dist_info = purelib / "training_framework-0.5.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: training-framework\nVersion: 0.5.0\n")
    return root


def session(progress_dir=None, max_iterations=3):
    data = {
        "session_config": {
            "rng_seed": 17, "sessions_dir": "./runs", "max_iterations": max_iterations,
            "device": "cpu", "components_package": "my_project.components",
            "show_execution_graph": False,
        },
        "role_bindings": {"model": "integration_ddp_model", "dataset": "integration_dataset"},
        "integration_ddp_model": {"initial_weight": 0.0},
        "integration_dataset": {"dataset_size": 4},
        "ddp": {
            "world_size": 1, "backend": "gloo", "master_addr": "127.0.0.1",
            "parallel_components": ["model", "dataset", "data_manager", "integration_data",
                                    "integration_train", "integration_loss", "optimizer",
                                    "integration_results"],
        },
        "data_manager": {"batch_size": 1, "num_workers": 0, "pin_memory": False},
        "integration_data": {},
        "integration_train": {},
        "integration_loss": {},
        "optimizer": {"optimizer": {"name": "AdamW", "kwargs": {"lr": 0.1, "weight_decay": 0.0}}},
        "integration_results": {"output_dir": "results"},
    }
    if progress_dir is not None:
        data["integration_results"]["progress_dir"] = str(progress_dir)
    return data


def batch(tmp_path, environment, config_session, seeds):
    manifest = {
        "version": 1, "name": "e2e",
        "code": {"framework": "dependency", "paths": ["."]},
        "training": {"config": "configs/train.yaml",
                     "engine_options": ["--process_timeout_on_join", "30"]},
        "groups": {"cpu": {
            "resources": {"walltime": "00:30:00", "cpus": 2, "memory": "4G"},
            "experiments": [{"name": "seeds", "matrix": [
                {"path": "session_config.rng_seed", "values": seeds}]}],
        }},
    }
    root = make_repo(
        tmp_path, manifest=manifest, config={"sessions": [config_session]}, requirements=None,
        extra_files={"my_project/components.py": COMPONENTS.read_text()},
    )
    profile = copy.deepcopy(PROFILE)
    cluster = profile["profiles"]["cluster"]
    cluster["roots"] = {key: str(tmp_path / "cluster" / key) for key in ("batches", "envs", "data")}
    cluster["env"] = {"existing": str(environment)}
    cluster["hooks"] = {}
    return root, compile_repo(root, profile_data=profile)


def test_a_batch_runs_end_to_end(tmp_path, environment):
    root, compiled = batch(tmp_path, environment, session(), [1, 2])
    scheduler = LocalScheduler()
    receipt = submit(
        compiled.plan, compiled.manifest.raw, stager=LocalSourceStager(root),
        filesystem=LocalFilesystem(), executor=LocalExecutor(), submitter=scheduler,
        receipts_dir=str(tmp_path / "receipts"),
    )
    assert receipt.status == "submitted"
    assert scheduler.results == {"group-cpu": [0, 0]}, scheduler.outputs
    plan = compiled.plan
    for task in plan.tasks:
        record = json.loads(Path(plan.batch_dir, task.directory, "record.json").read_text())
        assert record["exit_status"] == 0 and record["interrupted"] is False
        assert record["sha"] == plan.sha and record["batch_id"] == plan.batch_id
        assert record["framework_distribution"] == "0.5.0"
        (run_directory,) = record["run_directories"]
        assert run_directory.startswith(f"{plan.batch_dir}/{task.directory}/sessions/session_")


def test_the_advance_signal_stops_a_task_cooperatively(tmp_path, environment):
    progress = tmp_path / "progress"
    root, compiled = batch(tmp_path, environment, session(progress, max_iterations=1_000_000), [1])
    plan = compiled.plan
    submit(compiled.plan, compiled.manifest.raw, stager=LocalSourceStager(root),
           filesystem=LocalFilesystem(), executor=LocalExecutor(), submitter=LocalScheduler(),
           receipts_dir=str(tmp_path / "receipts"), confirm=lambda plan: False)
    script = f"{plan.batch_dir}/groups/cpu/group-cpu.sbatch"
    job = subprocess.Popen(["bash", script], env={**os.environ, "SLURM_ARRAY_TASK_ID": "0"},
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 120
    while not (progress / "rank_0.progress").exists():
        assert job.poll() is None, job.stdout.read()
        assert time.monotonic() < deadline, "the task made no progress"
        time.sleep(0.1)
    job.send_signal(signal.SIGUSR1)       # what Slurm's --signal=B:USR1 delivers
    output, _ = job.communicate(timeout=120)
    assert job.returncode == 75, output
    record = json.loads(Path(plan.batch_dir, plan.tasks[0].directory, "record.json").read_text())
    assert record["interrupted"] is True and record["exit_status"] == 75
