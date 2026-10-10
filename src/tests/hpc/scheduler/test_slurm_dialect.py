from training_framework.hpc.compile.plan import ExecutionContext, JobNode
from training_framework.hpc.config.schema import Resources, SlurmSettings
from training_framework.hpc.scheduler.slurm import SlurmDialect

SETTINGS = SlurmSettings(
    defaults={"account": "lab", "qos": "normal"},
    partitions={"gpu": "gpu-part", "cpu": "cpu-part"},
    max_array_size=500,
)
CONTEXT = ExecutionContext(
    phase="task", interpreter="/env/bin/python", environment="/env",
    working_directory="/b/source", pythonpath=("/b/source",),
    set_variables=(("PYTHONNOUSERSITE", "1"),),
    unset_variables=("WORLD_SIZE",), hooks=("module load cuda",),
)


def resources(gpus=1, options=None):
    return Resources("02:00:00", 7200, gpus, 8, "64G", options or {})


def test_portable_resources_and_profile_placement_render():
    node = JobNode(id="group-g", kind="task_set", location="job", depends_on=(),
                   context=CONTEXT, resources=resources(options={"slurm": {"constraint": "a100"}}),
                   group="g", task_count=6, max_parallel=2)
    script = SlurmDialect(SETTINGS).render(node, "batch-1", "/b", ["/env/bin/python", "-m", "x", "--config", "{config}"], 150)
    for line in (
            "#SBATCH --job-name=batch-1.group-g", "#SBATCH --time=02:00:00",
            "#SBATCH --cpus-per-task=8", "#SBATCH --mem=64G", "#SBATCH --gres=gpu:1",
            "#SBATCH --partition=gpu-part", "#SBATCH --account=lab", "#SBATCH --qos=normal",
            "#SBATCH --constraint=a100", "#SBATCH --array=0-5%2",
            "#SBATCH --signal=B:USR1@150", "#SBATCH --chdir=/b/source",
            "module load cuda", "unset WORLD_SIZE", "export PYTHONNOUSERSITE=1",
            "export PYTHONPATH=/b/source", "trap on_signal USR1",
    ):
        assert line in script.splitlines(), line
    assert '--config "$config" &' in script


def test_cpu_groups_use_the_cpu_partition_and_no_gres():
    node = JobNode(id="setup", kind="setup", location="job", depends_on=(),
                   context=CONTEXT, resources=resources(gpus=0))
    script = SlurmDialect(SETTINGS).render(node, "batch-1", "/b", ["python3", "/b/bootstrap.py"], None)
    assert "#SBATCH --partition=cpu-part" in script
    assert "gres" not in script and "--signal" not in script
    assert script.rstrip().endswith("exec python3 /b/bootstrap.py")


def test_owned_directives_and_foreign_schedulers_are_refused():
    dialect = SlurmDialect(SETTINGS)
    problems = dialect.check_resources(
        resources(options={"slurm": {"partition": "x", "array": "1"}, "pbs": {"q": "x"}}), "g",
    )
    assert any("partition is set by the HPC layer" in p for p in problems)
    assert any("array is set by the HPC layer" in p for p in problems)
    assert any("other than the selected 'slurm'" in p for p in problems)


def test_missing_partitions_are_reported():
    dialect = SlurmDialect(SlurmSettings({}, {}, 10))
    assert any("no slurm.partitions.gpu" in p for p in dialect.check_resources(resources(), "g"))


def test_owned_directives_are_refused_in_profile_defaults():
    dialect = SlurmDialect(SlurmSettings({"time": "1:00:00", "account": "lab"}, {}, 10))
    assert dialect.check_profile() == ["profile slurm.defaults: 'time' is not allowed"]


def test_the_signal_lead_covers_the_join_timeout_and_grace():
    assert SlurmDialect(SETTINGS).signal_lead(30.5, 120) == 151
