import pytest

from training_framework.hpc.schema import Resources
from training_framework.hpc.topology import check_topology, parse_engine_options


def resources(gpus=0, cpus=8):
    return Resources("01:00:00", 3600, gpus, cpus, "16G", {})


def session(ddp=None):
    data = {"session_config": {}}
    if ddp is not None:
        data["resources"] = {"ddp": ddp}
    return data


def test_nccl_needs_world_size_equal_to_gpus():
    assert not check_topology(session({"backend": "nccl", "world_size": 2}), resources(gpus=2))
    assert check_topology(session({"backend": "nccl", "world_size": 2}), resources(gpus=1))
    assert check_topology(session({"backend": "nccl", "world_size": 1}), resources(gpus=0))


def test_gloo_runs_on_cpus():
    assert not check_topology(session({"backend": "gloo", "world_size": 4}), resources(cpus=8))
    problems = check_topology(session({"backend": "gloo", "world_size": 16}), resources(cpus=8))
    assert "only 8 CPUs" in problems[0]


def test_gpus_with_an_implicit_backend_are_refused():
    problems = check_topology(session({"world_size": 1}), resources(gpus=1))
    assert "ddp.backend is not set" in problems[0]


def test_world_size_must_be_explicit():
    problems = check_topology(session({"backend": "nccl"}), resources(gpus=1))
    assert "set explicitly" in problems[0]


def test_without_ddp_at_most_one_gpu():
    assert not check_topology(session(), resources(gpus=1))
    assert "configures no 'ddp'" in check_topology(session(), resources(gpus=2))[0]


def test_unsupported_backends_are_refused():
    assert "not supported" in check_topology(
        session({"backend": "mpi", "world_size": 1}), resources())[0]


def test_engine_options_accept_known_options_and_read_the_join_timeout():
    timeout, problems = parse_engine_options(
        ("--heartbeat-timeout", "60", "--process_timeout_on_join=45"),
    )
    assert not problems and timeout == 45


@pytest.mark.parametrize("option", [
    "--config", "--resume-session", "--extend-session", "--debug", "--override",
    "--heartbeat",  # an argparse abbreviation
])
def test_engine_options_that_change_what_runs_are_refused(option):
    _, problems = parse_engine_options((option, "x"))
    assert "is not allowed" in problems[0]


def test_engine_option_values_must_be_numbers():
    _, problems = parse_engine_options(("--heartbeat-timeout", "soon"))
    assert "finite non-negative" in problems[0]
