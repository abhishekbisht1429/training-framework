"""The pure compiler: manifest + profile + dialect + repository snapshot +
local facts + clock -> `ExecutionPlan`.

Every offline check (building block B8) happens here; problems are collected
and raised together as one `PlanError`.
"""

from __future__ import annotations

import dataclasses
import itertools
import tomllib
from datetime import datetime, timezone
from typing import Any

import yaml
from omegaconf import OmegaConf

from training_framework.hpc import EXECUTION_PROTOCOL_VERSION
from training_framework.hpc import address as addresses
from training_framework.hpc import data as data_module
from training_framework.hpc.data import LocalFacts
from training_framework.hpc.errors import Problems
from training_framework.hpc.names import hash_bytes, hash_value
from training_framework.hpc.plan import (
    Artifact,
    ExecutionContext,
    ExecutionPlan,
    JobNode,
    Task,
)
from training_framework.hpc.python_rules import (
    interpreter_expectation,
    parse_requires_python,
)
from training_framework.hpc.repository import RepositoryReader
from training_framework.hpc.requirements import (
    Requirements,
    RequirementsError,
    parse_requirements,
)
from training_framework.hpc.schema import Manifest, Profile, parse_manifest
from training_framework.hpc.slurm import SlurmDialect
from training_framework.hpc.topology import check_topology, parse_engine_options

UNSET_VARIABLES = ("WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
SET_VARIABLES = (("PYTHONNOUSERSITE", "1"),)
FRAMEWORK_DISTRIBUTION = "training-framework"


@dataclasses.dataclass(frozen=True)
class Compiled:
    plan: ExecutionPlan
    manifest: Manifest
    suggestions: list[dict]


def _load_yaml(data: bytes, source: str, problems: Problems) -> Any:
    try:
        return yaml.safe_load(data)
    except yaml.YAMLError as error:
        problems.add(f"{source} is not valid YAML ({error})")
        return None


def compile_plan(
        *,
        manifest_path: str,
        repository: RepositoryReader,
        profile: Profile,
        facts: LocalFacts,
        clock: datetime,
        fresh_env: bool = False,
) -> Compiled:
    problems = Problems()

    if not repository.is_clean():
        problems.add(
            "the repository has uncommitted changes or untracked files; "
            "commit or ignore them (batches run an exact commit)"
        )
    head = repository.resolve("HEAD")
    head_bytes = repository.read(head, manifest_path)
    if head_bytes is None:
        problems.add(f"{manifest_path} is not committed")
        problems.raise_if_any()
    raw_manifest = _load_yaml(head_bytes, manifest_path, problems)
    problems.raise_if_any()
    manifest = parse_manifest(raw_manifest, manifest_path)

    sha = repository.resolve(manifest.source_ref)
    if sha != head and repository.read(sha, manifest_path) != head_bytes:
        problems.add(
            f"{manifest_path} differs between HEAD and source.ref "
            f"'{manifest.source_ref}'; check out that ref first"
        )

    dialect = SlurmDialect(profile.slurm)
    for problem in dialect.check_profile():
        problems.add(problem)

    base_config, base_session = _training_config(manifest, repository, sha, problems)
    for path in manifest.code_paths:
        if not repository.exists(sha, path):
            problems.add(f"code.paths: '{path}' does not exist at {sha[:12]}")
    problems.raise_if_any()

    # Environment, Python, code layout ------------------------------------
    managed = profile.env.existing is None
    requires_python, problem = _requires_python(repository, sha)
    notes: list[str] = []
    if problem:
        problems.add(problem)
    if requires_python is None and not problem:
        notes.append("pyproject.toml declares no requires-python; the interpreter version is not checked against it")
    requirements = _requirements(manifest, repository, sha, managed, problems)
    expectation, python_problems = interpreter_expectation(
        profile.env, manifest.python_pin, requires_python, managed,
    )
    for problem in python_problems:
        problems.add(problem)
    framework_expectation = _framework_expectation(manifest, requirements, managed, problems)

    clock = clock.astimezone(timezone.utc)
    batch_id = f"{manifest.name}-{clock:%Y%m%dT%H%M%SZ}-{sha[:12]}"
    batch_dir = f"{profile.roots['batches']}/{batch_id}"
    source_dir = f"{batch_dir}/source"

    environment, setup_context = _environment(
        manifest, profile, requirements, expectation, batch_id, batch_dir, fresh_env,
    )
    timeout, option_problems = parse_engine_options(manifest.engine_options)
    for problem in option_problems:
        problems.add(problem)

    # Tasks ------------------------------------------------------------------
    tasks_raw = _expand(manifest, base_session, dialect, problems)
    problems.raise_if_any()

    sessions = [session for _, _, _, _, session, _ in tasks_raw]
    sources = data_module.parse_sources(manifest.data_sources, problems, profile)
    problems.raise_if_any()

    def code_identity(inputs):
        if not inputs:
            return {"sha": sha}
        identity = {}
        for path in inputs:
            object_id = repository.object_id(sha, path)
            if object_id is None:
                problems.add(f"prepare inputs: '{path}' does not exist at {sha[:12]}")
            identity[path] = object_id
        return identity

    data = data_module.compile_sources(
        sources, sessions, profile, facts, environment.get("key"), code_identity, problems,
    )
    suggestions = data_module.suggest_sources(sessions, sources, facts)
    if suggestions:
        notes.append(
            f"{len(suggestions)} absolute local path(s) in the task configs are "
            "not data sources; `validate --suggest-data` prints entries for them"
        )
    problems.raise_if_any()

    tasks: list[Task] = []
    seen_hashes: dict[str, str] = {}
    for index, (group, experiment, number, array_index, session, variant) in enumerate(tasks_raw):
        for target, path in data.rewrites.get(index, {}).items():
            session = addresses.write(session, target, path)
        name = f"{group.name}/{experiment.name}/{number:03d}"
        identity_hash = hash_value(session)
        if identity_hash in seen_hashes:
            problems.add(
                f"tasks {seen_hashes[identity_hash]} and {name} resolve to the same config"
            )
        seen_hashes[identity_hash] = name
        session = _with_sessions_dir(session, f"{batch_dir}/tasks/{name}/sessions")
        config = {**base_config, "sessions": [session]}
        tasks.append(Task(
            group.name, experiment.name, number, array_index, tuple(variant),
            config, hash_value(config),
        ))
    problems.raise_if_any()

    # Job graph and scripts -----------------------------------------------
    task_context = _context("task", environment, source_dir, manifest, profile)
    prepare_context = _context("prepare", environment, source_dir, manifest, profile)
    plan_file = f"{batch_dir}/plan.json"
    nodes: list[JobNode] = []
    if environment["managed"]:
        nodes.append(JobNode(
            id="setup", kind="setup", location=profile.setup_location,
            depends_on=(), context=setup_context, artifact="environment",
            resources=profile.setup_resources if profile.setup_location == "job" else None,
        ))
    setup_dep = ("setup",) if environment["managed"] else ()
    for node_id, artifact, source, _location in data.prepare_nodes:
        nodes.append(JobNode(
            id=node_id, kind="prepare", location=profile.prepare_location,
            depends_on=setup_dep, context=prepare_context,
            resources=source.resources, artifact=artifact.key,
        ))
    for group in manifest.groups:
        indices = {i for i, raw in enumerate(tasks_raw) if raw[0].name == group.name}
        prepares = tuple(
            node_id for node_id, users in sorted(data.users.items()) if users & indices
        )
        nodes.append(JobNode(
            id=f"group-{group.name}", kind="task_set", location="job",
            depends_on=setup_dep + prepares, context=task_context,
            resources=group.resources, group=group.name,
            task_count=len(indices), max_parallel=group.max_parallel,
            stop_grace_seconds=group.stop_grace_seconds,
        ))

    for node in nodes:
        if node.kind != "task_set" and node.resources is not None:
            where = "profile setup_resources" if node.kind == "setup" else f"prepare node {node.id}"
            for problem in dialect.check_resources(node.resources, where):
                problems.add(problem)

    rendered = []
    for node in nodes:
        if node.kind == "task_set":
            command = [
                node.context.interpreter, "-m", "training_framework.hpc.task",
                "--config", "{config}", "--record", "{record}",
                "--plan", plan_file, *manifest.engine_options,
            ]
            lead = dialect.signal_lead(timeout, node.stop_grace_seconds)
            if lead >= node.resources.walltime_seconds:
                problems.add(
                    f"group '{node.group}': the shutdown signal lead ({lead}s "
                    "= process_timeout_on_join + stop_grace) is not shorter "
                    f"than the walltime ({node.resources.walltime})"
                )
        elif node.kind == "prepare":
            command = [
                node.context.interpreter, "-m", "training_framework.hpc.task",
                "--prepare", node.id, "--plan", plan_file,
            ]
            lead = None
        else:
            command = [node.context.interpreter, f"{batch_dir}/bootstrap.py", "--plan", plan_file]
            lead = None
        script = dialect.render(node, batch_id, batch_dir, command, lead)
        rendered.append(dataclasses.replace(node, script=script))
    problems.raise_if_any()

    components_packages = sorted({
        task.config["sessions"][0]["session_config"]["components_package"]
        for task in tasks
    })
    artifacts = list(data.artifacts)
    if environment["managed"]:
        artifacts.insert(0, Artifact(
            "environment", "environment", environment["key"],
            environment["location"], "key-derived",
            {"backend": profile.env.backend, "requirements": requirements.key()},
        ))
    plan = ExecutionPlan(
        protocol_version=EXECUTION_PROTOCOL_VERSION,
        batch_id=batch_id,
        batch_dir=batch_dir,
        sha=sha,
        manifest_hash=hash_bytes(head_bytes),
        profile=profile.name,
        scheduler=dialect.name,
        code_mode=manifest.code_framework,
        tasks=tuple(tasks),
        artifacts=tuple(artifacts),
        nodes=tuple(rendered),
        environment=environment,
        preflight={
            "protocol_version": EXECUTION_PROTOCOL_VERSION,
            "interpreter": expectation.as_json(),
            "framework": framework_expectation,
            "components_packages": components_packages,
            "cluster_paths": sorted(data.cluster_paths),
            "requires_python": str(requires_python) if requires_python else None,
        },
        engine_options=manifest.engine_options,
        notes=tuple(notes),
    )
    return Compiled(plan, manifest, suggestions)


# Helpers -----------------------------------------------------------------------


def _training_config(manifest, repository, sha, problems):
    raw = repository.read(sha, manifest.training_config)
    if raw is None:
        problems.add(f"training.config '{manifest.training_config}' does not exist at {sha[:12]}")
        return {}, {}
    try:
        config = OmegaConf.to_container(OmegaConf.create(raw.decode()))
    except Exception as error:  # OmegaConf raises several error types
        problems.add(f"training.config '{manifest.training_config}' cannot be read ({error})")
        return {}, {}
    sessions = config.get("sessions") if isinstance(config, dict) else None
    if not isinstance(sessions, list) or not sessions:
        problems.add(f"training.config '{manifest.training_config}' has no 'sessions' list")
        return {}, {}
    if manifest.training_session >= len(sessions):
        problems.add(
            f"training.session {manifest.training_session} is out of range "
            f"({len(sessions)} session(s))"
        )
        return {}, {}
    session = sessions[manifest.training_session]
    if not isinstance(session, dict):
        problems.add("the selected session must be a mapping")
        return {}, {}
    session_config = session.get("session_config")
    if not isinstance(session_config, dict) or not isinstance(
            session_config.get("components_package"), str):
        problems.add(
            "the selected session needs session_config.components_package "
            "(the package that registers its components)"
        )
    return config, session


def _requires_python(repository, sha):
    raw = repository.read(sha, "pyproject.toml")
    if raw is None:
        return None, None
    try:
        project = tomllib.loads(raw.decode()).get("project", {})
    except tomllib.TOMLDecodeError as error:
        return None, f"pyproject.toml cannot be read ({error})"
    return parse_requires_python(project.get("requires-python"))


def _requirements(manifest, repository, sha, managed, problems) -> Requirements | None:
    if manifest.requirements is None:
        if managed:
            problems.add(
                "environment.requirements is required for a managed "
                "environment (or set env.existing in the profile)"
            )
        return None
    raw = repository.read(sha, manifest.requirements)
    if raw is None:
        problems.add(f"environment.requirements '{manifest.requirements}' does not exist at {sha[:12]}")
        return None
    try:
        return parse_requirements(raw.decode(), manifest.requirements)
    except RequirementsError as error:
        for problem in error.problems:
            problems.add(problem)
        return None


def _framework_expectation(manifest, requirements, managed, problems) -> dict:
    pins = requirements.pins_named(FRAMEWORK_DISTRIBUTION) if requirements else []
    if manifest.code_framework == "checkout":
        if pins:
            problems.add(
                "code.framework is 'checkout', so the requirements must not "
                f"contain {FRAMEWORK_DISTRIBUTION} (it comes from the checkout)"
            )
        return {"location": "checkout", "distribution": None}
    if requirements is not None and len(pins) != 1:
        problems.add(
            "code.framework is 'dependency', so the requirements must pin "
            f"{FRAMEWORK_DISTRIBUTION} exactly once"
        )
        return {"location": "environment", "distribution": None}
    version = pins[0].version if pins and managed else None
    return {"location": "environment", "distribution": version}


def _environment(manifest, profile, requirements, expectation, batch_id, batch_dir, fresh_env):
    env = profile.env
    hooks_setup = tuple(profile.hooks["common"]) + tuple(profile.hooks["setup"])
    if env.existing is not None:
        environment = {
            "managed": False, "key": None, "location": env.existing,
            "interpreter": f"{env.existing}/bin/python",
        }
        return environment, None
    key_parts = {
        "requirements": requirements.normalized() if requirements else None,
        "backend": env.backend,
        "interpreter": expectation.key_part(),
        "hooks": list(hooks_setup),
    }
    if env.backend == "conda":
        key_parts["conda"] = {"packages": list(env.conda_packages), "python": manifest.python_pin}
    key = hash_value(key_parts)
    name = f"{key}-{batch_id}" if fresh_env else key
    location = f"{profile.roots['envs']}/{name}"
    environment = {
        "managed": True, "key": key, "location": location,
        "interpreter": f"{location}/bin/python", "fresh": fresh_env,
    }
    if env.bootstrap_python:
        bootstrap = env.bootstrap_python
    elif env.python_source == "site":
        bootstrap = "python3"
    elif env.python_source == "uv":
        bootstrap = f"$(uv python find {manifest.python_pin})"
    else:
        bootstrap = "$(conda run -n base which python)"
    setup_context = ExecutionContext(
        phase="setup", interpreter=bootstrap, environment=location,
        working_directory=batch_dir, pythonpath=(), set_variables=SET_VARIABLES,
        unset_variables=UNSET_VARIABLES, hooks=hooks_setup,
    )
    return environment, setup_context


def _context(phase, environment, source_dir, manifest, profile) -> ExecutionContext:
    return ExecutionContext(
        phase=phase,
        interpreter=environment["interpreter"],
        environment=environment["location"],
        working_directory=source_dir,
        pythonpath=tuple(
            source_dir if path == "." else f"{source_dir}/{path}"
            for path in manifest.code_paths
        ),
        set_variables=SET_VARIABLES,
        unset_variables=UNSET_VARIABLES,
        hooks=tuple(profile.hooks["common"]) + tuple(profile.hooks[phase]),
    )


def _with_sessions_dir(session: dict, path: str) -> dict:
    session_config = dict(session.get("session_config") or {})
    session_config["sessions_dir"] = path
    return {**session, "session_config": session_config}


def _parse_entries(entries, field, where, base, problems):
    parsed = []
    seen = set()
    for index, entry in enumerate(entries):
        target, target_problems = addresses.parse_address(entry, f"{where}[{index}]")
        for problem in target_problems:
            problems.add(problem)
        if target is None:
            continue
        if problem := addresses.check_target(target, base):
            problems.add(f"{where}[{index}]: {problem}")
            continue
        if target in seen:
            problems.add(f"{where}[{index}]: {target.describe()} is addressed twice")
            continue
        seen.add(target)
        parsed.append((target, entry[field]))
    return parsed


def _expand(manifest, base_session, dialect, problems):
    """(group, experiment, number, array index, session, variant) per task."""
    tasks = []
    for group in manifest.groups:
        where = f"groups.{group.name}"
        for problem in dialect.check_resources(group.resources, f"{where}.resources"):
            problems.add(problem)
        array_index = 0
        for experiment in group.experiments:
            e_where = f"{where}.experiments[{experiment.name}]"
            overrides = _parse_entries(experiment.overrides, "value", f"{e_where}.overrides", base_session, problems)
            matrix = _parse_entries(experiment.matrix, "values", f"{e_where}.matrix", base_session, problems)
            session = base_session
            for target, value in overrides:
                try:
                    session = addresses.write(session, target, value)
                except ValueError as error:
                    problems.add(f"{e_where}: {error}")
            combinations = itertools.product(*(values for _, values in matrix)) if matrix else [()]
            for number, combination in enumerate(combinations):
                task_session = session
                variant = []
                for (target, _), value in zip(matrix, combination):
                    try:
                        task_session = addresses.write(task_session, target, value)
                    except ValueError as error:
                        problems.add(f"{e_where}: {error}")
                    variant.append({"address": target.as_json(), "value": value})
                for problem in check_topology(task_session, group.resources):
                    problems.add(f"{group.name}/{experiment.name}/{number:03d}: {problem}")
                tasks.append((group, experiment, number, array_index, task_session, variant))
                array_index += 1
        if array_index > dialect.max_task_set_size:
            problems.add(
                f"group '{group.name}' has {array_index} tasks, more than the "
                f"maximum task-set size {dialect.max_task_set_size}; split it"
            )
    return tasks

