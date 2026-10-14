import copy
import json
import re
import subprocess
from datetime import timedelta

import pytest
import yaml

from training_framework.hpc import EXECUTION_PROTOCOL_VERSION
from training_framework.hpc.errors import PlanError
from tests.hpc.batch_repo import (
    CLOCK,
    MANIFEST,
    PROFILE,
    TRAINING_CONFIG,
    FakeFacts,
    compile_repo,
    make_repo,
    manifest,
    profile,
)


def problems_of(root, **kwargs):
    with pytest.raises(PlanError) as error:
        compile_repo(root, **kwargs)
    return "\n".join(error.value.problems)


def commit(root, relative, content, message="change"):
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / relative).write_text(content)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit",
                    "-qm", message], cwd=root, check=True)


def with_data(sources, **changes):
    data = manifest(**changes)
    data["data"] = {"sources": sources}
    return data


# Determinism and identity --------------------------------------------------


def test_the_compiler_is_deterministic(tmp_path):
    root = make_repo(tmp_path)
    assert compile_repo(root).plan.canonical() == compile_repo(root).plan.canonical()


def test_plans_from_different_clocks_differ_only_in_batch_id(tmp_path):
    root = make_repo(tmp_path)
    first = compile_repo(root).plan
    second = compile_repo(root, clock=CLOCK + timedelta(minutes=5)).plan
    assert first.batch_id != second.batch_id

    def normalized(plan):
        # Config hashes cover each task's sessions_dir, which contains the
        # batch ID; compare the configs themselves instead.
        data = json.loads(plan.canonical().replace(plan.batch_id, "ID"))
        for task in data["tasks"]:
            task.pop("config_hash")
        return data, [json.dumps(t.config).replace(plan.batch_id, "ID") for t in plan.tasks]

    assert normalized(first) == normalized(second)


def test_the_plan_carries_identity_and_the_protocol_version(tmp_path):
    root = make_repo(tmp_path)
    plan = compile_repo(root).plan
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()
    assert plan.sha == head
    assert plan.batch_id == f"sweep-20261009T120000Z-{head[:12]}"
    assert plan.protocol_version == EXECUTION_PROTOCOL_VERSION
    assert plan.preflight["protocol_version"] == EXECUTION_PROTOCOL_VERSION


def test_a_dirty_repository_is_refused(tmp_path):
    root = make_repo(tmp_path)
    (root / "untracked.txt").write_text("x")
    assert "uncommitted changes" in problems_of(root)


def test_an_uncommitted_manifest_is_refused(tmp_path):
    root = make_repo(tmp_path)
    with pytest.raises(PlanError, match="not committed"):
        from training_framework.hpc.compile.compiler import compile_plan
        from training_framework.hpc.ports.repository import GitRepository
        from training_framework.hpc.config.schema import parse_profiles
        compile_plan(manifest_path="missing.yaml", repository=GitRepository(root),
                     profile=parse_profiles(PROFILE, None, "p"), facts=FakeFacts(root),
                     clock=CLOCK)


# Expansion -------------------------------------------------------------------


def two_experiments():
    data = manifest()
    group = data["groups"]["probes"]
    group["experiments"].append({
        "name": "unfrozen",
        "matrix": [{"path": "session_config.rng_seed", "values": [7, 8, 9]}],
    })
    return data


def test_tasks_are_numbered_in_product_order_and_indexed_per_group(tmp_path):
    plan = compile_repo(make_repo(tmp_path, manifest=two_experiments())).plan
    assert [task.name for task in plan.tasks] == [
        "probes/frozen/000", "probes/frozen/001", "probes/frozen/002",
        "probes/frozen/003", "probes/unfrozen/000", "probes/unfrozen/001",
        "probes/unfrozen/002",
    ]
    assert [task.array_index for task in plan.tasks] == list(range(7))
    first = plan.tasks[1].config["sessions"][0]
    assert first["session_config"]["rng_seed"] == 42
    assert first["resources"]["optimizer"]["optimizer"]["kwargs"]["lr"] == 0.001
    assert first["fine_tuned_model"]["frozen"] == ["module.*"]
    assert plan.tasks[1].variant[1] == {
        "address": {"path": ["session_config", "rng_seed"]}, "value": 42,
    }


def test_the_matrix_value_wins_over_an_override(tmp_path):
    data = manifest()
    experiment = data["groups"]["probes"]["experiments"][0]
    experiment["overrides"].append({"path": "session_config.rng_seed", "value": 1})
    plan = compile_repo(make_repo(tmp_path, manifest=data)).plan
    assert [t.config["sessions"][0]["session_config"]["rng_seed"] for t in plan.tasks] == [41, 42, 41, 42]


def test_each_task_gets_its_own_sessions_dir_and_keeps_the_rest_of_the_file(tmp_path):
    config = copy.deepcopy(TRAINING_CONFIG)
    config["shared"] = {"anchor": 1}
    plan = compile_repo(make_repo(tmp_path, config=config)).plan
    for task in plan.tasks:
        assert task.config["shared"] == {"anchor": 1}
        assert task.config["sessions"][0]["session_config"]["sessions_dir"] == \
            f"{plan.batch_dir}/tasks/{task.name}/sessions"


def test_identical_tasks_are_refused(tmp_path):
    data = manifest()
    group = data["groups"]["probes"]
    group["experiments"] = [
        {"name": "a", "matrix": [{"path": "session_config.rng_seed", "values": [1]}]},
        {"name": "b", "matrix": [{"path": "session_config.rng_seed", "values": [1]}]},
    ]
    assert "probes/a/000 and probes/b/000 resolve to the same config" in problems_of(
        make_repo(tmp_path, manifest=data))


def test_an_address_twice_in_the_matrix_is_refused(tmp_path):
    data = manifest()
    matrix = data["groups"]["probes"]["experiments"][0]["matrix"]
    matrix.append(copy.deepcopy(matrix[0]))
    assert "addressed twice" in problems_of(make_repo(tmp_path, manifest=data))


def test_targets_are_checked_against_the_session(tmp_path):
    data = manifest()
    data["groups"]["probes"]["experiments"][0]["overrides"] = [
        {"component": "nope", "path": "x", "value": 1},
        {"path": "session_config.sessions_dir", "value": "/elsewhere"},
    ]
    text = problems_of(make_repo(tmp_path, manifest=data))
    assert "'nope' is not listed" in text and "set by the HPC layer" in text


def test_a_group_larger_than_the_task_set_size_is_refused(tmp_path):
    data = profile()
    data["profiles"]["cluster"]["slurm"]["max_array_size"] = 3
    assert "more than the maximum task-set size 3" in problems_of(
        make_repo(tmp_path), profile_data=data)


def test_topology_problems_name_the_task(tmp_path):
    config = copy.deepcopy(TRAINING_CONFIG)
    config["sessions"][0]["resources"]["ddp"] = {"backend": "nccl", "world_size": 2}
    assert "probes/frozen/000: ddp backend 'nccl'" in problems_of(make_repo(tmp_path, config=config))


def test_the_signal_lead_must_fit_in_the_walltime(tmp_path):
    data = manifest()
    data["groups"]["probes"]["resources"]["walltime"] = "00:02:00"
    assert "shutdown signal lead (150s" in problems_of(make_repo(tmp_path, manifest=data))


def test_refused_engine_options_are_reported(tmp_path):
    data = manifest()
    data["training"]["engine_options"] = ["--debug"]
    assert "'--debug' is not allowed" in problems_of(make_repo(tmp_path, manifest=data))


def test_a_session_without_a_components_package_is_refused(tmp_path):
    config = copy.deepcopy(TRAINING_CONFIG)
    del config["sessions"][0]["session_config"]["components_package"]
    assert "components_package" in problems_of(make_repo(tmp_path, config=config))


# Environment and code layout --------------------------------------------------


def test_the_environment_key_ignores_the_commit_but_follows_requirements(tmp_path):
    root = make_repo(tmp_path)
    first = compile_repo(root).plan
    commit(root, "notes.txt", "unrelated")
    second = compile_repo(root).plan
    assert first.sha != second.sha
    assert first.environment["key"] == second.environment["key"]
    commit(root, "requirements/hpc.txt", "torch==2.13.0\ntraining-framework==0.5.1\n")
    assert compile_repo(root).plan.environment["key"] != first.environment["key"]


def test_declared_interpreter_identities_change_the_environment_key(tmp_path):
    root = make_repo(tmp_path)
    other = profile()
    other["profiles"]["cluster"]["env"]["site_python_identity"]["version"] = "3.12.5"
    assert compile_repo(root).plan.environment["key"] != \
        compile_repo(root, profile_data=other).plan.environment["key"]


def test_a_missing_site_identity_is_a_compiler_error(tmp_path):
    data = profile()
    del data["profiles"]["cluster"]["env"]["site_python_identity"]
    assert "site_python_identity is required" in problems_of(make_repo(tmp_path), profile_data=data)


def test_a_fresh_environment_gets_its_own_location(tmp_path):
    root = make_repo(tmp_path)
    shared = compile_repo(root).plan.environment
    fresh = compile_repo(root, fresh_env=True).plan.environment
    assert fresh["key"] == shared["key"] and fresh["location"] != shared["location"]


def test_an_existing_environment_needs_no_requirements_or_setup(tmp_path):
    data = profile()
    data["profiles"]["cluster"]["env"] = {"existing": "/home/me/env"}
    plan = compile_repo(make_repo(tmp_path, requirements=None,
                                  manifest=manifest(environment={})), profile_data=data).plan
    assert [node.id for node in plan.nodes] == ["group-probes"]
    assert plan.environment["interpreter"] == "/home/me/env/bin/python"
    assert plan.preflight["interpreter"]["source"] == "existing"


def test_a_managed_environment_needs_requirements(tmp_path):
    root = make_repo(tmp_path, requirements=None, manifest=manifest(environment={}))
    assert "environment.requirements is required" in problems_of(root)


def test_dependency_mode_expects_the_pinned_framework(tmp_path):
    plan = compile_repo(make_repo(tmp_path)).plan
    assert plan.preflight["framework"] == {"location": "environment", "distribution": "0.5.0"}


def test_dependency_mode_needs_exactly_one_framework_pin(tmp_path):
    root = make_repo(tmp_path, requirements="torch==2.13.0\n")
    assert "pin training-framework exactly once" in problems_of(root)


def test_checkout_mode_refuses_a_framework_pin(tmp_path):
    data = manifest(code={"framework": "checkout", "paths": ["."]})
    assert "must not contain training-framework" in problems_of(make_repo(tmp_path, manifest=data))


def test_missing_code_paths_are_refused(tmp_path):
    data = manifest(code={"framework": "dependency", "paths": ["src"]})
    assert "'src' does not exist" in problems_of(make_repo(tmp_path, manifest=data))


def test_requires_python_is_checked_against_the_declared_identity(tmp_path):
    root = make_repo(tmp_path, pyproject='[project]\nrequires-python = ">=3.13"\n')
    assert "does not satisfy requires-python" in problems_of(root)


def test_a_missing_requires_python_is_noted(tmp_path):
    plan = compile_repo(make_repo(tmp_path, pyproject=None)).plan
    assert any("no requires-python" in note for note in plan.notes)


# Data sources -------------------------------------------------------------------


UPLOAD = {"name": "flowers", "target": {"component": "flowers102", "path": "root"}, "upload": {}}


def flowers_config(root_value):
    config = copy.deepcopy(TRAINING_CONFIG)
    config["sessions"][0]["flowers102"]["root"] = root_value
    return config


def test_an_upload_is_keyed_by_content_and_rewritten(tmp_path):
    root = make_repo(tmp_path, manifest=with_data([UPLOAD]), config=flowers_config("/data/flowers"))
    facts = FakeFacts(root, existing={"/data/flowers"}, contents={"/data/flowers": ("K1", 2, 20)})
    plan = compile_repo(root, facts=facts).plan
    upload = [a for a in plan.artifacts if a.kind == "upload"][0]
    assert upload.key == "K1"
    assert upload.location == "/scratch/me/data/K1/data/flowers"
    assert upload.location_kind == "key-derived"
    for task in plan.tasks:
        assert task.config["sessions"][0]["flowers102"]["root"] == upload.location


def test_an_upload_under_a_path_map_prefix_goes_there(tmp_path):
    data = profile()
    data["profiles"]["cluster"]["path_map"] = {"/data": "/project/lab/data", "/data/flowers": "/project/flowers"}
    root = make_repo(tmp_path, manifest=with_data([UPLOAD]), config=flowers_config("/data/flowers"))
    facts = FakeFacts(root, existing={"/data/flowers"})
    upload = [a for a in compile_repo(root, profile_data=data, facts=facts).plan.artifacts if a.kind == "upload"][0]
    assert upload.location == "/project/flowers"           # longest prefix
    assert upload.location_kind == "user-chosen"


@pytest.mark.parametrize("value", ["/home/me", "/", "relative/path", "/missing"])
def test_refused_or_missing_upload_paths(tmp_path, value):
    root = make_repo(tmp_path, manifest=with_data([UPLOAD]), config=flowers_config(value))
    facts = FakeFacts(root, existing={"/home/me", "/"})
    assert "data source 'flowers'" in problems_of(root, facts=facts)


def test_uploads_containing_or_inside_the_repository_are_refused(tmp_path):
    root = make_repo(tmp_path, manifest=with_data([UPLOAD]), config=flowers_config(str(tmp_path / "repo" / "my_project")))
    facts = FakeFacts(root, existing={str(root / "my_project")})
    assert "inside the repository" in problems_of(root, facts=facts)


def test_a_declared_upload_version_is_keyed_by_source_name_and_version(tmp_path):
    sources = [dict(copy.deepcopy(UPLOAD), upload={"version": "v1"})]
    root = make_repo(tmp_path, manifest=with_data(sources), config=flowers_config("/data/flowers"))
    facts = FakeFacts(root, existing={"/data/flowers"})
    first = [a for a in compile_repo(root, facts=facts).plan.artifacts if a.kind == "upload"][0]
    assert not facts.content_calls
    assert first.details["version"] == "v1" and first.details["declared"] is True
    root2 = make_repo(tmp_path / "second", manifest=with_data(
        [dict(copy.deepcopy(UPLOAD), name="renamed", upload={"version": "v1"})]),
        config=flowers_config("/data/flowers"))
    second = [a for a in compile_repo(root2, facts=FakeFacts(root2, existing={"/data/flowers"})).plan.artifacts
              if a.kind == "upload"][0]
    assert first.key != second.key


def test_a_cluster_source_is_mapped_or_replaced(tmp_path):
    data = profile()
    data["profiles"]["cluster"]["path_map"] = {"/data": "/project/data"}
    mapped = {"name": "flowers", "target": UPLOAD["target"], "cluster": {}}
    root = make_repo(tmp_path, manifest=with_data([mapped]), config=flowers_config("/data/flowers"))
    plan = compile_repo(root, profile_data=data).plan
    assert plan.preflight["cluster_paths"] == ["/project/data/flowers"]
    given = {"name": "flowers", "target": UPLOAD["target"], "cluster": {"path": "/shared/flowers"}}
    root2 = make_repo(tmp_path / "2", manifest=with_data([given]), config=flowers_config("/data/flowers"))
    plan2 = compile_repo(root2).plan
    assert plan2.tasks[0].config["sessions"][0]["flowers102"]["root"] == "/shared/flowers"


def test_a_relative_unmapped_cluster_path_is_refused(tmp_path):
    source = {"name": "flowers", "target": UPLOAD["target"], "cluster": {}}
    root = make_repo(tmp_path, manifest=with_data([source]), config=flowers_config("data/flowers"))
    assert "not absolute" in problems_of(root)


def prepare_source(**changes):
    source = {
        "name": "inat", "target": UPLOAD["target"],
        "prepare": {"command": ["python", "-m", "prep", "{path}"], "version": "1",
                    "resources": {"walltime": "02:00:00", "cpus": 4, "memory": "16G"}},
    }
    source["prepare"].update(changes)
    return source


def test_each_prepared_path_is_its_own_node_feeding_the_groups_that_use_it(tmp_path):
    data = with_data([prepare_source()])
    data["groups"]["other"] = {
        "resources": {"walltime": "01:00:00", "cpus": 2, "memory": "8G"},
        "experiments": [{"name": "plain", "overrides": [
            {"component": "flowers102", "path": "root", "value": "/other/path"}]}],
    }
    root = make_repo(tmp_path, manifest=data, config=flowers_config("/data/flowers"))
    plan = compile_repo(root).plan
    nodes = {node.id: node for node in plan.nodes}
    assert set(nodes) == {"setup", "prepare-inat-0", "prepare-inat-1", "group-probes", "group-other"}
    assert nodes["group-probes"].depends_on == ("setup", "prepare-inat-0")
    assert nodes["group-other"].depends_on == ("setup", "prepare-inat-1")
    assert nodes["prepare-inat-0"].depends_on == ("setup",)
    assert "#SBATCH --time=02:00:00" in nodes["prepare-inat-0"].script


def test_the_prepare_key_follows_the_commit_unless_inputs_narrow_it(tmp_path):
    def key(root):
        return [a for a in compile_repo(root).plan.artifacts if a.kind == "prepare"][0].key

    root = make_repo(tmp_path, manifest=with_data([prepare_source()]), config=flowers_config("/data/flowers"))
    before = key(root)
    commit(root, "notes.txt", "unrelated")
    assert key(root) != before

    narrowed = with_data([prepare_source(inputs=["my_project"])])
    root2 = make_repo(tmp_path / "2", manifest=narrowed, config=flowers_config("/data/flowers"))
    before = key(root2)
    commit(root2, "notes.txt", "unrelated")
    assert key(root2) == before
    commit(root2, "my_project/prep.py", "changed = True\n")
    assert key(root2) != before


def test_prepare_rules(tmp_path):
    bad = prepare_source(command=["run", "{other}"], inputs=["missing"])
    del bad["prepare"]["resources"]
    root = make_repo(tmp_path, manifest=with_data([bad]), config=flowers_config("/data/flowers"))
    text = problems_of(root)
    assert "unknown placeholder {other}" in text
    assert "resources is required" in text


def test_prepare_resources_are_refused_on_the_login_node(tmp_path):
    data = profile(prepare_location="login")
    root = make_repo(tmp_path, manifest=with_data([prepare_source()]), config=flowers_config("/data/flowers"))
    assert "login node" in problems_of(root, profile_data=data)


def test_uncovered_local_paths_are_suggested_and_noted(tmp_path):
    root = make_repo(tmp_path, config=flowers_config("/data/flowers"))
    compiled = compile_repo(root, facts=FakeFacts(root, existing={"/data/flowers"}))
    assert compiled.suggestions == [{
        "name": "flowers102_root",
        "target": {"component": "flowers102", "path": ["root"]},
        "upload": {},
    }]
    assert any("not data sources" in note for note in compiled.plan.notes)


# Graph, contexts, preflight -------------------------------------------------


def test_task_contexts_isolate_the_environment(tmp_path):
    plan = compile_repo(make_repo(tmp_path)).plan
    context = plan.node("group-probes").context
    assert context.pythonpath == (f"{plan.batch_dir}/source",)
    assert ("PYTHONNOUSERSITE", "1") in context.set_variables
    assert set(context.unset_variables) == {"WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    assert context.hooks == ("module load python/3.12", "module load cuda")
    assert plan.node("setup").context.hooks == ("module load python/3.12",)


def test_preflight_expectations_are_precomputed(tmp_path):
    plan = compile_repo(make_repo(tmp_path)).plan
    assert plan.preflight["components_packages"] == ["my_project.components"]
    assert plan.preflight["interpreter"]["identity"]["version"] == "3.12.4"
    json.dumps(plan.as_json())  # serializable


@pytest.mark.parametrize("source", ["uv", "conda"])
def test_a_found_bootstrap_interpreter_is_linked_to_a_fixed_path(tmp_path, source):
    data = profile()
    data["profiles"]["cluster"]["env"] = {"python_source": source}
    root = make_repo(tmp_path, manifest=manifest(environment={
        "requirements": "requirements/hpc.txt", "python": "3.12"}))
    plan = compile_repo(root, profile_data=data).plan
    context = plan.node("setup").context
    assert context.interpreter == f"{plan.batch_dir}/nodes/setup/python"
    assert "ln -sfn" in context.hooks[-1] and "$(" in context.hooks[-1]
    assert f"exec {context.interpreter} {plan.batch_dir}/bootstrap.py" in plan.node("setup").script


def test_validate_refuses_a_conda_backend_on_the_site_interpreter(tmp_path):
    data = copy.deepcopy(PROFILE)
    data["profiles"]["cluster"]["env"]["backend"] = "conda"
    with pytest.raises(PlanError, match="cannot use python_source 'site'"):
        compile_repo(make_repo(tmp_path), profile_data=data)


def test_suggested_names_are_valid_unique_and_compile_when_pasted(tmp_path):
    # Component instances such as `imagenet#train` contain characters that
    # source names may not; two paths can also end in the same keys.
    config = copy.deepcopy(TRAINING_CONFIG)
    session = config["sessions"][0]
    session["imagenet#train"] = {"root": "/data/imagenet/train"}
    session["imagenet#val"] = {"root": "/data/imagenet/val"}
    session["flowers102"]["root"] = "/data/flowers"
    session["extra"] = {"flowers102": {"root": "/data/flowers-extra"}}
    existing = {"/data/imagenet/train", "/data/imagenet/val", "/data/flowers", "/data/flowers-extra"}
    root = make_repo(tmp_path / "first", config=config)
    suggestions = compile_repo(root, facts=FakeFacts(root, existing=existing)).suggestions
    names = [suggestion["name"] for suggestion in suggestions]
    assert "imagenet_train_root" in names and "imagenet_val_root" in names
    assert len(set(names)) == len(names) == 4
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names)
    assert {"component": "imagenet#train", "path": ["root"]} in [s["target"] for s in suggestions]

    pasted = make_repo(tmp_path / "second", manifest=with_data(suggestions), config=config)
    contents = {path: (f"K{index}", 1, 1) for index, path in enumerate(sorted(existing))}
    compiled = compile_repo(pasted, facts=FakeFacts(pasted, existing=existing, contents=contents))
    assert not compiled.suggestions
    assert len([a for a in compiled.plan.artifacts if a.kind == "upload"]) == 4


def imagenet_batch(tmp_path, train, val, train_spec=None, val_spec=None, path_map=None):
    config = copy.deepcopy(TRAINING_CONFIG)
    config["sessions"][0]["imagenet#train"] = {"root": train}
    config["sessions"][0]["imagenet#val"] = {"root": val}
    sources = [
        {"name": "train", "target": {"component": "imagenet#train", "path": ["root"]},
         "upload": train_spec or {}},
        {"name": "val", "target": {"component": "imagenet#val", "path": ["root"]},
         "upload": val_spec or {}},
    ]
    data = profile(path_map=path_map) if path_map else None
    root = make_repo(tmp_path, manifest=with_data(sources), config=config)
    contents = {"/data/imagenet": ("K1", 10, 100), "/data/other": ("K2", 10, 100)}
    facts = FakeFacts(root, existing=set(contents), contents=contents)
    return root, data, facts


def test_two_sources_naming_one_dataset_upload_it_once(tmp_path):
    root, data, facts = imagenet_batch(tmp_path, "/data/imagenet", "/data/imagenet")
    plan = compile_repo(root, profile_data=data, facts=facts).plan
    (upload,) = [a for a in plan.artifacts if a.kind == "upload"]
    configs = [task.config["sessions"][0] for task in plan.tasks]
    assert {c["imagenet#train"]["root"] for c in configs} == {upload.location}
    assert {c["imagenet#val"]["root"] for c in configs} == {upload.location}


def test_one_declared_dataset_under_two_names_is_also_one_upload(tmp_path):
    spec = {"version": "ilsvrc2012"}
    root, data, facts = imagenet_batch(tmp_path, "/data/imagenet", "/data/imagenet", spec, spec,
                                       path_map={"/data": "/N/project/me/data"})
    plan = compile_repo(root, profile_data=data, facts=facts).plan
    assert len([a for a in plan.artifacts if a.kind == "upload"]) == 1


def test_different_content_for_one_location_is_refused(tmp_path):
    # Two versions of the same local path map to one cluster location.
    root, data, facts = imagenet_batch(tmp_path, "/data/imagenet", "/data/imagenet",
                                       {"version": "1"}, {"version": "2"},
                                       path_map={"/data": "/N/project/me/data"})
    assert "would upload different content to the same location" in problems_of(
        root, profile_data=data, facts=facts)
