"""Write a plan's files: the batch directory as it will be staged, minus
the source checkout, receipt, records, logs, and sessions."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from training_framework.hpc.errors import HpcError
from training_framework.hpc.names import canonical_json
from training_framework.hpc.plan import ExecutionPlan


def plan_files(plan: ExecutionPlan, manifest_raw: dict) -> dict[str, str]:
    """Relative path -> content of every file the plan stages."""
    files: dict[str, str] = {
        "plan.json": json.dumps(plan.as_json(), indent=2, sort_keys=True) + "\n",
        "manifest.yaml": yaml.safe_dump(manifest_raw, sort_keys=False),
    }
    rows = ["source\tvalue\tlocation\tsize\tkind"]
    for artifact in plan.artifacts:
        if artifact.kind == "environment":
            continue
        details = artifact.details
        value = details.get("local") or details.get("value") or artifact.location
        size = details.get("size")
        rows.append("\t".join([
            artifact.source, value, artifact.location,
            "" if size is None else str(size), artifact.kind,
        ]))
    files["data.tsv"] = "\n".join(rows) + "\n"

    tables: dict[str, list[str]] = {}
    for task in plan.tasks:
        tables.setdefault(task.group, ["index\ttask\tvariant\tconfig\tconfig_hash"]).append(
            "\t".join([
                str(task.array_index), task.name, canonical_json(list(task.variant)),
                f"{task.directory}/config.yaml", task.config_hash,
            ])
        )
        files[f"{task.directory}/config.yaml"] = yaml.safe_dump(task.config, sort_keys=False)
        files[f"{task.directory}/variant.json"] = json.dumps(list(task.variant), indent=2) + "\n"
    for group, rows in tables.items():
        files[f"groups/{group}/tasks.tsv"] = "\n".join(rows) + "\n"
    for node in plan.nodes:
        if node.kind == "task_set":
            files[f"groups/{node.group}/{node.id}.sbatch"] = node.script
        else:
            files[f"nodes/{node.id}/{node.id}.sbatch"] = node.script
    return files


def write_plan(plan: ExecutionPlan, manifest_raw: dict, output: Path) -> list[str]:
    if output.exists() and any(output.iterdir()):
        raise HpcError(f"{output} is not empty; choose a new output directory")
    files = plan_files(plan, manifest_raw)
    for relative, content in sorted(files.items()):
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return sorted(files)
