# Comparing Runs

[← Docs](../README.md) · [Project README](../../README.md)

After a sweep you have many run directories, each with its TensorBoard
events and the config it ran with. `summarize` reads them all and writes one
report: the hyperparameters that differ between the runs, each metric you
ask for, and the runs that differ only in their seed grouped into
mean ± std. It works the same on local runs, on an HPC batch directory, and
on results brought back with [`hpc fetch`](08-hpc-batches.md#11-getting-results-back).

It assumes the runs use the [`tensorboard`](../reference/builtin-components.md#tensorboard)
resource, usually with
[`tensorboard_summaries`](../reference/builtin-components.md#tensorboard_summaries)
writing the values you want to compare.

## A first report

```bash
python -m training_framework.reporting summarize runs/ \
  --metric train/loss:min --metric val/accuracy:max --sort val/accuracy
```

`runs/` here holds one `sessions_dir` per learning rate and seed
(`runs/lr0.001/seed41/session_<timestamp>/`, ...). The report (Markdown, on
standard output unless you pass `-o FILE`) starts with what it compared and
then has four sections:

```text
## Seed groups

| group | optimizer.optimizer.kwargs.lr | runs | seeds | train/loss last | train/loss best | val/accuracy last | val/accuracy best |
|---|---|---|---|---|---|---|---|
| G1 | 0.001  | 3 | 41, 42, 43 | 0.5 ± 0.01 (n=3) | 0.5 ± 0.01 (n=3) | 0.772727 ± 0.00455 (n=3) | 0.772727 ± 0.00455 (n=3) |
| G2 | 0.0005 | 3 | 41, 42, 43 | 0.8 ± 0.01 (n=3) | 0.8 ± 0.01 (n=3) | 0.636364 ± 0.00455 (n=3) | 0.636364 ± 0.00455 (n=3) |

## Runs

| run | group | optimizer.optimizer.kwargs.lr | session_config.rng_seed | extended | attempts | train/loss last | ... |
|---|---|---|---|---|---|---|---|
| lr0.001/seed41/session_20261031_104100 | G1 | 0.001 | 41 | - | 1 | 0.49 @ 1000 | ... |
...

## Shared settings

    ddp.world_size = 1
    optimizer.optimizer.kind = adamw
    session_config.max_iterations = 1000

Not compared (ignored keys): `session_config.sessions_dir`, `ddp.master_addr`, ...

## Problems

None.
```

- **Seed groups**: the runs whose settings are equal except for the seed,
  with the mean ± sample standard deviation of each value over the group
  and the number of runs it is over.
- **Runs**: one row per run. The hyperparameter columns are only the
  settings that differ between the runs compared; `value @ step` gives the
  step a value was logged at.
- **Shared settings**: every setting equal in all runs, so the report says
  what was held fixed. When some runs have no readable `config.yaml`, the
  section says so first: the settings are equal in the runs with a config
  and unknown for the ones named.
- **Problems**: anything that makes a number less than it looks (below).
  Nothing is dropped or averaged over without appearing here.

## What a run is

A run is a session directory, `<sessions_dir>/session_<timestamp>/`. The
session writes its config to `config.yaml` there, and the `tensorboard`
resource writes its events to `Tensorboard_tensorboard/` inside it (one
directory per instance, `Tensorboard_tensorboard_<suffix>/`, whose tags are
then prefixed with the directory name).

A root can be a `sessions_dir`, any directory above several of them, a single
session directory, an HPC batch directory, or `hpc-results/<batch-id>/`.
Several roots go in one report; each run's id then starts with its root's
name:

```bash
python -m training_framework.reporting summarize runs/baseline runs/sweep-2 --metric loss:min
```

A run's id is its path below the root. An HPC task is called by its task
name, `<group>/<experiment>/<NNN>`, and its matrix values from `variant.json`
are columns too, labelled like the batch manifest (`optimizer:optimizer.kwargs.lr`).
`--include GLOB` and `--exclude GLOB` choose runs by id.

## The values

`--metric TAG` reports the tag's last value. `TAG:min` or `TAG:max` also
reports its best value, the lowest or highest; the direction is never
guessed from the tag's name. `--last-n N` adds the mean of the last N
points, which is steadier than a single last value for a noisy loss.
`--tags GLOB` adds every tag matching it (`--tags 'optimizer/*'`). With no
`--metric` and no `--tags`, every scalar tag is reported, last value only.
`--sort TAG` orders groups and runs by the best value of a `--metric` with a
direction.

Every point is read. TensorBoard's own reader keeps a sample of 10 000 points
per tag and, when a step goes backwards, drops later points without a word;
`summarize` reads the event files itself.

**Resumed runs.** A resume adds a new event file to the same writer
directory and writes again the steps after the checkpoint it started from.
The files are ordered by when they were written and merged by step, the
later attempt winning: the values you get are the ones the finished run
actually went through. The `attempts` column shows how many launches the run
had and how many steps the later ones rewrote.

**Extended runs.** A run continued with `--extend-session` has
`config_extension_<timestamp>.yaml` files, counted in `extended`. Its
hyperparameters are the ones in `config.yaml`, that is the latest; the
earlier ones applied to the first part of the run.

## Seed groups

Runs are grouped when every compared setting is equal except the seed keys,
by default `session_config.rng_seed`. Give other seeds with `--seed-key`
(repeatable; it replaces the default):

```bash
python -m training_framework.reporting summarize runs/ --metric loss:min \
  --seed-key session_config.rng_seed --seed-key data_manager:seed
```

A seed key is a config path (`resources.data_manager.seed`) or, like a
matrix address in a batch manifest, `component:path`: the path inside that
component's entry, wherever the entry is listed in each run's config (the
top level or a `resources`/`hooks`/`steps` group). Matrix values match
either form.

A seed key that no run has is an error, so a misspelt key does not turn every
run into a group of one. `--no-seed-groups` reports runs only. The report
shows, rather than hides:

- a group of one run (no standard deviation, `n=1`);
- members whose metric ends at different steps (an unfinished or extended
  run), whose last values are not from the same iteration;
- two members with the same seed, which counts one configuration twice;
- non-finite values (a diverged run): shown in the run's row, left out of
  `best` and of the group statistics, and counted.

A seed the report does not know about, such as a data manager's own seed
left at different values, simply separates the runs: it shows up as a
column, so you can see why.

## Which settings are compared

Settings are compared by their path in `config.yaml`: `optimizer.optimizer.kwargs.lr`,
list items as `steps[0]`, and a key that is not a plain name quoted, as in
`["projection_head#student"].out_dim`. A setting one run has and another
does not is a column, `absent` where it is missing. Values of different
types are different (`1`, `1.0`, and `true` are three values).

Some settings differ between runs by construction and are not compared:
`session_config.sessions_dir` (every HPC task has its own) and the rendezvous
`ddp.master_addr` / `ddp.master_port` (the engine picks a free port at
every launch), at the top level or in a `resources` group. `ddp.world_size`
is compared: it changes the global batch size. `--ignore-key PATH` adds a
path (and everything under it) to these; `--no-default-ignores` compares
the defaults too.

## Problems and `--strict`

The **Problems** section lists, among others: a truncated or corrupt event
file (with the byte it stopped at; what came before is kept), event files in
no session directory, a run without `config.yaml` (its hyperparameters are
`unknown`, and it is grouped with no other run), a tag missing in some runs,
and the group problems above. With `--strict` the command exits with status
1 when there are any, for scripts that must not go on with a doubtful
comparison. Options that cannot be met, such as a `--metric` no run writes
(the error lists the tags that exist), exit with status 2 and write nothing.

## Other formats and Python

`--format csv` writes one table, `--table groups` (the default) or
`--table runs`, with every number in full and a column per statistic, for a
spreadsheet. `--format json` writes everything: groups, runs with their
attempts, shared settings, and problems. The report is never written into a
run directory.

The same is available from Python:

```python
from training_framework.reporting import (
    discover_runs, render_markdown, summarize_runs,
)

discovery = discover_runs("runs/baseline", "runs/sweep-2")
summary = summarize_runs(discovery, ["train/loss:min", "val/accuracy:max"], last_n=5)
for group in summary.groups:
    stats = group.stats[("val/accuracy", "best")]
    print(group.hyperparameters, stats.mean, stats.std, stats.n)
print(render_markdown(summary))
```

Every option of the command line is described in the
[CLI reference](../reference/cli.md#run-summaries).

---

**See also:** [Running batches on an HPC cluster](08-hpc-batches.md) for
producing and fetching sweeps.
