"""`load_config`: session entries that include other files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from training_framework.config_files import (
    ConfigFileError,
    RepositoryFiles,
    check_no_interpolation,
    load_config,
    unescape,
)


def _write(root: Path, files: dict[str, object]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content if isinstance(content, str) else yaml.safe_dump(content), encoding="utf-8")
    return root / next(iter(files))


def _session(root, files):
    return load_config(_write(root, files)).config["sessions"][0]


def test_fragments_merge_in_order_and_the_entry_overrides_them(tmp_path):
    session = _session(tmp_path, {
        "train.yaml": {"sessions": [{
            "includes": ["common/model.yaml", "common/optimizer.yaml"],
            "session_config": {"rng_seed": 1},
            "optimizer": {"optimizer": {"kwargs": {"lr": 0.0005}}},
        }]},
        "common/model.yaml": {"vit": {"depth": 12, "width": 384}},
        "common/optimizer.yaml": {"optimizer": {"optimizer": {"kind": "adamw", "kwargs": {"lr": 0.001, "wd": 0.04}}}},
    })

    assert session == {
        "vit": {"depth": 12, "width": 384},
        "optimizer": {"optimizer": {"kind": "adamw", "kwargs": {"lr": 0.0005, "wd": 0.04}}},
        "session_config": {"rng_seed": 1},
    }
    assert "includes" not in session


def test_a_fragment_includes_others_relative_to_itself(tmp_path):
    session = _session(tmp_path, {
        "configs/train.yaml": {"sessions": [{"includes": ["../shared/recipe.yaml"]}]},
        "shared/recipe.yaml": {"includes": ["parts/data.yaml"], "model": {"depth": 2}},
        "shared/parts/data.yaml": {"data_manager": {"batch_size": 8}},
    })

    assert session == {"data_manager": {"batch_size": 8}, "model": {"depth": 2}}


def test_a_fragment_overrides_what_it_includes(tmp_path):
    session = _session(tmp_path, {
        "train.yaml": {"sessions": [{"includes": ["recipe.yaml"]}]},
        "recipe.yaml": {"includes": ["base.yaml"], "model": {"depth": 4}},
        "base.yaml": {"model": {"depth": 2, "width": 8}},
    })

    assert session == {"model": {"depth": 4, "width": 8}}


def test_lists_are_replaced_whole(tmp_path):
    session = _session(tmp_path, {
        "train.yaml": {"sessions": [{"includes": ["base.yaml"], "freeze": {"rules": ["head.*"]}}]},
        "base.yaml": {"freeze": {"rules": ["backbone.*", "embed.*"], "until": 3}},
    })

    assert session == {"freeze": {"rules": ["head.*"], "until": 3}}


def test_two_fragments_setting_a_value_differently_is_an_error(tmp_path):
    with pytest.raises(ConfigFileError, match=r"two included files set the same value: optimizer\.lr is 0\.001 in .*a\.yaml and 0\.01 in .*b\.yaml"):
        _session(tmp_path, {
            "train.yaml": {"sessions": [{"includes": ["a.yaml", "b.yaml"]}]},
            "a.yaml": {"optimizer": {"lr": 0.001}},
            "b.yaml": {"optimizer": {"lr": 0.01}},
        })


def test_two_fragments_setting_a_value_alike_is_fine(tmp_path):
    session = _session(tmp_path, {
        "train.yaml": {"sessions": [{"includes": ["a.yaml", "b.yaml"]}]},
        "a.yaml": {"optimizer": {"lr": 0.001, "kind": "sgd"}},
        "b.yaml": {"optimizer": {"lr": 0.001, "momentum": 0.9}},
    })

    assert session == {"optimizer": {"lr": 0.001, "kind": "sgd", "momentum": 0.9}}


@pytest.mark.parametrize("first, second", [
    ({"model": {"depth": 2}}, {"model": 3}),
    ({"model": 3}, {"model": {"depth": 2}}),
    ({"flag": 1}, {"flag": True}),
])
def test_fragments_that_disagree_on_a_value_or_its_shape_are_an_error(tmp_path, first, second):
    with pytest.raises(ConfigFileError, match="two included files set the same value"):
        _session(tmp_path, {
            "train.yaml": {"sessions": [{"includes": ["a.yaml", "b.yaml"]}]},
            "a.yaml": first,
            "b.yaml": second,
        })


@pytest.mark.parametrize("order", [["null.yaml", "map.yaml"], ["map.yaml", "null.yaml"]])
def test_a_null_entry_merges_with_a_mapping(tmp_path, order):
    # `ddp:` with no settings is the same as `ddp: {}`, in either order.
    session = _session(tmp_path, {
        "train.yaml": {"sessions": [{"includes": order}]},
        "null.yaml": "ddp:\n",
        "map.yaml": {"ddp": {"world_size": 2}},
    })

    assert session == {"ddp": {"world_size": 2}}


def test_a_null_entry_in_the_including_file_keeps_the_fragment_settings(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"ddp": {"world_size": 2}}))
    (tmp_path / "train.yaml").write_text("sessions:\n  - includes: [base.yaml]\n    ddp:\n")

    assert load_config(tmp_path / "train.yaml").config["sessions"] == [{"ddp": {"world_size": 2}}]


def test_the_including_entry_may_override_any_fragment(tmp_path):
    session = _session(tmp_path, {
        "train.yaml": {"sessions": [{"includes": ["a.yaml"], "model": 3}]},
        "a.yaml": {"model": {"depth": 2}},
    })

    assert session == {"model": 3}


def test_an_empty_fragment_adds_nothing(tmp_path):
    session = _session(tmp_path, {
        "train.yaml": {"sessions": [{"includes": ["empty.yaml", "a.yaml"]}]},
        "empty.yaml": "",
        "a.yaml": {"model": {"depth": 2}},
    })

    assert session == {"model": {"depth": 2}}


def test_each_session_has_its_own_includes(tmp_path):
    loaded = load_config(_write(tmp_path, {
        "train.yaml": {"sessions": [{"includes": ["a.yaml"]}, {"model": {"depth": 9}}]},
        "a.yaml": {"model": {"depth": 2}},
    }))

    assert loaded.config["sessions"] == [{"model": {"depth": 2}}, {"model": {"depth": 9}}]
    assert [len(sources) for sources in loaded.sources] == [2, 1]


def test_sources_list_every_file_with_its_hash_and_includer(tmp_path):
    top = _write(tmp_path, {
        "train.yaml": {"sessions": [{"includes": ["recipe.yaml"]}]},
        "recipe.yaml": {"includes": ["base.yaml"]},
        "base.yaml": {"model": {"depth": 2}},
    })

    (sources,) = load_config(top).sources

    names = [(Path(s.path).name, s.included_by and Path(s.included_by).name) for s in sources]
    assert names == [("train.yaml", None), ("recipe.yaml", "train.yaml"), ("base.yaml", "recipe.yaml")]
    assert sources[2].sha256 == hashlib.sha256((tmp_path / "base.yaml").read_bytes()).hexdigest()


# -- refused -------------------------------------------------------------------------


def test_a_missing_file_names_the_chain(tmp_path):
    with pytest.raises(ConfigFileError, match=r"missing\.yaml: no such file \(included via .*train\.yaml -> .*recipe\.yaml\)"):
        _session(tmp_path, {
            "train.yaml": {"sessions": [{"includes": ["recipe.yaml"]}]},
            "recipe.yaml": {"includes": ["missing.yaml"]},
        })


def test_a_cycle_is_an_error_showing_it(tmp_path):
    with pytest.raises(ConfigFileError, match=r"includes form a cycle: .*a\.yaml -> .*b\.yaml -> .*a\.yaml"):
        _session(tmp_path, {
            "train.yaml": {"sessions": [{"includes": ["a.yaml"]}]},
            "a.yaml": {"includes": ["b.yaml"]},
            "b.yaml": {"includes": ["a.yaml"]},
        })


def test_including_the_same_file_twice_side_by_side_is_fine(tmp_path):
    # Not a cycle: neither includes the other.
    session = _session(tmp_path, {
        "train.yaml": {"sessions": [{"includes": ["a.yaml", "b.yaml"]}]},
        "a.yaml": {"includes": ["base.yaml"]},
        "b.yaml": {"includes": ["base.yaml"]},
        "base.yaml": {"model": {"depth": 2}},
    })

    assert session == {"model": {"depth": 2}}


@pytest.mark.parametrize("fragment, message", [
    ("- a\n- b\n", "a fragment file must be a mapping, not a list"),
    ({"sessions": []}, "cannot have 'sessions'"),
    ("a: [1,\n", "not readable YAML"),
])
def test_a_fragment_that_is_not_a_session_fragment_is_refused(tmp_path, fragment, message):
    with pytest.raises(ConfigFileError, match=message):
        _session(tmp_path, {
            "train.yaml": {"sessions": [{"includes": ["bad.yaml"]}]},
            "bad.yaml": fragment,
        })


@pytest.mark.parametrize("includes", ["a.yaml", [""], [3], {"a": 1}])
def test_includes_must_be_a_list_of_paths(tmp_path, includes):
    with pytest.raises(ConfigFileError, match=r"sessions\[0\]\.includes must be a list of file paths"):
        _session(tmp_path, {"train.yaml": {"sessions": [{"includes": includes}]}})


def test_includes_at_the_top_of_the_file_are_refused(tmp_path):
    with pytest.raises(ConfigFileError, match="belongs in a session entry"):
        _session(tmp_path, {"train.yaml": {"includes": ["a.yaml"], "sessions": [{}]}})


# -- interpolation -------------------------------------------------------------------


def test_interpolation_is_refused_with_its_path(tmp_path):
    with pytest.raises(ConfigFileError, match=r"optimizer\.lr is '\$\{common\.lr\}'; interpolation"):
        _session(tmp_path, {
            "train.yaml": {"common": {"lr": 0.1}, "sessions": [{"optimizer": {"lr": "${common.lr}"}}]},
        })


def test_interpolation_in_a_fragment_is_refused_naming_the_fragment(tmp_path):
    with pytest.raises(ConfigFileError, match=r"a\.yaml.*: steps\.\[0\]\.name is"):
        _session(tmp_path, {
            "train.yaml": {"sessions": [{"includes": ["a.yaml"]}]},
            "a.yaml": {"steps": [{"name": "${oc.env:USER}"}]},
        })


def test_an_escaped_dollar_brace_stays_escaped_until_unescaped(tmp_path):
    session = _session(tmp_path, {"train.yaml": "sessions:\n  - note: \\${not.an.interpolation}\n"})

    assert session == {"note": "\\${not.an.interpolation}"}
    check_no_interpolation(session, "here")
    assert unescape(session) == {"note": "${not.an.interpolation}"}


# -- inside a repository -------------------------------------------------------------


def _repository(files):
    return RepositoryFiles(lambda path: files[path].encode() if path in files else None)


def test_repository_files_are_read_by_repository_path():
    reader = _repository({
        "configs/train.yaml": yaml.safe_dump({"sessions": [{"includes": ["common/a.yaml"]}]}),
        "configs/common/a.yaml": yaml.safe_dump({"model": {"depth": 2}}),
    })

    loaded = load_config("configs/train.yaml", reader=reader)

    assert loaded.config["sessions"] == [{"model": {"depth": 2}}]
    assert [s.path for s in loaded.sources[0]] == ["configs/train.yaml", "configs/common/a.yaml"]


@pytest.mark.parametrize("path", ["../../outside.yaml", "/etc/a.yaml", "a\\b.yaml"])
def test_a_repository_include_may_not_leave_the_repository(path):
    reader = _repository({"configs/train.yaml": yaml.safe_dump({"sessions": [{"includes": [path]}]})})

    with pytest.raises(ConfigFileError, match="files of the repository"):
        load_config("configs/train.yaml", reader=reader)
