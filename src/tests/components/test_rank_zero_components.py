"""Tests for deciding what a secondary DDP rank builds.

Ranks other than zero used to be given an opt-in list, `ddp.parallel_components`.
They now build every configured component except the ones declared
rank-zero-only, either with `@rank_zero_only` on the class or with
`ddp.rank_zero_components` in the session.

Component classes are created inside the test functions on purpose: the
autouse registry fixture clears the global registries before every test, so a
class registered at import time would be gone by the time its test runs.
"""

import warnings

import pytest

from training_framework.components import (
    LifecycleHook,
    Resource,
    Step,
    component_registry,
    hook,
    rank_zero_only,
    requires_resource,
    resource,
    step,
)
from tests.test_utils import build_session
from training_framework.session.config import TRAINING_SESSION_TYPE


def make_resource(name: str, *, requires: str | None = None) -> type[Resource]:
    def setup(self, session) -> None:
        pass

    def teardown(self, session) -> None:
        pass

    component = type(
        f"Resource_{name}",
        (Resource,),
        {"setup": setup, "teardown": teardown},
    )
    if requires is not None:
        component = requires_resource(requires)(component)
    return resource(name, session_type=TRAINING_SESSION_TYPE)(component)


def make_step(name: str, *, requires: str | None = None) -> type[Step]:
    def run(self, session) -> None:
        pass

    component = type(f"Step_{name}", (Step,), {"run": run})
    if requires is not None:
        component = requires_resource(requires)(component)
    return step(name, session_type=TRAINING_SESSION_TYPE)(component)


def make_hook(name: str, *, requires: str | None = None) -> type[LifecycleHook]:
    def noop(self, session) -> None:
        pass

    component = type(
        f"Hook_{name}",
        (LifecycleHook,),
        {
            "call_every": 1,
            "pre_session": noop,
            "post_session": noop,
            "pre_iteration_callback": noop,
            "post_iteration_callback": noop,
        },
    )
    if requires is not None:
        component = requires_resource(requires)(component)
    return hook(name, session_type=TRAINING_SESSION_TYPE)(component)


DDP_CONFIG = {
    "world_size": 2,
    "backend": "gloo",
    "master_addr": "127.0.0.1",
    "master_port": "29500",
}


def session_for(tmp_path, active, *, bindings=None):
    """A session configuring `active`, with the built-in `ddp` bound to its
    `model` role.

    The session also holds the default logger and checkpointer. Both are
    rank-zero-only by class, so no answer below includes them.
    """
    return build_session(
        tmp_path,
        {name: dict(DDP_CONFIG) if name == "ddp" else {} for name in active},
        bindings={"model": "rz_model", **(bindings or {})},
    )


def test_rank_zero_only_marks_the_class_and_its_subclasses():
    marked = rank_zero_only(make_hook("rz_marked"))

    class Derived(marked):
        pass

    assert marked.rank_zero_only is True
    assert Derived.rank_zero_only is True
    assert make_hook("rz_plain").rank_zero_only is False


def test_rank_zero_only_rejects_something_that_is_not_a_component():
    class NotAComponent:
        pass

    with pytest.raises(TypeError, match="NotAComponent"):
        rank_zero_only(NotAComponent)


def test_the_built_in_reporting_components_are_rank_zero_only():
    registry = component_registry(TRAINING_SESSION_TYPE)

    reporting = ("logger", "checkpointer", "tensorboard", "timer")
    assert all(registry[name].rank_zero_only for name in reporting)

    parallel = ("ddp", "optimizer", "data_manager")
    assert not any(registry[name].rank_zero_only for name in parallel)


def test_secondary_ranks_keep_everything_that_is_not_rank_zero_only(tmp_path):
    make_resource("rz_model")
    make_step("rz_train", requires="ddp")
    rank_zero_only(make_hook("rz_report"))
    active = {"ddp", "rz_model", "rz_train", "rz_report"}

    keep = session_for(tmp_path, active).rank_parallel_names()

    assert keep == {"ddp", "rz_model", "rz_train"}


def test_a_dependency_of_a_rank_zero_component_is_kept_unless_declared_too(tmp_path):
    """Nothing is dropped by inference, only by declaration."""
    make_resource("rz_model")
    make_resource("rz_reporting_sink")
    make_step("rz_train", requires="ddp")
    rank_zero_only(make_hook("rz_report", requires="rz_reporting_sink"))
    active = {"ddp", "rz_model", "rz_train", "rz_report", "rz_reporting_sink"}
    session = session_for(tmp_path, active)

    kept = session.rank_parallel_names()
    declared = session.rank_parallel_names(
        rank_zero_components=["rz_reporting_sink"],
    )

    assert kept == {"ddp", "rz_model", "rz_train", "rz_reporting_sink"}
    assert declared == {"ddp", "rz_model", "rz_train"}


def test_a_dependency_a_parallel_component_shares_is_kept(tmp_path):
    make_resource("rz_model")
    make_resource("rz_shared")
    make_step("rz_train", requires="rz_shared")
    rank_zero_only(make_hook("rz_report", requires="rz_shared"))
    active = {"ddp", "rz_model", "rz_train", "rz_report", "rz_shared"}

    keep = session_for(tmp_path, active).rank_parallel_names()

    assert keep == {"ddp", "rz_model", "rz_train", "rz_shared"}


def test_a_rank_zero_component_a_parallel_one_needs_is_kept_with_a_warning(tmp_path):
    """Correctness over pruning: a prerequisite has to exist on the rank."""
    make_resource("rz_model")
    rank_zero_only(make_resource("rz_report_sink"))
    make_step("rz_train", requires="rz_report_sink")
    active = {"ddp", "rz_model", "rz_train", "rz_report_sink"}

    with pytest.warns(RuntimeWarning, match="rz_report_sink"):
        keep = session_for(tmp_path, active).rank_parallel_names()

    assert keep == {"ddp", "rz_model", "rz_train", "rz_report_sink"}


def test_the_config_key_drops_a_component_that_is_not_marked(tmp_path):
    make_resource("rz_model")
    make_step("rz_train", requires="ddp")
    make_hook("rz_report")
    active = {"ddp", "rz_model", "rz_train", "rz_report"}

    keep = session_for(tmp_path, active).rank_parallel_names(rank_zero_components=["rz_report"],
    )

    assert keep == {"ddp", "rz_model", "rz_train"}


def test_the_config_key_warns_when_it_excludes_a_component_using_ddp(tmp_path):
    """The escape hatch can recreate the hang the opt-in list invited."""
    make_resource("rz_model")
    make_hook("rz_report", requires="ddp")
    active = {"ddp", "rz_model", "rz_report"}

    with pytest.warns(RuntimeWarning, match="rz_report"):
        keep = session_for(tmp_path, active).rank_parallel_names(rank_zero_components=["rz_report"],
        )

    assert keep == {"ddp", "rz_model"}


def test_a_class_marked_component_using_ddp_is_not_questioned(tmp_path):
    """`timer` requires `optimizer`, which requires `ddp`, and is marked."""
    make_resource("rz_model")
    rank_zero_only(make_hook("rz_report", requires="ddp"))
    active = {"ddp", "rz_model", "rz_report"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        keep = session_for(tmp_path, active).rank_parallel_names()

    assert keep == {"ddp", "rz_model"}
    assert [str(warning.message) for warning in caught] == []


def test_the_config_key_resolves_role_names_through_the_bindings(tmp_path):
    make_resource("rz_model")
    make_hook("rz_report")
    active = {"ddp", "rz_model", "rz_report"}
    session = session_for(tmp_path, active, bindings={"reporter": "rz_report"})

    keep = session.rank_parallel_names(
        rank_zero_components=["reporter"],
    )

    assert keep == {"ddp", "rz_model"}


def test_the_config_key_rejects_a_component_this_session_does_not_configure(tmp_path):
    make_resource("rz_model")
    make_hook("rz_report")
    active = {"ddp", "rz_model"}

    with pytest.raises(RuntimeError, match="rank_zero_components"):
        session_for(tmp_path, active).rank_parallel_names(rank_zero_components=["rz_report"],
        )


def test_the_ddp_resource_itself_is_never_dropped(tmp_path):
    make_resource("rz_model")
    active = {"ddp", "rz_model"}

    keep = session_for(tmp_path, active).rank_parallel_names(rank_zero_components=["ddp"],
    )

    assert keep == {"ddp", "rz_model"}


def test_parallel_components_still_decides_the_answer_when_given(tmp_path):
    """The deprecated list keeps its exact opt-in meaning, empty included."""
    make_resource("rz_model")
    make_step("rz_train")
    make_hook("rz_report")
    active = {"ddp", "rz_model", "rz_train", "rz_report"}
    session = session_for(tmp_path, active)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        empty = session.rank_parallel_names(parallel_components=[])
        listed = session.rank_parallel_names(parallel_components=["rz_train"])

    assert empty == {"ddp", "rz_model"}
    assert listed == {"ddp", "rz_model", "rz_train"}


def test_parallel_components_warns_when_it_prunes_a_component_using_ddp(tmp_path):
    make_resource("rz_model")
    make_step("rz_train", requires="ddp")
    active = {"ddp", "rz_model", "rz_train"}

    with pytest.warns(RuntimeWarning, match="rz_train"):
        keep = session_for(tmp_path, active).rank_parallel_names(parallel_components=[],
        )

    assert keep == {"ddp", "rz_model"}


def test_parallel_components_is_silent_when_the_list_is_complete(tmp_path):
    make_resource("rz_model")
    make_step("rz_train", requires="ddp")
    make_hook("rz_report")
    active = {"ddp", "rz_model", "rz_train", "rz_report"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        keep = session_for(tmp_path, active).rank_parallel_names(parallel_components=["rz_train"],
        )

    assert keep == {"ddp", "rz_model", "rz_train"}
    assert [str(warning.message) for warning in caught] == []
