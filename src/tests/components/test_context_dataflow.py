"""Ordering and checking steps by the iteration_context keys they use.

Components are declared inside the test functions on purpose: the autouse
registry fixture clears the global registries before each test.
"""

import pytest

from tests.test_utils import build_session, component_names, make_config
from training_framework.components import (
    LifecycleHook,
    Resource,
    SessionHook,
    StatefulResource,
    Step,
    hook,
    rank_zero_only,
    reads,
    requires_step,
    resource,
    step,
    writes,
)
from training_framework.components.edges import Edge, EdgeKind
from training_framework.engine import load_session_for_worker
from training_framework.session import TrainingSession


class _Hook(LifecycleHook):
    call_every = 1

    def pre_session(self, session):
        pass

    def post_session(self, session):
        pass

    def pre_iteration_callback(self, session):
        return _placeholder_outputs(self)

    def post_iteration_callback(self, session, **values):
        pass


class _InertModel(StatefulResource):
    def setup(self, session):
        pass

    def teardown(self, session):
        pass

    def get_state(self):
        return None

    def set_state(self, state):
        pass


def _placeholder_outputs(component):
    """1.0 for each key `component` declares writing, returned as declared
    writes are: one as the value, several as a mapping by name."""
    names = list(component.context_writes())
    if not names:
        return None
    if len(names) == 1:
        return 1.0
    return {name: 1.0 for name in names}


def _recording_step(name, ran, *, read=(), write=()):
    def run(self, session, **values):
        ran.append(name)
        return _placeholder_outputs(self)

    cls = type(name, (Step,), {"run": run})
    if read:
        cls = reads(*read)(cls)
    if write:
        cls = writes(*write)(cls)
    return step(name)(cls)


def _run_one(session):
    with session:
        next(session)


def test_a_reader_runs_after_its_writer_whatever_the_declaration_order(tmp_path):
    ran = []
    _recording_step("df_loss", ran, read=["logits"], write=["loss"])
    _recording_step("df_head", ran, read=["features"], write=["logits"])
    _recording_step("df_body", ran, write=["features"])

    session = build_session(
        tmp_path, {"df_loss": {}, "df_head": {}, "df_body": {}},
    )
    _run_one(session)

    assert ran == ["df_body", "df_head", "df_loss"]


def test_two_writers_of_one_key_are_rejected_naming_both(tmp_path):
    _recording_step("df_a", [], write=["x"])
    _recording_step("df_b", [], write=["x"])

    with pytest.raises(
            RuntimeError,
            match="'x' is written by both Step.df_a and Step.df_b",
    ):
        build_session(tmp_path, {"df_a": {}, "df_b": {}})


def test_updating_a_key_in_place_is_rejected(tmp_path):
    _recording_step("df_normalize", [], read=["x"], write=["x"])
    _recording_step("df_source", [], write=["y"])

    with pytest.raises(RuntimeError, match="reads and writes .*'x'.*new key"):
        build_session(tmp_path, {"df_normalize": {}, "df_source": {}})


def test_a_read_nobody_writes_names_the_key_and_what_is_written(tmp_path):
    _recording_step("df_reader", [], read=["logtis"])
    _recording_step("df_writer", [], write=["logits"])

    with pytest.raises(
            RuntimeError,
            match=r"Step.df_reader reads iteration_context key 'logtis'.*"
                  r"Keys written in this session: logits",
    ):
        build_session(tmp_path, {"df_reader": {}, "df_writer": {}})


def test_a_key_a_hook_writes_every_iteration_satisfies_a_step(tmp_path):
    ran = []
    hook("df_clock")(writes("tick")(type("Clock", (_Hook,), {})))
    _recording_step("df_reader", ran, read=["tick"])

    session = build_session(tmp_path, {"df_clock": {}, "df_reader": {}})
    _run_one(session)

    assert ran == ["df_reader"]


def test_a_step_cannot_read_what_a_hook_writes_only_now_and_then(tmp_path):
    hook("df_sometimes")(writes("tick")(
        type("Sometimes", (_Hook,), {"call_every": 5})
    ))
    _recording_step("df_reader", [], read=["tick"])

    with pytest.raises(RuntimeError, match="only every 5 iterations"):
        build_session(tmp_path, {"df_sometimes": {}, "df_reader": {}})


def test_a_hook_read_is_checked_for_a_writer(tmp_path):
    hook("df_reporter")(reads("metric")(type("Reporter", (_Hook,), {})))

    with pytest.raises(RuntimeError, match="Hook.df_reporter reads .*'metric'"):
        build_session(tmp_path, {"df_reporter": {}})

    _recording_step("df_metric", [], write=["metric"])
    build_session(tmp_path, {"df_reporter": {}, "df_metric": {}})


def test_undeclared_steps_keep_their_order(tmp_path):
    ran = []

    @step("df_second")
    @requires_step("df_first")
    class Second(Step):
        def run(self, session):
            ran.append("df_second")

    @step("df_first")
    class First(Step):
        def run(self, session):
            ran.append("df_first")

    _recording_step("df_independent", ran)

    session = build_session(
        tmp_path, {"df_second": {}, "df_independent": {}},
    )
    _run_one(session)

    # The order these three got before dataflow existed.
    assert ran == ["df_first", "df_independent", "df_second"]


def test_a_cycle_through_keys_is_reported_with_its_chain(tmp_path):
    _recording_step("df_a", [], read=["b_out"], write=["a_out"])
    _recording_step("df_b", [], read=["a_out"], write=["b_out"])

    with pytest.raises(RuntimeError) as raised:
        build_session(tmp_path, {"df_a": {}, "df_b": {}})

    message = str(raised.value)
    assert message.startswith("Cyclic dependency detected in the component graph!")
    assert (
        "Step.df_a -> Step.df_b (reads 'b_out') -> Step.df_a (reads 'a_out')"
        in message
    )


def test_a_cycle_mixing_requirements_and_keys_names_both_reasons(tmp_path):
    @step("df_consumer")
    @requires_step("df_producer")
    @writes("feedback")
    class Consumer(Step):
        def run(self, session):
            return 1.0

    _recording_step("df_producer", [], read=["feedback"])

    with pytest.raises(
            RuntimeError,
            match=r"Step.df_consumer -> Step.df_producer \(requires\) -> "
                  r"Step.df_consumer \(reads 'feedback'\)",
    ):
        build_session(tmp_path, {"df_consumer": {}})


def test_a_hand_registered_step_is_checked_when_the_session_is_ordered(tmp_path):
    reader = _recording_step("df_reader", [], read=["missing"])
    session = build_session(tmp_path, {})

    session.add_step(reader())

    with pytest.raises(RuntimeError, match="'missing', which no step or hook"):
        session.execution_graph()


def test_a_step_that_does_not_write_what_it_declared_is_caught(tmp_path):
    @step("df_liar")
    @writes("promised")
    class Liar(Step):
        def run(self, session):
            pass

    session = build_session(tmp_path, {"df_liar": {}})

    with pytest.raises(RuntimeError, match=r"declares it writes .*\['promised'\]"):
        _run_one(session)


def test_the_execution_graph_shows_the_dataflow(tmp_path):
    _recording_step("df_loss", [], read=["logits"], write=["loss"])
    _recording_step("df_head", [], write=["logits"])

    graph = build_session(
        tmp_path, {"df_loss": {}, "df_head": {}},
    ).execution_graph()

    assert "DATAFLOW\n  logits: Step.df_head -> Step.df_loss\n" in graph
    assert "  loss: Step.df_loss -> (not read)" in graph
    assert "Step.df_loss.run() [writes: loss; reads: logits]" in graph


def test_declarations_are_for_steps_and_hooks_only():
    with pytest.raises(TypeError, match="Step or IterationHook"):
        writes("x")(type("R", (Resource,), {}))
    with pytest.raises(TypeError, match="at least one"):
        reads()
    with pytest.raises(ValueError, match="non-empty strings"):
        reads("")
    with pytest.raises(ValueError, match="more than once"):
        writes("x")(writes("x")(type("S", (Step,), {"run": lambda s, x: None})))


# -- iteration phases -----------------------------------------------------------------


class _SessionOnlyHook(SessionHook):
    def pre_session(self, session):
        pass

    def post_session(self, session):
        pass


def test_a_session_hook_cannot_declare_keys():
    with pytest.raises(TypeError, match="Step or IterationHook"):
        writes("x")(type("OnlySession", (_SessionOnlyHook,), {}))


def test_a_session_hook_overriding_its_keys_is_rejected_when_sorted(tmp_path):
    @hook("df_session_writer")
    class Pretender(_SessionOnlyHook):
        def context_writes(self):
            return {"x": "x"}

    _recording_step("df_reader", [], read=["x"])

    with pytest.raises(RuntimeError, match="only steps and iteration hooks"):
        build_session(tmp_path, {"df_session_writer": {}, "df_reader": {}})


def test_a_hook_that_does_not_write_what_it_declared_is_caught(tmp_path):
    @hook("df_silent")
    @writes("tick")
    class Silent(_Hook):
        def pre_iteration_callback(self, session):
            pass

    _recording_step("df_reader", [], read=["tick"])
    session = build_session(tmp_path, {"df_silent": {}, "df_reader": {}})

    with pytest.raises(
            RuntimeError, match=r"Hook.df_silent declares it writes .*\['tick'\]",
    ):
        _run_one(session)


# -- what a rank keeps ------------------------------------------------------------------


def _rank_session(tmp_path):
    """A two-rank session whose rank-zero-only step writes what another reads."""

    @resource("df_model")
    class Model(_InertModel):
        pass

    rank_zero_only(_recording_step("df_rank_zero_writer", [], write=["x"]))
    _recording_step("df_reader", [], read=["x"])
    config = make_config(tmp_path)
    config["session_config"]["show_execution_graph"] = False
    config.update({
        "component_bindings": {"model": "df_model"},
        "df_model": {},
        "ddp": {
            "world_size": 2, "backend": "gloo",
            "master_addr": "127.0.0.1", "master_port": "12355",
        },
        "df_rank_zero_writer": {},
        "df_reader": {},
    })
    return TrainingSession(config)


def test_a_rank_keeps_the_writer_of_what_it_reads(tmp_path):
    session = _rank_session(tmp_path)

    with pytest.warns(RuntimeWarning, match="df_rank_zero_writer"):
        keep = session.rank_parallel_names()

    assert {"df_reader", "df_rank_zero_writer"} <= keep


def test_a_worker_keeps_the_writer_from_the_recorded_state(tmp_path):
    state = _rank_session(tmp_path).get_state()

    with pytest.warns(RuntimeWarning, match="df_rank_zero_writer"):
        worker = load_session_for_worker(state, 1)

    assert {"df_reader", "df_rank_zero_writer"} <= component_names(worker)


def test_a_rank_plan_that_would_not_run_is_rejected_in_the_parent(
        tmp_path, monkeypatch,
):
    session = _rank_session(tmp_path)
    # Stand in for a kind of edge the rank planning fails to follow: the
    # reduced graph is sorted before the plan is returned, so the gap is
    # reported here rather than in a worker the other ranks wait for.
    monkeypatch.setattr(
        Edge, "kept_with_source",
        property(lambda edge: edge.kind is not EdgeKind.READS),
    )

    with pytest.raises(RuntimeError, match="cannot run on its own.*'x'"):
        session.rank_parallel_names()


def test_a_hook_cannot_read_from_a_hook_that_runs_less_often(tmp_path):
    hook("df_every_five")(writes("tick")(
        type("EveryFive", (_Hook,), {"call_every": 5})
    ))
    hook("df_every_one")(reads("tick")(type("EveryOne", (_Hook,), {})))

    with pytest.raises(
            RuntimeError,
            match="writes only every 5 iterations, but it runs every 1.*"
                  "multiple of 5",
    ):
        build_session(tmp_path, {"df_every_five": {}, "df_every_one": {}})


def test_a_hook_reading_on_a_multiple_of_its_writer_cadence_always_finds_the_key(
        tmp_path,
):
    seen = []
    hook("df_every_two")(writes("tick")(
        type("EveryTwo", (_Hook,), {"call_every": 2})
    ))

    @hook("df_every_four")
    @reads("tick")
    class EveryFour(_Hook):
        call_every = 4

        def post_iteration_callback(self, session, tick):
            seen.append((session.iteration, tick))

    session = build_session(
        tmp_path, {"df_every_two": {}, "df_every_four": {}},
    )
    session.update_max_iters(9)
    with session:
        list(session)

    assert [iteration for iteration, _ in seen] == [1, 4, 8, 9]


@pytest.mark.parametrize("cadence", [0, -1, "5", 2.0, True])
def test_an_invalid_writer_cadence_is_reported_when_the_session_is_built(
        tmp_path, cadence,
):
    hook("df_writer")(writes("tick")(
        type("Writer", (_Hook,), {"call_every": cadence})
    ))
    hook("df_reader")(reads("tick")(type("Reader", (_Hook,), {})))

    with pytest.raises(
            RuntimeError,
            match="Hook.df_writer call_every must be a positive integer",
    ):
        build_session(tmp_path, {"df_writer": {}, "df_reader": {}})


def test_a_hook_without_a_cadence_is_reported_when_the_session_is_built(tmp_path):
    @hook("df_no_cadence")
    @writes("tick")
    class NoCadence(LifecycleHook):
        def pre_session(self, session):
            pass

        def post_session(self, session):
            pass

        def pre_iteration_callback(self, session):
            return 1.0

        def post_iteration_callback(self, session):
            pass

    _recording_step("df_reader", [], read=["tick"])

    with pytest.raises(RuntimeError, match="call_every must be a positive integer; got <missing>"):
        build_session(tmp_path, {"df_no_cadence": {}, "df_reader": {}})


def test_an_invalid_cadence_is_reported_for_a_hook_that_neither_wraps_nor_reads(
        tmp_path,
):
    hook("df_lonely")(type("Lonely", (_Hook,), {"call_every": 0}))

    with pytest.raises(RuntimeError, match="Hook.df_lonely call_every must be a positive integer; got 0"):
        build_session(tmp_path, {"df_lonely": {}})


# -- values passed in and returned --------------------------------------------------


def _writer(name, keys, value):
    """A step declaring `keys` whose `run` returns `value`."""
    return step(name)(writes(*keys)(
        type(name, (Step,), {"run": lambda self, session: value})
    ))


def _capture(name, keys):
    """A step reading `keys`; returns the list its `run` appends them to."""
    seen = []
    step(name)(reads(*keys)(type(name, (Step,), {
        "run": lambda self, session, **values: seen.append(values),
    })))
    return seen


def test_declared_reads_arrive_as_keyword_arguments(tmp_path):
    seen = []
    _writer("df_head", ["logits"], 3.0)

    @step("df_loss")
    @reads("logits")
    class Loss(Step):
        def run(self, session, logits):
            seen.append(logits)

    _run_one(build_session(tmp_path, {"df_loss": {}, "df_head": {}}))

    assert seen == [3.0]


def test_a_key_from_configuration_reaches_a_fixed_parameter_name(tmp_path):
    seen = []
    _writer("df_head", ["total_loss"], 2.0)

    @step("df_backward")
    class Backward(Step):
        def __init__(self, config=None):
            self.key = config["key"]

        def context_reads(self):
            return {"loss": self.key}

        def run(self, session, loss):
            seen.append(loss)

    session = build_session(
        tmp_path, {"df_backward": {"key": "total_loss"}, "df_head": {}},
    )
    _run_one(session)

    assert seen == [2.0]
    assert "reads: total_loss (as loss)" in session.execution_graph()


@pytest.mark.parametrize(
    "returned",
    [(1.0, 2.0), {"b": 2.0, "a": 1.0}],
    ids=["tuple in declaration order", "mapping by name"],
)
def test_several_writes_are_returned_as_a_tuple_or_a_mapping(tmp_path, returned):
    _writer("df_pair", ["a", "b"], returned)
    seen = _capture("df_reader", ["a", "b"])

    _run_one(build_session(tmp_path, {"df_pair": {}, "df_reader": {}}))

    assert seen == [{"a": 1.0, "b": 2.0}]


@pytest.mark.parametrize(
    "value", [(1.0, 2.0), {"x": 1.0}], ids=["tuple", "mapping"],
)
def test_a_single_write_stores_the_returned_value_whole(tmp_path, value):
    _writer("df_single", ["x"], value)
    seen = _capture("df_reader", ["x"])

    _run_one(build_session(tmp_path, {"df_single": {}, "df_reader": {}}))

    assert seen == [{"x": value}]


@pytest.mark.parametrize(
    "returned, shown",
    [
        ([1.0, 2.0], "a list"),
        ((1.0,), "a tuple of 1"),
        ({"a": 1.0}, r"a mapping of \['a'\]"),
        ({"a": 1.0, "b": 2.0, "c": 3.0}, r"a mapping of \['a', 'b', 'c'\]"),
    ],
)
def test_several_writes_returned_in_another_shape_are_refused(
        tmp_path, returned, shown,
):
    _writer("df_pair", ["a", "b"], returned)

    session = build_session(tmp_path, {"df_pair": {}})

    with pytest.raises(
            RuntimeError,
            match=rf"Step.df_pair.run declares writing \['a', 'b'\].* "
                  rf"it returned {shown}",
    ):
        _run_one(session)


def test_a_declared_output_returned_as_none_is_refused(tmp_path):
    _writer("df_pair", ["a", "b"], (1.0, None))

    session = build_session(tmp_path, {"df_pair": {}})

    with pytest.raises(RuntimeError, match=r"returned None for \['b'\]"):
        _run_one(session)


def test_a_declared_key_written_directly_is_refused(tmp_path):
    @step("df_direct")
    @writes("x")
    class Direct(Step):
        def run(self, session):
            session.iteration_context["x"] = 1.0
            return 1.0

    session = build_session(tmp_path, {"df_direct": {}})

    with pytest.raises(RuntimeError, match="'x'.*already written.*return the value"):
        _run_one(session)


def test_a_value_returned_without_declared_writes_is_refused(tmp_path):
    @step("df_forgetful")
    class Forgetful(Step):
        def run(self, session):
            return 1.0

    session = build_session(tmp_path, {"df_forgetful": {}})

    with pytest.raises(RuntimeError, match="declares no iteration_context writes"):
        _run_one(session)


def test_a_hook_post_callback_returning_a_value_is_refused(tmp_path):
    @hook("df_chatty")
    class Chatty(_Hook):
        def post_iteration_callback(self, session, **values):
            return 1.0

    session = build_session(tmp_path, {"df_chatty": {}})

    with pytest.raises(RuntimeError, match="post_iteration_callback returned a float"):
        _run_one(session)


def test_a_read_the_callback_does_not_take_is_reported_when_built(tmp_path):
    _writer("df_head", ["logits"], 1.0)

    @step("df_loss")
    @reads("logits")
    class Loss(Step):
        def run(self, session, logit):
            pass

    with pytest.raises(TypeError, match=r"reads \['logits'\], but does not take it"):
        build_session(tmp_path, {"df_loss": {}, "df_head": {}})


def test_a_parameter_nothing_fills_is_reported_when_built(tmp_path):
    @step("df_loss")
    class Loss(Step):
        def run(self, session, logits):
            pass

    with pytest.raises(TypeError, match=r"parameters \['logits'\] that nothing fills"):
        build_session(tmp_path, {"df_loss": {}})


def test_a_key_that_is_not_an_identifier_needs_keyword_arguments(tmp_path):
    _writer("df_head", ["head/out"], 1.0)
    seen = _capture("df_reader", ["head/out"])

    @step("df_strict")
    @reads("head/out")
    class Strict(Step):
        def run(self, session):
            pass

    _run_one(build_session(tmp_path, {"df_head": {}, "df_reader": {}}))
    assert seen == [{"head/out": 1.0}]

    with pytest.raises(TypeError, match="or \\*\\*kwargs"):
        build_session(tmp_path, {"df_head": {}, "df_strict": {}})


def test_keys_from_configuration_must_be_a_mapping(tmp_path):
    @step("df_old_style")
    class OldStyle(Step):
        def context_reads(self):
            return ("x",)

        def run(self, session, **values):
            pass

    with pytest.raises(TypeError, match="must return a mapping of name"):
        build_session(tmp_path, {"df_old_style": {}})


@pytest.mark.parametrize("key", ["session", "self"])
def test_a_read_named_like_a_parameter_the_session_fills_is_reported(
        tmp_path, key,
):
    _writer("df_head", [key], 1.0)
    step("df_reader")(reads(key)(type("Reader", (Step,), {
        "run": lambda self, session, **values: None,
    })))

    with pytest.raises(
            TypeError,
            match=rf"reads \['{key}'\], which is also the name of a "
                  "parameter the session fills positionally",
    ):
        build_session(tmp_path, {"df_head": {}, "df_reader": {}})


def test_a_read_named_like_a_positional_only_parameter_is_passed(tmp_path):
    seen = []
    _writer("df_head", ["session"], 1.0)

    @step("df_reader")
    @reads("session")
    class Reader(Step):
        def run(self, session, /, **values):
            seen.append(values)

    _run_one(build_session(tmp_path, {"df_head": {}, "df_reader": {}}))

    assert seen == [{"session": 1.0}]
