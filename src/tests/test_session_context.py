from training_framework.training_session import TrainingSession, SessionHook, hook
import pickle



@hook("session_context_writer_hook")
class SessionContextWriterHook(SessionHook):
    def __init__(self):
        self.context_ids = []
        self.seen_in_teardown = []
        self._context = None

    def setup(self, session):
        self._context = session.session_context
        self.context_ids.append(id(self._context))
        self._context["shared_key"] = "shared-value"
        self._context.setdefault("events", []).append("writer_setup")

    def teardown(self, session):
        self.seen_in_teardown.append(self._context["shared_key"])
        self._context.setdefault("events", []).append("writer_teardown")


@hook("session_context_reader_hook")
class SessionContextReaderHook(SessionHook):
    def __init__(self):
        self.context_ids = []
        self.seen_values = []
        self._context = None

    def setup(self, session):
        self._context = session.session_context
        self.context_ids.append(id(self._context))
        self.seen_values.append(self._context.get("shared_key"))
        self._context.setdefault("events", []).append("reader_setup")

    def teardown(self, session):
        self._context.setdefault("events", []).append("reader_teardown")


def test_session_context_is_shared_between_hooks(tmp_path):
    config = {
        "rng_seed": 123,
        "sessions_dir": str(tmp_path),
        "max_iterations": 1,
        "device": "cpu",
    }

    session = TrainingSession(config)
    writer = SessionContextWriterHook()
    reader = SessionContextReaderHook()

    session.register_hook(writer)
    session.register_hook(reader)

    assert session.session_context == {}
    assert session.session_context is session.session_context

    with session:
        assert writer.context_ids[0] == id(session.session_context)
        assert reader.context_ids[0] == id(session.session_context)
        assert reader.seen_values == ["shared-value"]
        assert session.session_context["shared_key"] == "shared-value"
        assert session.session_context["events"][:2] == ["writer_setup", "reader_setup"]

    assert writer.seen_in_teardown == ["shared-value"]


@hook("session_context_seed_hook")
class SessionContextSeedHook(SessionHook):
    def setup(self, session):
        session.session_context["shared_value"] = "hello"
        session.session_context["numbers"] = [1, 2, 3]

    def teardown(self, session):
        pass

def test_session_context_is_saved_restored_and_cleared(tmp_path):
    config = {
        "rng_seed": 123,
        "sessions_dir": str(tmp_path),
        "max_iterations": 1,
        "device": "cpu",
    }

    session = TrainingSession(config)
    session.register_hook(SessionContextSeedHook())

    with session:
        assert session.session_context["shared_value"] == "hello"
        assert session.session_context["numbers"] == [1, 2, 3]

        # Restore the actual session object from its serialized form.
        restored = pickle.loads(pickle.dumps(session))

        assert restored.session_context["shared_value"] == "hello"
        assert restored.session_context["numbers"] == [1, 2, 3]

    # After the session ends, the original session context must be cleared.
    assert session.session_context == {}