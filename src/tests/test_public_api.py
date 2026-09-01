import training_framework.session as session


def test_session_package_exports_pluggable_session_type_api():
    assert session.TRAINING_SESSION_TYPE == "training"
    assert session.ANALYSIS_SESSION_TYPE == "analysis"
    assert session.normalize_session_type("evaluation") == "evaluation"
    assert callable(session.register_session_type)
    assert callable(session.session_class_for_type)
