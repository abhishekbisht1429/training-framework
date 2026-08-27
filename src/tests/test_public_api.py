import inspect

import training_framework.components as components
import training_framework.components.registry as registry
import training_framework.engine as engine
import training_framework.session as session


def test_public_classes_have_canonical_domain_module_paths():
    expected_modules = {
        session.Session: "training_framework.session.base",
        session.TrainingSession: "training_framework.session.training",
        session.AnalysisSession: "training_framework.session.analysis",
        engine.TrainingEngine: "training_framework.engine.core",
        engine.SessionProcessWrapper: "training_framework.engine.worker",
    }

    for public_class, expected_module in expected_modules.items():
        assert public_class.__module__ == expected_module


def test_session_constructor_signatures_are_public():
    assert str(inspect.signature(session.Session)) == "(config: dict)"
    assert str(inspect.signature(session.TrainingSession)) == "(config: dict)"
    assert str(inspect.signature(session.AnalysisSession)) == (
        "(config: dict, *, "
        "model_checkpoint_path: str | os.PathLike[str])"
    )


def test_components_package_exports_registry_api_by_identity():
    assert components.resource is registry.resource
    assert components.hook is registry.hook
    assert components.step is registry.step
    assert components.requires_resource is registry.requires_resource
    assert components.requires_hook is registry.requires_hook
    assert components.requires_step is registry.requires_step
    assert components.wraps is registry.wraps


def test_effective_registries_overlay_scoped_components_on_shared_components():
    training_registry = registry.component_registry("training")
    analysis_registry = registry.component_registry("analysis")

    assert training_registry.maps[0] is registry._SESSION_COMPONENT_REGISTRIES["training"]
    assert analysis_registry.maps[0] is registry._ANALYSIS_COMPONENT_REGISTRY
    assert training_registry.maps[1] is registry._SHARED_COMPONENT_REGISTRY
    assert analysis_registry.maps[1] is registry._SHARED_COMPONENT_REGISTRY


def test_session_package_exports_pluggable_session_type_api():
    assert session.TRAINING_SESSION_TYPE == "training"
    assert session.ANALYSIS_SESSION_TYPE == "analysis"
    assert session.normalize_session_type("evaluation") == "evaluation"
    assert callable(session.register_session_type)
    assert callable(session.session_class_for_type)


def test_public_functions_are_defined_in_canonical_modules():
    assert registry.topological_sort_of_components.__module__ == (
        "training_framework.components.registry"
    )
    assert registry.format_execution_graph.__module__ == (
        "training_framework.components.registry"
    )
    assert engine.load_session_for_worker.__module__ == "training_framework.engine.worker"
