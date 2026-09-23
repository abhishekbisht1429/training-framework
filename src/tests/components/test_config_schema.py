"""Tests for the opt-in `config_schema` component configuration parser."""

from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from training_framework.components import ModuleResource, Resource, resource
from training_framework.components import parse_component_config


@dataclass(frozen=True)
class _Schema:
    width: int
    grid_size: tuple = (1, 1)
    label: str = "default"

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError(f"width must be positive; got {self.width}")


class _Configured(Resource):
    name = "configured"
    config_schema = _Schema

    def setup(self, session):
        pass

    def teardown(self, session):
        pass


def test_defaults_are_applied_and_required_keys_are_read():
    parsed = parse_component_config(_Configured, {"width": 4})

    assert parsed.width == 4
    assert parsed.grid_size == (1, 1)
    assert parsed.label == "default"


def test_unknown_keys_name_the_component_and_the_accepted_keys():
    with pytest.raises(ValueError) as raised:
        parse_component_config(_Configured, {"width": 4, "depth": 2})

    message = str(raised.value)
    assert "Invalid configured config" in message
    assert "['depth']" in message
    assert "width" in message and "grid_size" in message


def test_missing_required_keys_are_reported():
    with pytest.raises(ValueError, match=r"missing required keys \['width'\]"):
        parse_component_config(_Configured, {})


def test_a_non_mapping_config_is_rejected():
    with pytest.raises(TypeError, match="config must be a mapping"):
        parse_component_config(_Configured, [1, 2])


def test_post_init_failures_carry_the_component_name():
    with pytest.raises(ValueError, match="Invalid configured config: width"):
        parse_component_config(_Configured, {"width": 0})


def test_sequences_are_coerced_for_tuple_fields():
    parsed = parse_component_config(_Configured, {"width": 2, "grid_size": [3, 5]})

    assert parsed.grid_size == (3, 5)


def test_omegaconf_containers_are_normalized():
    config = OmegaConf.create({"width": 2, "grid_size": [3, 5]})

    parsed = parse_component_config(_Configured, config)

    assert parsed.grid_size == (3, 5)
    assert isinstance(parsed.grid_size, tuple)


def test_a_component_populates_cfg_from_its_schema():
    component = _Configured({"width": 7})

    assert component._cfg.width == 7


def test_a_module_resource_populates_cfg_from_its_schema():
    @dataclass(frozen=True)
    class _Sizes:
        embed_dim: int
        layers: list = field(default_factory=list)

    @resource("schema_module")
    class SchemaModule(ModuleResource):
        config_schema = _Sizes

    component = SchemaModule({"embed_dim": 3})

    assert component._cfg.embed_dim == 3
    assert component._cfg.layers == []
    assert component.config == {"embed_dim": 3}


def test_a_component_without_a_schema_is_untouched():
    class _Plain(Resource):
        name = "plain"

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

    component = _Plain({"anything": 1})

    assert not hasattr(component, "_cfg")


def test_keys_are_not_turned_into_strings():
    # A YAML `1:` where a name was meant stays an unknown key, reported as
    # one, rather than becoming the string "1" some schema might accept.
    with pytest.raises(ValueError, match=r"unknown keys \[1\]"):
        parse_component_config(_Configured, {"width": 4, 1: "x"})
