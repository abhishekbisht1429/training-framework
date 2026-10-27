"""The torchvision dataset resources: config, samples, transforms, a session.

ImageNet and iNaturalist read real (tiny) folders written here. CIFAR-10,
Flowers-102 and Stanford Cars verify their files' checksums, so torchvision's
class is replaced by a stand-in that records what it was asked for.
"""

from __future__ import annotations

import importlib
import pickle

import pytest

torchvision = pytest.importorskip("torchvision")

import torch
from PIL import Image

from tests.test_utils import (
    build_session,
    component_named,
    make_config,
    resource_named,
    stub_process_group,
)
from torch import nn

from training_framework.components import (
    StatefulResource,
    Step,
    reads,
    resource,
    step,
    writes,
)
from training_framework.components.builtin.datasets import (
    CIFAR10,
    Flowers102,
    ImageNet,
    INaturalist,
    StanfordCars,
)
from training_framework.session import TrainingSession


def _write_image(path, shade: int, size=(40, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (shade, shade, shade)).save(path)


def _image_folders(root, split="train", classes=("n02", "n01"), per_class=2):
    """`root/<split>/<class>/<image>`, written in an unsorted order."""
    for class_index, name in enumerate(classes):
        for image_index in reversed(range(per_class)):
            _write_image(
                root / split / name / f"img{image_index}.png",
                shade=40 * class_index + 10 * image_index,
            )
    return root


class _FakeSource:
    """Stands in for a torchvision dataset class and records its kwargs."""

    calls: list[dict] = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)
        self.classes = ["a", "b", "c"]

    def __len__(self):
        return 3

    def __getitem__(self, index):
        return Image.new("RGB", (32, 32), (index * 50,) * 3), index


@pytest.fixture
def fake_sources(monkeypatch):
    _FakeSource.calls = []
    for name in ("CIFAR10", "Flowers102", "StanfordCars"):
        monkeypatch.setattr(torchvision.datasets, name, _FakeSource)
    return _FakeSource.calls


def augment(image):
    """A transform named by dotted path in the tests below."""
    return torch.full((1,), 7.0)


class NotATransformInstance:
    pass


# -- configuration ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("dataset", "config", "message"),
    [
        (CIFAR10, {"split": "val", "transform": "eval"},
         r"split must be one of \['train', 'test'\]; got 'val'"),
        (Flowers102, {"split": "train", "transform": "augmented"},
         "transform must be one of"),
        (Flowers102, {"split": "train"}, "transform"),
        (StanfordCars, {"split": "train", "transform": "eval", "download": "yes"},
         "download must be a boolean"),
        (CIFAR10, {"split": "train", "transform": "eval", "image_size": 0},
         "image_size must be a positive integer"),
        (CIFAR10, {"split": "train", "transform": "a.b", "image_size": 64},
         "drop image_size"),
        (ImageNet, {"split": "train", "transform": "eval", "download": True},
         "download is not supported"),
        (INaturalist, {"split": "2020", "transform": "eval"},
         "split must be one of"),
    ],
)
def test_invalid_configuration_is_reported_before_any_file_is_read(
        tmp_path, dataset, config, message,
):
    with pytest.raises((TypeError, ValueError), match=message):
        dataset({"root": str(tmp_path / "missing"), **config})


def test_each_split_reaches_torchvision_as_its_own_argument(tmp_path, fake_sources):
    root = str(tmp_path)
    CIFAR10({"root": root, "split": "train", "transform": "eval"})
    CIFAR10({"root": root, "split": "test", "transform": "eval"})
    Flowers102({"root": root, "split": "val", "transform": "eval", "download": True})
    StanfordCars({"root": root, "split": "test", "transform": "eval"})

    assert fake_sources == [
        {"root": root, "train": True, "download": False},
        {"root": root, "train": False, "download": False},
        {"root": root, "split": "val", "download": True},
        {"root": root, "split": "test", "download": False},
    ]


# -- samples and transforms ---------------------------------------------------------


def test_imagenet_reads_classes_and_images_in_sorted_order(tmp_path):
    dataset = ImageNet({
        "root": str(_image_folders(tmp_path)),
        "split": "train",
        "transform": "eval",
    })

    assert dataset.labels == ["n01", "n02"]
    assert dataset.num_classes == 2
    assert len(dataset) == 4
    # Folders were written n02 first and images in reverse; the order is
    # sorted regardless, so an index means one image on every machine.
    assert [dataset[i][1] for i in range(4)] == [0, 0, 1, 1]
    darker, lighter = dataset[0][0], dataset[1][0]
    assert darker.mean() < lighter.mean()


def test_imagenet_names_its_classes_at_construction_and_lists_images_on_first_use(
        tmp_path,
):
    root = _image_folders(tmp_path)
    dataset = ImageNet({"root": str(root), "split": "train", "transform": "eval"})

    # Written after construction: still counted, so the images were not
    # listed then -- the launching process never walks the whole tree.
    _write_image(root / "train" / "n01" / "late.png", shade=0)

    assert dataset.num_classes == 2
    assert len(dataset) == 5


def test_imagenet_refuses_class_folders_changed_after_construction(tmp_path):
    root = _image_folders(tmp_path)
    dataset = ImageNet({"root": str(root), "split": "train", "transform": "eval"})

    _write_image(root / "train" / "n00" / "img0.png", shade=0)

    with pytest.raises(RuntimeError, match="class folders .* changed"):
        len(dataset)


def test_the_eval_preset_is_deterministic_and_normalized(tmp_path):
    dataset = ImageNet({
        "root": str(_image_folders(tmp_path)),
        "split": "train",
        "transform": "eval",
        "image_size": 16,
    })

    image, label = dataset[0]  # n01/img0, a uniform shade of 40

    assert image.shape == (3, 16, 16)
    assert image.dtype == torch.float32
    assert isinstance(label, int)
    # A uniform grey image normalizes to (grey - mean) / std per channel.
    grey = 40 / 255
    expected = [(grey - m) / s for m, s in zip(dataset.mean, dataset.std)]
    torch.testing.assert_close(
        image.mean(dim=(1, 2)), torch.tensor(expected), atol=1e-3, rtol=0,
    )


def _two_tone_folder(root):
    """One image, half black and half white: a random crop or flip of it
    changes its pixels."""
    image = Image.new("RGB", (64, 48))
    image.paste((255, 255, 255), (32, 0, 64, 48))
    (root / "train" / "a").mkdir(parents=True)
    image.save(root / "train" / "a" / "img.png")
    return root


@pytest.mark.parametrize(("preset", "varies"), [("train", True), ("eval", False)])
def test_only_the_train_preset_is_random(tmp_path, preset, varies):
    dataset = ImageNet({
        "root": str(_two_tone_folder(tmp_path)),
        "split": "train",
        "transform": preset,
        "image_size": 24,
    })

    torch.manual_seed(0)
    samples = [dataset[0][0] for _ in range(8)]

    assert all(sample.shape == (3, 24, 24) for sample in samples)
    assert any(not torch.equal(samples[0], s) for s in samples[1:]) is varies


def test_cifar10_keeps_its_native_size_unless_asked(tmp_path, fake_sources):
    native = CIFAR10({"root": str(tmp_path), "split": "train", "transform": "train"})
    larger = CIFAR10({
        "root": str(tmp_path), "split": "test", "transform": "eval", "image_size": 64,
    })

    assert native[1][0].shape == (3, 32, 32)
    assert larger[1][0].shape == (3, 64, 64)
    assert native.labels == ["a", "b", "c"]


def _grey_channel_means(dataset):
    """The per-channel mean of sample 0: n01/img0, a uniform shade of 40."""
    return dataset[0][0].mean(dim=(1, 2))


@pytest.mark.parametrize(
    ("statistics", "mean", "std"),
    [
        ({}, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ({"mean": [0.2, 0.3, 0.4], "std": [0.1, 0.2, 0.25]},
         (0.2, 0.3, 0.4), (0.1, 0.2, 0.25)),
        ({"mean": "imagenet", "std": "imagenet"},
         (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ({"mean": "cifar10", "std": "cifar10"},
         (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ],
    ids=["default", "listed", "imagenet", "cifar10"],
)
def test_presets_normalize_with_the_configured_statistics(
        tmp_path, statistics, mean, std,
):
    dataset = ImageNet({
        "root": str(_image_folders(tmp_path)),
        "split": "train",
        "transform": "eval",
        "image_size": 8,
        **statistics,
    })

    assert dataset.mean == mean
    assert dataset.std == std
    grey = 40 / 255
    torch.testing.assert_close(
        _grey_channel_means(dataset),
        torch.tensor([(grey - m) / s for m, s in zip(mean, std)]),
        atol=1e-3,
        rtol=0,
    )


def test_every_dataset_defaults_to_half_normalization(tmp_path, fake_sources):
    root = str(tmp_path)
    for dataset in (
            CIFAR10({"root": root, "split": "train", "transform": "eval"}),
            Flowers102({"root": root, "split": "train", "transform": "train"}),
            StanfordCars({"root": root, "split": "test", "transform": "eval"}),
    ):
        assert (dataset.mean, dataset.std) == ((0.5,) * 3, (0.5,) * 3)
    # Sample 0 of the stand-in is black: (0 - 0.5) / 0.5 on every channel.
    black = CIFAR10({"root": root, "split": "test", "transform": "eval"})[0][0]
    torch.testing.assert_close(black, torch.full_like(black, -1.0))


@pytest.mark.parametrize(
    ("statistics", "message"),
    [
        ({"mean": [0.5, 0.5, 0.5]}, "must be set together"),
        ({"std": "imagenet"}, "must be set together"),
        ({"mean": [0.5, 0.5], "std": [0.5, 0.5, 0.5]},
         "mean must be three numbers"),
        ({"mean": [0.5, 0.5, 0.5], "std": [0.5, True, 0.5]},
         "std must be three numbers"),
        ({"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.0, 0.5]},
         "greater than 0"),
        ({"mean": "imagenet", "std": "cifar10"}, "same statistics"),
        ({"mean": "imagenet", "std": [0.5, 0.5, 0.5]}, "same statistics"),
        ({"mean": "coco", "std": "coco"}, "same statistics"),
        ({"transform": "a.b", "mean": "imagenet", "std": "imagenet"},
         "drop them"),
    ],
)
def test_invalid_normalization_is_reported(tmp_path, statistics, message):
    config = {"root": str(tmp_path), "split": "train", "transform": "eval"}
    with pytest.raises(ValueError, match=message):
        ImageNet({**config, **statistics})


def test_a_transform_named_by_dotted_path_replaces_the_preset(tmp_path):
    dataset = ImageNet({
        "root": str(_image_folders(tmp_path)),
        "split": "train",
        "transform": f"{__name__}.augment",
    })

    image, label = dataset[0]

    assert dataset.transform is augment
    torch.testing.assert_close(image, torch.full((1,), 7.0))
    assert label == 0


@pytest.mark.parametrize(
    ("path", "error", "message"),
    [
        (f"{__name__}.NotATransformInstance", TypeError, "not a class"),
        (f"{__name__}.missing_transform", ValueError, "has no attribute"),
        ("no_such_module.transform", ImportError, "could not be imported"),
    ],
)
def test_a_dotted_transform_that_cannot_be_used_is_reported(
        tmp_path, path, error, message,
):
    with pytest.raises(error, match=message):
        ImageNet({
            "root": str(_image_folders(tmp_path)),
            "split": "train",
            "transform": path,
        })


def test_inaturalist_labels_are_the_species_folders(tmp_path):
    species = [
        "00000_Animalia_Chordata_Aves_Order_Family_Genus_alpha",
        "00001_Plantae_Tracheophyta_Magnoliopsida_Order_Family_Genus_beta",
    ]
    for index, name in enumerate(species):
        _write_image(tmp_path / "2021_valid" / name / "img.jpg", shade=index * 90)
    dataset = INaturalist({
        "root": str(tmp_path), "split": "2021_valid", "transform": "eval",
        "image_size": 8,
    })

    assert dataset.labels == species
    assert [dataset[i][1] for i in range(len(dataset))] == [0, 1]
    assert dataset[1][0].shape == (3, 8, 8)


# -- in a session ----------------------------------------------------------------


def test_a_training_session_loads_labelled_image_batches(tmp_path, monkeypatch):
    stub_process_group(monkeypatch)
    # The registry is reset around every test; reloading the module registers
    # `imagenet` again, as importing a components package does in a run.
    importlib.reload(
        importlib.import_module("training_framework.components.builtin.datasets.imagenet")
    )
    seen = []

    @resource("dataset_test_model")
    class Model(nn.Module, StatefulResource):
        def __init__(self, config=None):
            nn.Module.__init__(self)
            self.linear = nn.Linear(1, 1)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def get_state(self):
            return {}

        def set_state(self, state):
            pass

    @reads("images", "labels")
    @step("dataset_recorder")
    class Recorder(Step):
        def run(self, session, *, images, labels):
            seen.append((images, labels))

    config = make_config(tmp_path, max_iterations=2)
    config["session_config"]["show_execution_graph"] = False
    config.update({
        "component_bindings": {
            "dataset": "imagenet#train", "model": "dataset_test_model",
        },
        "dataset_test_model": {},
        "imagenet#train": {
            "root": str(_image_folders(tmp_path / "imagenet", per_class=3)),
            "split": "train",
            "transform": "eval",
            "image_size": 8,
        },
        "ddp": {
            "world_size": 1, "backend": "gloo",
            "master_addr": "localhost", "master_port": "12355",
        },
        "data_manager": {"batch_size": 4, "num_workers": 0, "pin_memory": False},
        "load_batch": {"fields": ["images", "labels"]},
        "dataset_recorder": {},
    })
    session = TrainingSession(config)
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")
    placeholder = resource_named(session, "ddp")
    session.unregister_resource("ddp")
    session.register_resource(type(placeholder)(config=placeholder.config, rank=0))

    with session:
        list(session)

    assert len(seen) == 2
    for images, labels in seen:
        assert images.shape == (4, 3, 8, 8)
        assert labels.dtype == torch.int64
        assert set(labels.tolist()) <= {0, 1}


def test_the_optional_component_table_lists_exactly_these_datasets():
    from training_framework.components.builtin import datasets
    from training_framework.components.optional import (
        OPTIONAL_COMPONENTS,
        VISION_DATASETS,
    )

    registered = {
        exported.name
        for exported in map(lambda name: getattr(datasets, name), datasets.__all__)
        if "name" in vars(exported)
    }

    assert VISION_DATASETS.module == datasets.__name__
    assert registered == {
        name for name, package in OPTIONAL_COMPONENTS.items()
        if package is VISION_DATASETS
    }


@pytest.mark.parametrize(
    ("statistics", "message"),
    [
        ("mean: [.nan, 0.5, 0.5]\nstd: [0.5, 0.5, 0.5]",
         r"mean values must be finite numbers; got \[nan, 0.5, 0.5\]"),
        ("mean: [0.5, .inf, 0.5]\nstd: [0.5, 0.5, 0.5]",
         r"mean values must be finite numbers; got \[0.5, inf, 0.5\]"),
        ("mean: [0.5, 0.5, 0.5]\nstd: [0.5, .nan, 0.5]",
         r"std values must be finite numbers"),
        ("mean: [0.5, 0.5, 0.5]\nstd: [0.5, 0.5, .inf]",
         r"std values must be finite numbers"),
    ],
)
def test_non_finite_normalization_from_yaml_is_refused(tmp_path, statistics, message):
    import yaml

    config = {"root": str(tmp_path), "split": "train", "transform": "eval"}
    with pytest.raises(ValueError, match=message):
        ImageNet({**config, **yaml.safe_load(statistics)})


# -- serialization ------------------------------------------------------------------

#: Registered name -> (module in the datasets package, class name).
_DATASET_MODULES = {
    "cifar10": ("cifar10", "CIFAR10"),
    "flowers102": ("flowers102", "Flowers102"),
    "stanford_cars": ("stanford_cars", "StanfordCars"),
    "inaturalist": ("inaturalist", "INaturalist"),
    "imagenet": ("imagenet", "ImageNet"),
}


def _dataset_config(name, root):
    """A config for `name` whose data exists: folders on disk for the two
    folder datasets, the stand-in source (see `fake_sources`) for the rest."""
    if name == "imagenet":
        return {"root": str(_image_folders(root)), "split": "train"}
    if name == "inaturalist":
        for index in range(2):
            _write_image(
                root / "2021_valid"
                / f"0000{index}_K_P_C_O_F_G_s{index}" / "img.jpg",
                shade=index * 90,
            )
        return {"root": str(root), "split": "2021_valid"}
    split = {"cifar10": "test", "flowers102": "val", "stanford_cars": "test"}[name]
    return {"root": str(root), "split": split}


@pytest.fixture
def registered_datasets():
    """Register the dataset resources again: the registry is reset around
    every test, as importing a components package does in a run."""
    for module, _ in _DATASET_MODULES.values():
        importlib.reload(importlib.import_module(
            f"training_framework.components.builtin.datasets.{module}"
        ))


def _dataset_class(name):
    """The class as its module now defines it -- after a reload, the one
    pickle finds by name."""
    module, class_name = _DATASET_MODULES[name]
    return getattr(importlib.import_module(
        f"training_framework.components.builtin.datasets.{module}"
    ), class_name)


def _same_dataset(restored, original):
    assert type(restored).__name__ == type(original).__name__
    assert restored.split == original.split
    assert restored.labels == original.labels
    assert len(restored) == len(original)
    assert (restored.mean, restored.std) == (original.mean, original.std)
    for index in range(len(original)):
        image, label = restored[index]
        expected_image, expected_label = original[index]
        torch.testing.assert_close(image, expected_image)
        assert label == expected_label


@pytest.mark.parametrize("name", sorted(_DATASET_MODULES))
def test_a_pickled_dataset_reads_the_same_samples(
        tmp_path, fake_sources, registered_datasets, name,
):
    original = _dataset_class(name)({
        **_dataset_config(name, tmp_path),
        "transform": "eval",
        "image_size": 8,
        "mean": "imagenet",
        "std": "imagenet",
    })

    _same_dataset(pickle.loads(pickle.dumps(original)), original)


@pytest.mark.parametrize("name", sorted(_DATASET_MODULES))
def test_a_checkpointed_session_restores_its_dataset_instance(
        tmp_path, monkeypatch, fake_sources, registered_datasets, name,
):
    from training_framework.components.builtin import Checkpointer

    stub_process_group(monkeypatch)

    @resource("dataset_test_model")
    class Model(nn.Module, StatefulResource):
        def __init__(self, config=None):
            nn.Module.__init__(self)
            self.linear = nn.Linear(1, 1)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def get_state(self):
            return {}

        def set_state(self, state):
            pass

    instance = f"{name}#train_split"
    config = make_config(tmp_path / "session")
    config["session_config"]["show_execution_graph"] = False
    config.update({
        "component_bindings": {
            "dataset": instance, "model": "dataset_test_model",
        },
        "dataset_test_model": {},
        instance: {
            **_dataset_config(name, tmp_path / "data"),
            "transform": "eval",
            "image_size": 8,
            "mean": [0.1, 0.2, 0.3],
            "std": [0.4, 0.5, 0.6],
        },
        "ddp": {
            "world_size": 1, "backend": "gloo",
            "master_addr": "localhost", "master_port": "12355",
        },
        "data_manager": {"batch_size": 1, "num_workers": 0, "pin_memory": False},
    })
    session = TrainingSession(config)
    original = resource_named(session, instance)

    path = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")
    restored = resource_named(Checkpointer.load_checkpoint(path), instance)

    assert restored is not original
    assert restored.name == instance
    _same_dataset(restored, original)


# -- uint8 images, normalized on the device -------------------------------------------


def _noise_folders(root):
    """Two classes of two random-noise images: every pixel value differs."""
    generator = torch.Generator().manual_seed(0)
    for name in ("n01", "n02"):
        for index in range(2):
            path = root / "train" / name / f"img{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = torch.randint(0, 256, (30, 40, 3), generator=generator)
            Image.fromarray(pixels.to(torch.uint8).numpy()).save(path)
    return root


def _noise_config(root, **overrides):
    return {
        "root": str(_noise_folders(root)),
        "split": "train",
        "transform": "eval",
        "image_size": 16,
        "mean": "imagenet",
        "std": "imagenet",
        **overrides,
    }


@writes("pixels")
@step("pixel_source")
class PixelSource(Step):
    """Writes the key `normalize_images` reads; the tests call that step
    directly, so this never runs."""

    def run(self, session):
        raise AssertionError("not run")


def _normalizing_session(tmp_path, dataset_config):
    for module in ("imagenet", "normalization"):
        importlib.reload(importlib.import_module(
            f"training_framework.components.builtin.datasets.{module}"
        ))
    step("pixel_source")(PixelSource)
    return build_session(
        tmp_path / "session",
        {
            "imagenet#train": dataset_config,
            "pixel_source": {},
            "normalize_images": {"input": "pixels", "output": "images"},
        },
        bindings={"dataset": "imagenet#train"},
    )


@pytest.mark.parametrize("preset", ["train", "eval"])
def test_a_uint8_preset_returns_the_pixels_before_scaling(tmp_path, preset):
    dataset = ImageNet(_noise_config(
        tmp_path, transform=preset, image_dtype="uint8",
    ))

    image, label = dataset[0]

    assert dataset.image_dtype == "uint8"
    assert image.dtype == torch.uint8
    assert image.shape == (3, 16, 16)
    assert isinstance(label, int)


def test_image_dtype_defaults_to_float32_and_is_none_for_a_dotted_transform(
        tmp_path,
):
    root = str(_image_folders(tmp_path))
    preset = ImageNet({"root": root, "split": "train", "transform": "eval"})
    dotted = ImageNet({
        "root": root, "split": "train",
        "transform": f"{__name__}.augment",
    })

    assert preset.image_dtype == "float32"
    assert dotted.image_dtype is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"transform": "eval", "image_dtype": "float16"},
         r"image_dtype must be one of \['float32', 'uint8'\]"),
        ({"transform": "a.b", "image_dtype": "uint8"}, "drop image_dtype"),
    ],
)
def test_invalid_image_dtype_is_reported(tmp_path, config, message):
    with pytest.raises(ValueError, match=message):
        ImageNet({"root": str(tmp_path), "split": "train", **config})


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="needs a GPU",
        )),
    ],
)
def test_normalize_images_matches_the_float32_preset_bit_for_bit(
        tmp_path, device,
):
    config = _noise_config(tmp_path)
    session = _normalizing_session(tmp_path, {**config, "image_dtype": "uint8"})
    normalize = component_named(session, "normalize_images")
    uint8_dataset = ImageNet({**config, "image_dtype": "uint8"})
    float_dataset = ImageNet(config)
    pixels = torch.stack([uint8_dataset[i][0] for i in range(4)])
    expected = torch.stack([float_dataset[i][0] for i in range(4)])

    images = normalize.run(session, pixels=pixels.to(device))

    assert images.dtype == torch.float32
    assert images.device.type == device
    assert torch.equal(images.cpu(), expected)


@pytest.mark.parametrize(
    ("dataset_overrides", "message"),
    [
        ({}, "returns float32 images.*set image_dtype: uint8"),
        ({"transform": f"{__name__}.augment", "image_size": None,
          "mean": None, "std": None},
         "images of its own making"),
    ],
)
def test_normalize_images_refuses_a_dataset_without_uint8_images(
        tmp_path, dataset_overrides, message,
):
    config = {
        key: value
        for key, value in {**_noise_config(tmp_path), **dataset_overrides}.items()
        if value is not None
    }
    with pytest.raises(ValueError, match=message):
        _normalizing_session(tmp_path, config)


def test_normalize_images_refuses_images_that_are_not_uint8(tmp_path):
    session = _normalizing_session(
        tmp_path, _noise_config(tmp_path, image_dtype="uint8"),
    )
    normalize = component_named(session, "normalize_images")

    with pytest.raises(TypeError, match="must be a uint8 tensor.*float32"):
        normalize.run(session, pixels=torch.zeros(1, 3, 16, 16))


def test_normalize_images_needs_two_different_keys(tmp_path):
    config = _noise_config(tmp_path, image_dtype="uint8")
    for module in ("imagenet", "normalization"):
        importlib.reload(importlib.import_module(
            f"training_framework.components.builtin.datasets.{module}"
        ))

    with pytest.raises(ValueError, match="must be different keys"):
        build_session(
            tmp_path / "session",
            {
                "imagenet#train": config,
                "normalize_images": {"input": "pixels", "output": "pixels"},
            },
            bindings={"dataset": "imagenet#train"},
        )


def test_a_training_session_normalizes_uint8_batches_after_loading_them(
        tmp_path, monkeypatch,
):
    stub_process_group(monkeypatch)
    for module in ("imagenet", "normalization"):
        importlib.reload(importlib.import_module(
            f"training_framework.components.builtin.datasets.{module}"
        ))
    seen = []

    @resource("uint8_test_model")
    class Model(nn.Module, StatefulResource):
        def __init__(self, config=None):
            nn.Module.__init__(self)
            self.linear = nn.Linear(1, 1)

        def setup(self, session):
            pass

        def teardown(self, session):
            pass

        def get_state(self):
            return {}

        def set_state(self, state):
            pass

    @reads("pixels", "images")
    @step("uint8_recorder")
    class Recorder(Step):
        def run(self, session, *, pixels, images):
            seen.append((pixels, images))

    config = make_config(tmp_path, max_iterations=2)
    config["session_config"]["show_execution_graph"] = False
    config.update({
        "role_bindings": {
            "dataset": "imagenet#train", "model": "uint8_test_model",
        },
        "uint8_test_model": {},
        "imagenet#train": _noise_config(tmp_path / "imagenet", image_dtype="uint8"),
        "ddp": {
            "world_size": 1, "backend": "gloo",
            "master_addr": "localhost", "master_port": "12355",
        },
        "data_manager": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
        "load_batch": {"fields": ["pixels", "labels"]},
        "normalize_images": {"input": "pixels", "output": "images"},
        "uint8_recorder": {},
    })
    session = TrainingSession(config)
    session.unregister_hook("logger")
    session.unregister_hook("checkpointer")
    placeholder = resource_named(session, "ddp")
    session.unregister_resource("ddp")
    session.register_resource(type(placeholder)(config=placeholder.config, rank=0))

    with session:
        list(session)

    assert len(seen) == 2
    for pixels, images in seen:
        assert pixels.dtype == torch.uint8
        assert images.dtype == torch.float32
        assert images.shape == pixels.shape == (2, 3, 16, 16)


# -- transform: views (self-supervised views) ------------------------------------------


VIEWS = [
    {"count": 2, "size": 24, "scale": [0.4, 1.0], "blur_p": 1.0},
    {"count": 3, "size": 12, "scale": [0.05, 0.4], "blur_p": 0.5, "solarize_p": 0.2},
]


def _views_config(root, **overrides):
    return _noise_config(
        root, **{"transform": "views", "image_size": None, "views": VIEWS, **overrides},
    )


def _without_none(config):
    return {key: value for key, value in config.items() if value is not None}


def test_views_returns_each_groups_views_in_order(tmp_path):
    dataset = ImageNet(_without_none(_views_config(tmp_path)))

    views, label = dataset[0]

    assert dataset.num_views == 5
    assert [tuple(view.shape) for view in views] == [(3, 24, 24)] * 2 + [(3, 12, 12)] * 3
    assert all(view.dtype == torch.float32 for view in views)
    assert isinstance(label, int)


def test_views_are_random_and_differ_from_each_other(tmp_path):
    dataset = ImageNet(_without_none(_views_config(tmp_path)))

    views, _ = dataset[0]
    again, _ = dataset[0]

    assert not torch.equal(views[0], views[1])
    assert not torch.equal(views[0], again[0])


def test_views_collate_into_one_batch_per_view(tmp_path):
    dataset = ImageNet(_without_none(_views_config(tmp_path)))

    views, labels = torch.utils.data.default_collate([dataset[0], dataset[1]])

    assert [tuple(view.shape) for view in views] == (
        [(2, 3, 24, 24)] * 2 + [(2, 3, 12, 12)] * 3
    )
    assert labels.shape == (2,)


def test_other_transforms_have_no_views(tmp_path):
    dataset = ImageNet({"root": str(_image_folders(tmp_path)), "split": "train",
                        "transform": "eval"})

    assert dataset.num_views is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"transform": "train"}, "views describes the views of transform: views"),
        ({"views": None}, "transform: views needs views"),
        ({"views": []}, "transform: views needs views"),
        ({"image_size": 16}, "each group sets its own size, so drop image_size"),
        ({"views": [{"size": 8}]}, r"views\[0\] needs \['scale'\]"),
        ({"views": [{"size": 8, "scale": [0.5, 1], "crop": 1}]},
         r"unknown keys \['crop'\]"),
        ({"views": [{"size": 8, "scale": [0.6, 0.5]}]}, "0 < low <= high"),
        ({"views": [{"size": 8, "scale": [0, 0.5]}]}, "0 < low <= high"),
        ({"views": [{"size": 8, "scale": [0.5]}]}, r"scale must be \[low, high\]"),
        ({"views": [{"size": 8, "scale": [0.5, 1], "count": 0}]},
         "count must be a positive integer"),
        ({"views": [{"size": 8, "scale": [0.5, 1], "blur_p": 2}]},
         "blur_p must be a number between 0.0 and 1.0"),
        ({"views": [{"size": 8, "scale": [0.5, 1],
                     "color_jitter": {"gamma": 0.1}}]}, r"unknown keys \['gamma'\]"),
        ({"views": [{"size": 8, "scale": [0.5, 1],
                     "color_jitter": {"hue": 0.7}}]}, "hue must be a number between"),
    ],
)
def test_invalid_views_are_reported(tmp_path, overrides, message):
    config = {**_views_config(tmp_path), **overrides}
    with pytest.raises(ValueError, match=message):
        ImageNet(_without_none(config))


def test_normalize_images_normalizes_uint8_views_like_the_float32_ones(tmp_path):
    config = _without_none(_views_config(tmp_path))
    session = _normalizing_session(tmp_path, {**config, "image_dtype": "uint8"})
    normalize = component_named(session, "normalize_images")
    uint8_dataset = ImageNet({**config, "image_dtype": "uint8"})
    float_dataset = ImageNet(config)
    torch.manual_seed(5)
    pixels, _ = uint8_dataset[0]
    torch.manual_seed(5)
    expected, _ = float_dataset[0]

    images = normalize.run(session, pixels=[view.unsqueeze(0) for view in pixels])

    assert len(images) == 5
    for image, view in zip(images, expected):
        assert torch.equal(image[0], view)


def test_normalize_images_refuses_an_empty_list_of_views(tmp_path):
    config = _without_none(_views_config(tmp_path, image_dtype="uint8"))
    session = _normalizing_session(tmp_path, config)

    with pytest.raises(TypeError, match="empty list of view batches"):
        component_named(session, "normalize_images").run(session, pixels=[])


def test_a_pickled_views_dataset_reads_the_same_views(tmp_path, registered_datasets):
    dataset = _dataset_class("imagenet")(_without_none(_views_config(tmp_path)))
    copy = pickle.loads(pickle.dumps(dataset))

    torch.manual_seed(9)
    views, _ = dataset[1]
    torch.manual_seed(9)
    copied, _ = copy[1]

    assert all(torch.equal(a, b) for a, b in zip(views, copied))


def test_a_pickled_normalize_images_normalizes_alike(tmp_path):
    session = _normalizing_session(
        tmp_path, _noise_config(tmp_path, image_dtype="uint8"),
    )
    normalize = component_named(session, "normalize_images")
    pixels = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)

    copy = pickle.loads(pickle.dumps(normalize))

    assert torch.equal(
        copy.run(session, pixels=pixels), normalize.run(session, pixels=pixels),
    )


def test_a_checkpointed_session_restores_normalize_images(tmp_path):
    from training_framework.components.builtin import Checkpointer

    session = _normalizing_session(
        tmp_path, _noise_config(tmp_path, image_dtype="uint8"),
    )
    path = Checkpointer.save_checkpoint(session, tmp_path / "checkpoint")
    pixels = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)

    restored = Checkpointer.load_checkpoint(path, restore_rng=False)

    restored_step = component_named(restored, "normalize_images")
    assert torch.equal(
        restored_step.run(restored, pixels=pixels),
        component_named(session, "normalize_images").run(session, pixels=pixels),
    )
