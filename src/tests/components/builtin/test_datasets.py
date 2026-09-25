"""The torchvision dataset resources: config, samples, transforms, a session.

ImageNet and iNaturalist read real (tiny) folders written here. CIFAR-10,
Flowers-102 and Stanford Cars verify their files' checksums, so torchvision's
class is replaced by a stand-in that records what it was asked for.
"""

from __future__ import annotations

import importlib

import pytest

torchvision = pytest.importorskip("torchvision")

import torch
from PIL import Image

from tests.test_utils import make_config, resource_named, stub_process_group
from torch import nn

from training_framework.components import StatefulResource, Step, reads, resource, step
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
    expected = [(grey - m) / s for m, s in zip(dataset.MEAN, dataset.STD)]
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
