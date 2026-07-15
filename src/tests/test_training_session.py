import os
import sys
from pathlib import Path
from typing import Any, override

import pytest
import yaml
from torch import nn
import torch.nn.functional as F

from training_framework.configurator import Configurator
from training_framework.resources import Checkpointer, Tensorboard, Logger
from training_framework.dataloader import InfiniteSampler
from training_framework.training_session import TrainingSession, Step, step
from training_framework.training_engine import TrainingEngine

import torch
from torch.utils.data import Dataset, DataLoader


class DummyClassificationDataset(Dataset):
    def __init__(self, num_samples: int = 100, num_features: int = 5, num_classes: int = 2):
        """
        Generates random dummy data for framework testing.
        """
        self.num_samples = num_samples

        # Generate random continuous features: Shape (num_samples, num_features)
        self.features = torch.randn(num_samples, num_features)

        # Generate random integer labels for classification: Values between 0 and num_classes-1
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        # Tells the DataLoader how many samples are in this dataset
        return self.num_samples

    def __getitem__(self, idx):
        # Returns a single sample of (features, label)
        return self.features[idx], self.labels[idx]

    def collate_fn(self, batch):
        features_batch = []
        labels_batch = []
        for item in batch:
            x, y = item
            features_batch.append(torch.tensor(x))
            labels_batch.append(torch.tensor(y))

        return torch.stack(features_batch), torch.stack(labels_batch)

@step("sample_step")
class SampleStep(Step):
    def __init__(self):
        dataset = DummyClassificationDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            sampler=InfiniteSampler(len(dataset)),
            collate_fn=dataset.collate_fn,
        )
        self._dataloader_iter = iter(dataloader)
        self._model = nn.Sequential(nn.Linear(5, 2))

    @override
    def run(self, training_iterator: "TrainingSession") -> None:
        feature_batch, label_batch = next(self._dataloader_iter)
        feature_batch.to(device=training_iterator.device)
        label_batch.to(device=training_iterator.device)

        output = self._model.forward(feature_batch)
        loss = F.cross_entropy(output, label_batch)
        loss.backward()

        training_iterator.share_value("loss", loss.item())

    @override
    def get_state(self) -> Any:
        pass

    @override
    def set_state(self, state: Any) -> None:
        pass


@pytest.fixture
def sample_session_config(tmp_path):
    return {
        'max_iterations': 50,
        'batch_size': 4,
        'sessions_dir': str(tmp_path / 'sessions'),
        'device': 'cpu',
        'rng_seed': 0,
        'logger': {
            'log_every': 1,
            'log_file': str(tmp_path / 'log.txt')
        },
        'checkpointer': {
            'checkpoint_every': 10,
            'checkpoints_dir': str(tmp_path / "checkpoints")
        },
        'tensorboard': {
            'host': '0.0.0.0',
            'port': 16032,
        }
    }

@pytest.fixture
def sample_session_config2(tmp_path):
    return {
        'max_iterations': 50,
        'batch_size': 4,
        'sessions_dir': str(tmp_path / 'sessions2'),
        'device': 'cpu',
        'rng_seed': 0,
        'logger': {
            'log_every': 1,
            'log_file': str(tmp_path / 'log2.txt')
        },
        'checkpointer': {
            'checkpoint_every': 10,
            'checkpoints_dir': str(tmp_path / "checkpoints2")
        },
        'tensorboard': {
            'host': '0.0.0.0',
            'port': 16033,
        }
    }


@pytest.fixture
def training_engine():
    return TrainingEngine({})

def test_configurator(sample_session_config, tmp_path):
    # create a temp config yaml
    file_path = str(tmp_path / 'config.yaml')
    with open(file_path, 'w') as f:
        yaml.safe_dump(sample_session_config, f)

    sys.argv = ["", f"{file_path}"]
    configurator = Configurator()

    session_config = configurator.get_session_config()
    logger_config = configurator.get_resource_config("logger")

    assert session_config == sample_session_config
    assert logger_config == sample_session_config['logger']

def test_configurator_override(sample_session_config, tmp_path):
    # create a temp config yaml
    file_path = str(tmp_path / 'config.yaml')
    with open(file_path, 'w') as f:
        yaml.safe_dump(sample_session_config, f)

    sys.argv = ["", f"{file_path}", "--override", "checkpointer.checkpoint_every=5"]
    configurator = Configurator()

    session_config = configurator.get_session_config()
    checkpointer_config = configurator.get_resource_config("checkpointer")

    assert checkpointer_config != sample_session_config['checkpointer']
    assert checkpointer_config['checkpoint_every'] == 5



def test_logger(sample_session_config, training_engine):
    session = TrainingSession(sample_session_config)
    session.add_step(SampleStep())
    session.register_hook(Logger(sample_session_config['logger']))

    training_engine.register_session(session)

    with training_engine:
        training_engine._run_session(0)

    log_file_path = Path(sample_session_config['logger']['log_file'])
    with open(log_file_path, 'r') as f:
        assert log_file_path.stat().st_size > 0
        for i, line in enumerate(f.readlines()):
            assert line == f'Iteration {i+1}/{sample_session_config["max_iterations"]}\n'

def test_checkpointer(sample_session_config, training_engine):
    session = TrainingSession(sample_session_config)
    session.add_step(SampleStep())
    session.register_hook(Checkpointer(sample_session_config['checkpointer']))

    with training_engine:
        training_engine.register_session(session)
        training_engine._run_session(0)

    checkpoints_dir = Path(sample_session_config['checkpointer']['checkpoints_dir'])
    filepath_1 = os.path.join(str(checkpoints_dir), sorted(os.listdir(str(checkpoints_dir)))[0])
    filepath_2 = os.path.join(str(checkpoints_dir), sorted(os.listdir(str(checkpoints_dir)))[1])

    # load checkpoint
    reloaded_session_1 = Checkpointer.load_checkpoint(filepath_1)
    reloaded_session_2 = Checkpointer.load_checkpoint(filepath_2)

    assert reloaded_session_1.session_config == session.session_config
    assert reloaded_session_2.session_config == session.session_config

    assert reloaded_session_1.iteration == 1
    assert reloaded_session_2.iteration == 10

    assert len(reloaded_session_1._hooks) == 1
    assert len(reloaded_session_2._hooks) == 1

    assert reloaded_session_1._hooks[0].call_wrapper_every == session._hooks[0].call_wrapper_every
    assert reloaded_session_2._hooks[0].call_wrapper_every == session._hooks[0].call_wrapper_every


def test_tensorboard(sample_session_config, training_engine):
    session = TrainingSession(sample_session_config)
    session.register_resource(Tensorboard(sample_session_config['tensorboard']))

    with training_engine:
        training_engine.register_session(session)
        training_engine._run_session(0)


def test_thread_execution(sample_session_config, sample_session_config2, training_engine):
    session1 = TrainingSession(sample_session_config)
    session1.add_step(SampleStep())
    session1.register_hook(Logger(sample_session_config['logger']))
    session1.register_hook(Checkpointer(sample_session_config['checkpointer']))

    session2 = TrainingSession(sample_session_config2)
    session2.add_step(SampleStep())
    session2.register_hook(Logger(sample_session_config['logger']))
    session2.register_hook(Checkpointer(sample_session_config['checkpointer']))

    with training_engine:
        training_engine.register_session(session1)
        training_engine.register_session(session2)
        training_engine.run_all()