import os
from typing import Any

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from training_framework import training_session
from training_framework.checkpointer import Checkpointer
from training_framework.dataloader import InfiniteSampler
from training_framework.training_session import TrainingSession, IterationComponent, Tensorboard

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

class SampleIterationComponent(IterationComponent):
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

    def run(self, training_iterator: "TrainingSession") -> None:
        print('iteration : ', training_iterator.iteration)
        feature_batch, label_batch = next(self._dataloader_iter)
        feature_batch.to(device=training_iterator.device)
        label_batch.to(device=training_iterator.device)

        output = self._model.forward(feature_batch)
        loss = F.cross_entropy(output, label_batch)
        loss.backward()

        training_iterator.share_value("loss", loss.item())

    def __getstate__(self) -> Any:
        pass

    def __setstate__(self, state: Any) -> None:
        pass


@pytest.fixture
def sample_session_config():
    return {
        'max_iterations': 50,
        'batch_size': 4,
        'sessions_dir': "tests/outputs/sessions",
        'device': 'cpu',
        'log_every': 1,
        'rng_seed': 0,
    }

@pytest.fixture
def sample_tensorboard_config():
    return {
        'host': '0.0.0.0',
        'port': 16032
    }
def test_1(sample_session_config):
    session = TrainingSession(sample_session_config)
    session.add_iteration_component(SampleIterationComponent())

    with session:
        session.start()

def test_checkpointer(sample_session_config, tmp_path):
    session = TrainingSession(sample_session_config)
    session.add_iteration_component(SampleIterationComponent())
    session.add_iteration_wrapper(Checkpointer({
        'checkpoint_every': 10,
        'checkpoints_dir': str(tmp_path)
    }))

    with session:
        session.start()

    filepath_1 = os.path.join(str(tmp_path), sorted(os.listdir(str(tmp_path)))[0])
    filepath_2 = os.path.join(str(tmp_path), sorted(os.listdir(str(tmp_path)))[1])

    # load checkpoint
    reloaded_session_1 = Checkpointer.load_checkpoint(filepath_1)
    reloaded_session_2 = Checkpointer.load_checkpoint(filepath_2)

    assert reloaded_session_1.session_config == session.session_config
    assert reloaded_session_2.session_config == session.session_config

    assert reloaded_session_1.iteration == 1
    assert reloaded_session_2.iteration == 10

    assert len(reloaded_session_1._iteration_wrappers) == 1
    assert len(reloaded_session_2._iteration_wrappers) == 1

    assert reloaded_session_1._iteration_wrappers[0].call_wrapper_every == session._iteration_wrappers[0].call_wrapper_every
    assert reloaded_session_2._iteration_wrappers[0].call_wrapper_every == session._iteration_wrappers[0].call_wrapper_every


def test_tensorboard(sample_session_config, sample_tensorboard_config):
    session = TrainingSession(sample_session_config)
    session.register_resource(Tensorboard(sample_tensorboard_config))

    with session:
        session.start()