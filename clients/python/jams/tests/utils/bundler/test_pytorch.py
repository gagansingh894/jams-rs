import os

import torch

from src.jamspy.utils.bundler.pytorch import PytorchBundler


def test_successfully_creates_bundle_for_pytorch_model() -> None:
    # Arrange - load existing model
    model = torch.jit.load('tests/utils/artefacts/pytorch-my_awesome_californiahousing_model.pt')  # type: ignore
    model.eval()

    # Act
    bundler = PytorchBundler(model)
    bundler.bundle('my_awesome_californiahousing_model')

    # Assert
    assert os.path.exists('jams_artefacts/pytorch-my_awesome_californiahousing_model.tar.gz')