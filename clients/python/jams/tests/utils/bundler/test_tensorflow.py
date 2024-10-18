import os

import tensorflow as tf

from src.jamspy.utils.bundler.tensorflow import TensorflowBundler


def test_successfully_creates_bundle_for_tensorflow_model() -> None:
    # Arrange - load existing model
    model = tf.keras.models.load_model('tests/utils/artefacts/tensorflow-my_awesome_penguin_model')

    # Act
    bundler = TensorflowBundler(model)
    bundler.bundle('my_awesome_penguin_model')

    # Assert
    assert os.path.exists('jams_artefacts/tensorflow-my_awesome_penguin_model.tar.gz')