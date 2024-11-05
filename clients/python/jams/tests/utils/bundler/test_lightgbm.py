import os

import lightgbm as lgb

from jamspy.utils.bundler.lightgbm_model import LGBMBundler


def test_successfully_creates_bundle_for_lightgbm_model() -> None:
    # Arrange - load existing model
    model = lgb.Booster(model_file='tests/utils/artefacts/lightgbm-my_awesome_reg_model.txt')

    # Act
    bundler = LGBMBundler(model)
    bundler.bundle('my_awesome_reg_model')

    # Assert
    assert os.path.exists('jams_artefacts/lightgbm-my_awesome_reg_model.tar.gz')