import os

from catboost import CatBoostClassifier, CatBoostRegressor

from src.jamspy.utils.bundler.catboost import CatBoostBundler


def test_successfully_creates_bundle_for_catboost_classifier_model() -> None:
    # Arrange - load existing model
    model = CatBoostClassifier()
    model.load_model('tests/utils/artefacts/catboost-titanic_model')

    # Act
    bundler = CatBoostBundler(model)
    bundler.bundle('titanic_model')

    # Assert
    assert os.path.exists('jams_artefacts/catboost-titanic_model.tar.gz')


def test_successfully_creates_bundle_for_catboost_regressor_model() -> None:
    # Arrange - load existing model
    model = CatBoostClassifier()
    model.load_model('tests/utils/artefacts/catboost-my_awesome_regressor_model')

    # Act
    bundler = CatBoostBundler(model)
    bundler.bundle('my_awesome_regressor_model')

    # Assert
    assert os.path.exists('jams_artefacts/catboost-my_awesome_regressor_model.tar.gz')

