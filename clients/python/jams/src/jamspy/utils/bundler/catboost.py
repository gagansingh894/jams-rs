from typing import Union

from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore

from src.jamspy.utils.bundler.common import ARTEFACTS_DIR, Bundle, create_tar_gz
from src.jamspy.utils.bundler.type import ModelSpec


class CatBoostBundler(Bundle):

    def __init__(self, model_obj: Union[CatBoostClassifier, CatBoostRegressor]):
        super().__init__()
        self.model = model_obj

    def bundle(self, model_name: str) -> None:
        framework = 'catboost'
        model_save_path = f'{ARTEFACTS_DIR}/{framework}-{model_name}'
        tar_gz_path = f'{ARTEFACTS_DIR}/{framework}.tar.gz'

        self.model.save_model(save_path, format='cbm')  # type: ignore
        create_tar_gz(model_save_path, tar_gz_path)

    def spec(self) -> ModelSpec:
        raise NotImplementedError
