from typing import Union

from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore

from src.jamspy.utils.bundler.common import ARTEFACTS_DIR, Bundle, create_tar_gz


class CatBoostBundler(Bundle):

    def __init__(self, model_obj: Union[CatBoostClassifier, CatBoostRegressor]):
        super().__init__()
        self.model = model_obj

    def bundle(self, model_name: str) -> None:
        framework = 'catboost'
        model_save_path = f'{ARTEFACTS_DIR}/{framework}-{model_name}'
        tar_gz_path = f'{ARTEFACTS_DIR}/{framework}-{model_name}.tar.gz'

        self.model.save_model(model_save_path, format='cbm')
        create_tar_gz(model_save_path, tar_gz_path)
