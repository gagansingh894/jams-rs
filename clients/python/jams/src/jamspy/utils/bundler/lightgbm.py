from lightgbm import Booster

from src.jamspy.utils.bundler.common import ARTEFACTS_DIR, Bundle, create_tar_gz
from src.jamspy.utils.bundler.type import ModelSpec


class LightBoostBundler(Bundle):

    def __init__(self, model_obj: Booster):
        super().__init__()
        self.model = model_obj

    def bundle(self, model_name: str) -> None:
        framework = 'lightgbm'
        # model_spec = _model_spec(framework)
        model_save_path = f'{ARTEFACTS_DIR}/{framework}-{model_name}.txt'
        tar_gz_path = f'{ARTEFACTS_DIR}/{framework}.tar.gz'

        self.model.save_model(model_save_path)
        create_tar_gz(model_save_path, tar_gz_path)

    def spec(self) -> ModelSpec:
        raise NotImplementedError
