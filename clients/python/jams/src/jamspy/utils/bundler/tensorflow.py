from typing import Union

import tensorflow as tf  # type: ignore

from src.jamspy.utils.bundler.common import ARTEFACTS_DIR, Bundle, create_tar_gz


class TensorflowBundler(Bundle):
    def __init__(self, model_obj: Union[tf.keras.Model, tf.keras.Sequential]):
        super().__init__()
        self.model = model_obj

    def bundle(self, model_name: str) -> None:
        framework = 'tensorflow'
        model_save_path = f'{ARTEFACTS_DIR}/{framework}-{model_name}'
        tar_gz_path = f'{ARTEFACTS_DIR}/{framework}-{model_name}.tar.gz'

        self.model.save(model_save_path)
        create_tar_gz(model_save_path, tar_gz_path)
