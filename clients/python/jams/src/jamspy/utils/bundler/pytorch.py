from typing import Union

from src.jamspy.utils.bundler.common import ARTEFACTS_DIR, Bundle, create_tar_gz

import torch.nn


class PytorchBundler(Bundle):
    def __init__(self, model_obj: Union[torch.nn.Module, torch.nn.Sequential]):
        super().__init__()
        self.model = model_obj

    def bundle(self, model_name: str) -> None:
        framework = 'pytorch'
        # model_spec = _model_spec(framework)
        model_save_path = f'{ARTEFACTS_DIR}/{framework}-{model_name}.pt'
        tar_gz_path = f'{ARTEFACTS_DIR}/{framework}-{model_name}.tar.gz'

        script_module = torch.jit.script(self.model)
        script_module.save(model_save_path)
        create_tar_gz(model_save_path, tar_gz_path)
