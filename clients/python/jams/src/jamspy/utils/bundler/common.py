import os
import tarfile
from abc import ABC, abstractmethod

ARTEFACTS_DIR = 'jams_artefacts'


class Bundle(ABC):

    def __init__(self) -> None:
        # create a directory with the name jams_artefacts
        os.makedirs(ARTEFACTS_DIR, exist_ok=True)

    @abstractmethod
    def bundle(self, model_name: str) -> None:
        pass


def create_tar_gz(input_filename: str, output_filename: str) -> None:
    # Ensure the file exists
    if os.path.exists(input_filename):
        # Create a tar.gz archive and add the input file
        with tarfile.open(output_filename, 'w:gz') as tar:
            tar.add(input_filename, arcname=os.path.basename(input_filename))
    else:
        raise FileNotFoundError
