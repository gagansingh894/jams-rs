import requests

from src.jams.http import models
from src.jams.common import type


class HttpClient:
    def __init__(self, base_url: str):
        self.base_url = f"http://{base_url}"

    def health_check(self) -> None:
        url = f"{self.base_url}/healthcheck"
        try:
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"health check failed with {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError

    def predict(self, model_name: str, model_input: str) -> type.Prediction:
        url = f"{self.base_url}/api/predict"
        request = models.PredictRequest(
            model_name=model_name, input=model_input
        ).model_dump()

        try:
            response = requests.post(url, json=request)
            if response.status_code != 200:
                raise Exception(f"predict failed with {response.status_code}")
            try:
                resp_obj = models.PredictResponse.model_validate(response.json())
                return type.Prediction(resp_obj.output)
            except ValueError as e:
                raise Exception(f"fail to parse predict response: {e}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError

    def add_model(self, model_name: str) -> None:
        url = f"{self.base_url}/api/models"
        request = models.AddModelsRequest(model_name=model_name).model_dump()
        try:
            response = requests.post(url, json=request)
            if response.status_code != 200:
                raise Exception(f"add model failed with {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError

    def update_model(self, model_name: str) -> None:
        url = f"{self.base_url}/api/models"
        request = models.UpdateModelsRequest(model_name=model_name).model_dump()
        try:
            response = requests.put(url, json=request)
            if response.status_code != 200:
                raise Exception(f"update model failed with {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError

    def delete_model(self, model_name: str) -> None:
        url = f"{self.base_url}/api/models?model_name={model_name}"
        try:
            response = requests.delete(url)
            if response.status_code != 200:
                raise Exception(f"delete model failed with {response.status_code}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError

    def get_models(self) -> models.GetModelsResponse:
        url = f"{self.base_url}/api/models"
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError
        if response.status_code != 200:
            raise Exception(f"get model failed with {response.status_code}")
        try:
            return models.GetModelsResponse.model_validate(response.json())
        except ValueError as e:
            raise Exception(f"fail to parse get model response: {e}")
