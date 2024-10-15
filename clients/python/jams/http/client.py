import requests

from models import PredictRequest, PredictResponse, AddModelsRequest, UpdateModelsRequest, GetModelsResponse


class HttpClient:

    def __init__(self, base_url: str):
        self.base_url = f'http://{base_url}'

    def health_check(self):
        url = f'{self.base_url}/healthcheck'
        response = requests.get(url)

    def predict(self, model_name: str, model_input: str):
        url = f'{self.base_url}/api/predict'
        request = PredictRequest(model_name=model_name, input=model_input).dict()
        response = requests.post(url, json=request)

        if response.status_code != 200:
            pass

        PredictResponse.parse_obj(response.json())

    def add_model(self, model_name: str):
        url = f'{self.base_url}/api/models'
        request = AddModelsRequest(model_name=model_name).dict()
        response = requests.post(url, json=request)


    def update_model(self, model_name: str):
        url = f'{self.base_url}/api/models'
        request = UpdateModelsRequest(model_name=model_name).dict()
        response = requests.put(url, json=request)


    def delete_model(self, model_name: str):
        url = f'{self.base_url}/api/models?model_name={model_name}'
        response = requests.delete(url)

    def get_models(self):
        url = f'{self.base_url}/api/models'
        response = requests.get(url)

        if response.status_code != 200:
            pass


        GetModelsResponse.parse_obj(response.json())
