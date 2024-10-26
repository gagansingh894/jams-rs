import httpx

from src.jamspy.client.models import common, http


class Client:
    def __init__(self, base_url: str, timeout: float = 5):
        self.base_url = f'http://{base_url}'
        if base_url.startswith('http://') or base_url.startswith('https://'):
            self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=timeout)

    async def health_check(self) -> None:
        url = f'{self.base_url}/healthcheck'
        try:
            response = await self.client.get(url)
            if response.status_code != 200:
                raise Exception(f'health check failed with {response.status_code}')
        except httpx.ConnectError:
            raise ConnectionError('Could not connect to the server.')

    async def predict(self, model_name: str, model_input: str) -> common.Prediction:
        url = f'{self.base_url}/api/predict'
        request = http.PredictRequest(
            model_name=model_name, input=model_input
        ).model_dump()

        try:
            response = await self.client.post(url, json=request)
            if response.status_code != 200:
                raise Exception(f'predict failed with {response.status_code}')
            try:
                resp_obj = http.PredictResponse.model_validate(response.json())
                return common.Prediction(resp_obj.output)
            except ValueError as e:
                raise Exception(f'fail to parse predict response: {e}')
        except httpx.ConnectError:
            raise ConnectionError('Could not connect to the server.')

    async def add_model(self, model_name: str) -> None:
        url = f'{self.base_url}/api/models'
        request = http.AddModelsRequest(model_name=model_name).model_dump()
        try:
            response = await self.client.post(url, json=request)
            if response.status_code != 200:
                raise Exception(f'add model failed with {response.status_code}')
        except httpx.ConnectError:
            raise ConnectionError('Could not connect to the server.')

    async def update_model(self, model_name: str) -> None:
        url = f'{self.base_url}/api/models'
        request = http.UpdateModelsRequest(model_name=model_name).model_dump()
        try:
            response = await self.client.put(url, json=request)
            if response.status_code != 200:
                raise Exception(f'update model failed with {response.status_code}')
        except httpx.ConnectError:
            raise ConnectionError('Could not connect to the server.')

    async def delete_model(self, model_name: str) -> None:
        url = f'{self.base_url}/api/models?model_name={model_name}'
        try:
            response = await self.client.delete(url)
            if response.status_code != 200:
                raise Exception(f'delete model failed with {response.status_code}')
        except httpx.ConnectError:
            raise ConnectionError('Could not connect to the server.')

    async def get_models(self) -> http.GetModelsResponse:
        url = f'{self.base_url}/api/models'
        try:
            response = await self.client.get(url)
            if response.status_code != 200:
                raise Exception(f'get model failed with {response.status_code}')
            return http.GetModelsResponse.model_validate(response.json())
        except httpx.ConnectError:
            raise ConnectionError('Could not connect to the server.')