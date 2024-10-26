import grpc  # type: ignore

from google.protobuf import empty_pb2
from src.jamspy.client.models.proto import jams_pb2
from src.jamspy.client.models.proto import jams_pb2_grpc
from src.jamspy.client.models import common

class Client:
    def __init__(self, base_url: str, timeout: float = 5):
        timeout=self._timeout = timeout
        self._channel = grpc.aio.insecure_channel(base_url)
        self._stub = jams_pb2_grpc.ModelServerStub(self._channel)  # type: ignore

    async def close(self) -> None:
        await self._channel.close()

    async def health_check(self) -> None:
        try:
            await self._stub.HealthCheck(empty_pb2.Empty(), timeout=self._timeout)
        except grpc.RpcError as e:
            raise e

    async def predict(self, model_name: str, model_input: str) -> common.Prediction:
        try:
            resp: jams_pb2.PredictResponse = await self._stub.Predict(
                jams_pb2.PredictRequest(model_name=model_name, input=model_input),
                timeout=self._timeout
            )
            return common.Prediction(resp.output)
        except grpc.RpcError as e:
            raise e

    async def add_model(self, model_name: str) -> None:
        try:
            await self._stub.AddModel(jams_pb2.AddModelRequest(model_name=model_name), timeout=self._timeout)
        except grpc.RpcError as e:
            raise e

    async def update_model(self, model_name: str) -> None:
        try:
            await self._stub.UpdateModel(jams_pb2.UpdateModelRequest(model_name=model_name), timeout=self._timeout)
        except grpc.RpcError as e:
            raise e

    async def delete_model(self, model_name: str) -> None:
        try:
            await self._stub.DeleteModel(jams_pb2.DeleteModelRequest(model_name=model_name), timeout=self._timeout)
        except grpc.RpcError as e:
            raise e

    async def get_models(self) -> jams_pb2.GetModelsResponse:
        try:
            resp: jams_pb2.GetModelsResponse = await self._stub.GetModels(empty_pb2.Empty(), timeout=self._timeout)
            return resp
        except grpc.RpcError as e:
            raise e

