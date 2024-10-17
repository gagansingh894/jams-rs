import grpc  # type: ignore

from google.protobuf import empty_pb2
from src.jamspy.client.models.proto import jams_pb2
from src.jamspy.client.models.proto import jams_pb2_grpc
from src.jamspy.client.models import common


class Client:
    def __init__(self, base_url: str):
        self._channel = grpc.insecure_channel(base_url)
        self._stub = jams_pb2_grpc.ModelServerStub(self._channel)  # type: ignore

    def __del__(self) -> None:
        self._channel.close()

    def health_check(self) -> None:
        try:
            self._stub.HealthCheck(empty_pb2.Empty())
        except grpc.RpcError as e:
            raise e

    def predict(self, model_name: str, model_input: str) -> common.Prediction:
        try:
            resp: jams_pb2.PredictResponse = self._stub.Predict(
                jams_pb2.PredictRequest(model_name=model_name, input=model_input)
            )
            return common.Prediction(resp.output)
        except grpc.RpcError as e:
            raise e

    def add_model(self, model_name: str) -> None:
        try:
            self._stub.AddModel(jams_pb2.AddModelRequest(model_name=model_name))
        except grpc.RpcError as e:
            raise e

    def update_model(self, model_name: str) -> None:
        try:
            self._stub.UpdateModel(jams_pb2.UpdateModelRequest(model_name=model_name))
        except grpc.RpcError as e:
            raise e

    def delete_model(self, model_name: str) -> None:
        try:
            self._stub.DeleteModel(jams_pb2.DeleteModelRequest(model_name=model_name))
        except grpc.RpcError as e:
            raise e

    def get_models(self) -> jams_pb2.GetModelsResponse:
        try:
            resp: jams_pb2.GetModelsResponse = self._stub.GetModels(empty_pb2.Empty())
            return resp
        except grpc.RpcError as e:
            raise e
