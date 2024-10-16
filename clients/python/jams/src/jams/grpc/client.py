import grpc  # type: ignore

from google.protobuf import empty_pb2
from src.jams.proto import jams_pb2_grpc, jams_pb2
from src.jams.common import type


class GrpcClient:
    def __init__(self, base_url: str):
        self._channel = grpc.insecure_channel(base_url)
        self._stub = jams_pb2_grpc.ModelServerStub(self._channel)  # type: ignore

    def health_check(self) -> None:
        try:
            self._stub.HealthCheck(empty_pb2.Empty())
        except grpc.RpcError as e:
            raise e
        finally:
            self._channel.close()

    def predict(self, model_name: str, model_input: str) -> type.Prediction:
        try:
            resp: jams_pb2.PredictResponse = self._stub.Predict(
                jams_pb2.PredictRequest(model_name=model_name, input=model_input)
            )
            return type.Prediction(resp.output)
        except grpc.RpcError as e:
            raise e
        finally:
            self._channel.close()

    def add_model(self, model_name: str) -> None:
        try:
            self._stub.AddModel(jams_pb2.AddModelRequest(model_name=model_name))
        except grpc.RpcError as e:
            raise e
        finally:
            self._channel.close()

    def update_model(self, model_name: str) -> None:
        try:
            self._stub.UpdateModel(jams_pb2.UpdateModelRequest(model_name=model_name))
        except grpc.RpcError as e:
            raise e
        finally:
            self._channel.close()

    def delete_model(self, model_name: str) -> None:
        try:
            self._stub.DeleteModel(jams_pb2.DeleteModelRequest(model_name=model_name))
        except grpc.RpcError as e:
            raise e
        finally:
            self._channel.close()

    def get_models(self) -> jams_pb2.GetModelsResponse:
        try:
            resp: jams_pb2.GetModelsResponse = self._stub.GetModels(empty_pb2.Empty())
            return resp
        except grpc.RpcError as e:
            raise e
        finally:
            self._channel.close()
