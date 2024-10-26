import pytest

from src.jamspy.client import grpc
from src.jamspy.client.models.proto import jams_pb2

from tests.client.helper import get_grpc_url


@pytest.mark.asyncio
async def test_successfully_makes_get_models_request() -> None:
    # Arrange
    base_url = get_grpc_url()
    grpc_client = grpc.Client(base_url)

    # Act
    resp = await grpc_client.get_models()

    # Assert
    # If the function errors out then we can assume the test has failed
    assert isinstance(resp, jams_pb2.GetModelsResponse)


@pytest.mark.asyncio
async def test_successfully_makes_delete_model_request() -> None:
    # Arrange
    base_url = get_grpc_url()
    grpc_client = grpc.Client(base_url)

    # Act
    await grpc_client.add_model(model_name="pytorch-my_awesome_californiahousing_model")
    await grpc_client.delete_model(model_name="my_awesome_californiahousing_model")

    # Assert
    # If the function errors out then we can assume the test has failed


@pytest.mark.asyncio
async def test_successfully_makes_add_model_request() -> None:
    # Arrange
    base_url = get_grpc_url()
    grpc_client = grpc.Client(base_url)

    # Act
    await grpc_client.delete_model(model_name="my_awesome_penguin_model")
    await grpc_client.add_model(model_name="tensorflow-my_awesome_penguin_model")

    # Assert
    # If the function errors out then we can assume the test has failed


@pytest.mark.asyncio
async def test_successfully_makes_update_model_request() -> None:
    # Arrange
    base_url = get_grpc_url()
    grpc_client = grpc.Client(base_url)

    # Act
    await grpc_client.update_model(model_name="titanic_model")

    # Assert
    # If the function errors out then we can assume the test has failed
