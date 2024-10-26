import pytest

from src.jamspy.client import http
from src.jamspy.client.models.http import GetModelsResponse

from tests.client.helper import get_http_url


@pytest.mark.asyncio
async def test_successfully_makes_get_models_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = http.Client(base_url)

    # Act
    resp = await http_client.get_models()

    # Assert
    # If the function errors out then we can assume the test has failed
    assert isinstance(resp, GetModelsResponse)


@pytest.mark.asyncio
async def test_successfully_makes_delete_model_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = http.Client(base_url)

    # Act
    await http_client.add_model(model_name="pytorch-my_awesome_californiahousing_model")
    await http_client.delete_model(model_name="my_awesome_californiahousing_model")

    # Assert
    # If the function errors out then we can assume the test has failed


@pytest.mark.asyncio
async def test_successfully_makes_add_model_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = http.Client(base_url)

    # Act
    await http_client.delete_model(model_name="my_awesome_penguin_model")
    await http_client.add_model(model_name="tensorflow-my_awesome_penguin_model")

    # Assert
    # If the function errors out then we can assume the test has failed


@pytest.mark.asyncio
async def test_successfully_makes_update_model_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = http.Client(base_url)

    # Act
    await http_client.update_model(model_name="titanic_model")

    # Assert
    # If the function errors out then we can assume the test has failed
