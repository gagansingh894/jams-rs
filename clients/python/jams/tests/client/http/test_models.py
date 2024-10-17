from src.jamspy.client import http
from src.jamspy.client.models.http import GetModelsResponse

from tests.helper import get_http_url


def test_successfully_makes_get_models_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = http.Client(base_url)

    # Act
    resp = http_client.get_models()

    # Assert
    # If the function errors out then we can assume the test has failed
    assert isinstance(resp, GetModelsResponse)


def test_successfully_makes_delete_model_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = http.Client(base_url)

    # Act
    http_client.delete_model(model_name="my_awesome_californiahousing_model")

    # Assert
    # If the function errors out then we can assume the test has failed


def test_successfully_makes_add_model_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = http.Client(base_url)

    # Act
    http_client.delete_model(model_name="my_awesome_penguin_model")
    http_client.add_model(model_name="tensorflow-my_awesome_penguin_model")

    # Assert
    # If the function errors out then we can assume the test has failed


def test_successfully_makes_update_model_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = http.Client(base_url)

    # Act
    http_client.update_model(model_name="titanic_model")

    # Assert
    # If the function errors out then we can assume the test has failed
