from src.jams.http import client

from tests.helper import get_http_url


def test_successfully_makes_health_check_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = client.HttpClient(base_url)

    # Act
    http_client.health_check()

    # Assert
    # If the function errors out then we can assume the test has failed
