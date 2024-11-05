import pytest

from jamspy.client import http

from tests.client.helper import get_http_url


@pytest.mark.asyncio
async def test_successfully_makes_health_check_request() -> None:
    # Arrange
    base_url = get_http_url()
    http_client = http.Client(base_url)

    # Act
    await http_client.health_check()

    # Assert
    # If the function errors out then we can assume the test has failed
