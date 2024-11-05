import pytest

from jamspy.client import grpc

from tests.client.helper import get_grpc_url


@pytest.mark.asyncio
async def test_successfully_makes_health_check_request() -> None:
    # Arrange
    base_url = get_grpc_url()
    grpc_client = grpc.Client(base_url)

    # Act
    await grpc_client.health_check()

    # Assert
    # If the function errors out then we can assume the test has failed
