from src.jams.client import grpc

from tests.helper import get_grpc_url


def test_successfully_makes_health_check_request() -> None:
    # Arrange
    base_url = get_grpc_url()
    grpc_client = grpc.Client(base_url)

    # Act
    grpc_client.health_check()

    # Assert
    # If the function errors out then we can assume the test has failed
