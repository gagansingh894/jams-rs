import os


def get_http_url() -> str:
    hostname = os.environ.get("JAMS_HTTP_HOSTNAME")
    if hostname is None:
        hostname = "0.0.0.0"

    return f"{hostname}:3000"


def get_grpc_url() -> str:
    hostname = os.environ.get("JAMS_GRPC_HOSTNAME")
    if hostname is None:
        hostname = "0.0.0.0"

    return f"{hostname}:4000"
