# JAMS-SERVE

This library crate is part of a wider project called **J.A.M.S - Just Another Model Server**. Please refer [here](https://github.com/gagansingh894/jams-rs).

`jams-serve` is a http and gRPC API library for jams-core.

Refer here for the [OpenAPI Spec](https://github.com/gagansingh894/jams-rs/blob/main/openapi.yml)

- `/healthcheck`: Endpoint for health checks

- `/api/predict`: Endpoint for making predictions

- `/api/models`: Endpoint for managing models

Refer here for the [proto definition](https://github.com/gagansingh894/jams-rs/blob/main/jams-serve/proto/api/v1/jams.proto)

- `HealthCheck`
- `Predict`
- `GetModels`
- `AddModel`
- `UpdateModel`
- `DeleteModel`


**Stable on Linux x86_64 architecture**
