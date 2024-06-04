use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use jams_core::manager::Manager;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A request for making a prediction.
///
/// This struct represents the data required to make a prediction using a specified model.
///
/// # Fields
/// - `model_name` (String): The name of the model to use for the prediction.
/// - `input` (String): The input data for the prediction, formatted as a JSON-like string.
///
/// # Example
/// ```json
/// {
///     "model_name": "example_model",
///     "input": "{\"key1\": \["value1]\", \"key2\": \["value2]\"}"
/// }
/// ```
#[derive(Deserialize, Serialize)]
pub struct PredictRequest {
    model_name: String,
    input: String,
}

/// The response from a prediction request.
///
/// This struct represents the output data from a prediction.
///
/// # Fields
/// - `output` (String): The output data from the prediction, formatted as a JSON-like string.
///
/// # Example 1 - Single Output
/// ```json
/// {
///     "output": "{\"result_key\": \"[[result_value]]\"}"
/// }
/// ```
/// # Example 2 - MultiClass Output
/// ```json
/// {
///     "output": "{\"result_key\": \"[[result_value_1, result_value_3, result_value_2]]\"}"
/// }
/// ```
#[derive(Deserialize, Serialize)]
pub struct PredictResponse {
    error: String,
    output: String,
}

/// Root endpoint handler.
///
/// This function handles the root ("/") endpoint and returns a status message indicating that the server is running.
///
/// # Returns
/// - `(StatusCode, &'static str)`: A tuple containing the HTTP status code and a static string message.
///
/// # Example
/// ```
/// // This endpoint returns the following response:
/// // StatusCode: 200 OK
/// // Body: "J.A.M.S - Just Another Model Server is running 🚀\n"
/// ```
pub async fn root() -> (StatusCode, &'static str) {
    (
        StatusCode::OK,
        "J.A.M.S - Just Another Model Server is running 🚀\n",
    )
}

/// Health check endpoint handler.
///
/// This function handles the health check ("/health") endpoint and returns a simple status code indicating the server is healthy.
///
/// # Returns
/// - `StatusCode`: An HTTP status code indicating the health status. Always returns `StatusCode::OK`.
///
/// # Example
/// ```
/// // This endpoint returns the following response:
/// // StatusCode: 200 OK
/// ```
pub async fn healthcheck() -> StatusCode {
    StatusCode::OK
}

/// Prediction endpoint handler.
///
/// This function handles the prediction ("/predict") endpoint. It takes a `PredictRequest` payload, uses the `Manager` to make a prediction, and returns a `PredictResponse`.
///
/// # Arguments
/// - `State(manager)`: The shared state containing the `Manager`.
/// - `Json(payload)`: The JSON payload containing the prediction request.
///
/// # Returns
/// - `(StatusCode, Json<PredictResponse>)`: A tuple containing the HTTP status code and the JSON response. If the prediction is successful, returns `StatusCode::OK` and the prediction result. Otherwise, returns `StatusCode::INTERNAL_SERVER_ERROR` and the error message.
///
/// # Example
/// ```
/// // Example request:
/// // POST /predict
/// // Body: {"model_name": "example_model", "input": "{\"key\": \"value\"}"}
///
/// // Example response for success:
/// // StatusCode: 200 OK
/// // Body: {"error": "", "output": "{\"result_key\": \"[[result_value_1], [result_value_2], [result_value_3]]\"}"}
///
/// // Example response for error:
/// // StatusCode: 500 INTERNAL SERVER ERROR
/// // Body: {"error": "error_message", "output": ""}
/// ```
pub async fn predict(
    State(manager): State<Arc<Manager>>,
    Json(payload): Json<PredictRequest>,
) -> (StatusCode, Json<PredictResponse>) {
    match manager.predict(payload.model_name, payload.input.as_str()) {
        Ok(output) => {
            let error_msg = "".to_string();
            (
                StatusCode::OK,
                Json(PredictResponse {
                    error: error_msg,
                    output,
                }),
            )
        }
        Err(e) => {
            let output = "".to_string();
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(PredictResponse {
                    error: e.to_string(),
                    output,
                }),
            )
        }
    }
}
