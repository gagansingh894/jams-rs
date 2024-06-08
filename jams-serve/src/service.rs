use crate::router::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use jams_core::manager::Manager;
use jams_core::model_store::storage::Metadata;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::sync::oneshot::Sender;

/// Response structure for retrieving the list of models.
///
/// Represents the JSON response structure returned by the API when
/// retrieving the list of models.
#[derive(Serialize)]
pub struct GetModelsResponse {
    /// Total number of models.
    total: i32,
    /// List of model names.
    models: Vec<Metadata>,
}

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

pub async fn get_models(
    State(app_state): State<Arc<AppState>>,
) -> (StatusCode, Json<GetModelsResponse>) {
    match app_state.manager.get_models() {
        Ok(models) => (
            StatusCode::OK,
            Json(GetModelsResponse {
                total: models.len() as i32,
                models,
            }),
        ),
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(GetModelsResponse {
                total: 0,
                models: vec![],
            }),
        ),
    }
}

/// Prediction endpoint handler.
///
/// This function asynchronously processes prediction requests. It takes a `PredictRequest` payload,
/// uses the shared `Manager` to make predictions on a separate thread pool (`cpu_pool`), and returns
/// a `PredictResponse` containing the prediction result or error message.
///
/// # Arguments
/// - `State(app_state)`: Shared state containing an `Arc<AppState>` with a `Manager` and a `ThreadPool`.
/// - `Json(payload)`: JSON payload containing the prediction request (`model_name` and `input`).
///
/// # Returns
/// - A tuple `(StatusCode, Json<PredictResponse>)`: HTTP status code and JSON response. If the prediction
///   is successful, returns `StatusCode::OK` and the prediction result. Otherwise, returns
///   `StatusCode::INTERNAL_SERVER_ERROR` and the error message.
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
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<PredictRequest>,
) -> (StatusCode, Json<PredictResponse>) {
    let (tx, rx) = oneshot::channel();

    let cpu_pool = &app_state.cpu_pool;
    let manager = Arc::clone(&app_state.manager);

    cpu_pool.spawn(move || predict_and_send(manager, payload, tx));

    match rx.await {
        Ok(predictions) => match predictions {
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
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(PredictResponse {
                    error: format!("Failed to make predictions: {}", e),
                    output: "".to_string(),
                }),
            ),
        },
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(PredictResponse {
                error: format!("Failed to make predictions: {}", e),
                output: "".to_string(),
            }),
        ),
    }
}

/// Asynchronously predicts an outcome using a shared manager and sends the result or error
/// message through a channel.
///
/// This function takes an Arc-wrapped `Manager`, a `PredictRequest` containing model name
/// and input data, and a `Sender<anyhow::Result<String>>` channel for sending the prediction result.
///
/// # Arguments
///
/// * `manager` - An `Arc` reference to the shared `Manager` instance used for predictions.
/// * `payload` - A `PredictRequest` containing the model name and input data for prediction.
/// * `tx` - A `Sender<anyhow::Result<String>>` channel endpoint for sending the prediction result.
///
/// The function asynchronously sends the prediction result through the provided
/// channel (`tx`) based on the result of the prediction operation using the shared `Manager`.
fn predict_and_send(
    manager: Arc<Manager>,
    payload: PredictRequest,
    tx: Sender<anyhow::Result<String>>,
) {
    // we do not handle the result here
    let predictions = manager.predict(payload.model_name, payload.input.as_str());
    let _ = tx.send(predictions);
}
