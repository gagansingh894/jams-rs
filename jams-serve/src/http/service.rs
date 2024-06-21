use crate::common::state::AppState;
use crate::common::worker;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::Json;
use jams_core::model_store::storage::Metadata;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::oneshot;

#[derive(Deserialize)]
pub struct AddModelRequest {
    model_name: String,
}

#[derive(Deserialize)]
pub struct UpdateModelRequest {
    model_name: String,
}

#[derive(Deserialize)]
pub struct DeleteModelRequest {
    model_name: String,
}

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
    output: String,
}

/// Health check endpoint handler.
///
/// This function handles the health check ("/health") endpoint and returns a simple status code indicating the server is healthy.
///
/// # Returns
/// - `StatusCode`: An HTTP status code indicating the health status. Always returns `StatusCode::OK`.
pub async fn healthcheck() -> StatusCode {
    StatusCode::OK
}

/// Adds a new model to the model store.
///
/// # Arguments
///
/// * `State(app_state)` - An `Arc` wrapped `AppState` instance representing the application state.
/// * `Json(payload)` - A `Json` wrapped `AddModelRequest` instance containing the model name and path.
///
/// # Returns
///
/// * `StatusCode::OK` if the model is successfully added.
/// * `StatusCode::INTERNAL_SERVER_ERROR` if there is an error during the addition process.
///
pub async fn add_model(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<AddModelRequest>,
) -> StatusCode {
    match app_state.manager.add_model(payload.model_name).await {
        Ok(_) => StatusCode::OK,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Updates an existing model in the model store.
///
/// # Arguments
///
/// * `State(app_state)` - An `Arc` wrapped `AppState` instance representing the application state.
/// * `Json(payload)` - A `Json` wrapped `UpdateModelRequest` instance containing the model name.
///
/// # Returns
///
/// * `StatusCode::OK` if the model is successfully updated.
/// * `StatusCode::INTERNAL_SERVER_ERROR` if there is an error during the update process or if the model does not exist.
pub async fn update_model(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<UpdateModelRequest>,
) -> StatusCode {
    match app_state.manager.update_model(payload.model_name).await {
        Ok(_) => StatusCode::OK,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Deletes an existing model from the model store.
///
/// # Arguments
///
/// * `State(app_state)` - An `Arc` wrapped `AppState` instance representing the application state.
/// * `Query(request)` - A `Query` wrapped `DeleteModelRequest` instance containing the model name.
///
/// # Returns
///
/// * `StatusCode::OK` if the model is successfully deleted.
/// * `StatusCode::INTERNAL_SERVER_ERROR` if there is an error during the deletion process or if the model does not exist.
pub async fn delete_model(
    State(app_state): State<Arc<AppState>>,
    request: Query<DeleteModelRequest>,
) -> StatusCode {
    match app_state.manager.delete_model(request.0.model_name) {
        Ok(_) => StatusCode::OK,
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Retrieves the list of models.
///
/// This endpoint fetches the list of models available in the server and returns their metadata.
///
/// # Arguments
///
/// * `State(app_state)` - The application state containing the manager that handles model operations.
///
/// # Returns
///
/// A tuple containing:
/// * `StatusCode::OK` and a JSON response with the list of models and their count if successful.
/// * `StatusCode::INTERNAL_SERVER_ERROR` and an empty list of models if there is an error.
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
/// // Body: {"output": ""}
pub async fn predict(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<PredictRequest>,
) -> (StatusCode, Json<PredictResponse>) {
    let (tx, rx) = oneshot::channel();

    let cpu_pool = &app_state.cpu_pool;
    let manager = Arc::clone(&app_state.manager);
    let model_name = payload.model_name;
    let model_input = payload.input;

    cpu_pool.spawn(move || worker::predict_and_send(manager, model_name, model_input, tx));

    match rx.await {
        Ok(predictions) => match predictions {
            Ok(output) => (StatusCode::OK, Json(PredictResponse { output })),
            Err(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(PredictResponse {
                    output: "".to_string(),
                }),
            ),
        },
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(PredictResponse {
                output: "".to_string(),
            }),
        ),
    }
}
