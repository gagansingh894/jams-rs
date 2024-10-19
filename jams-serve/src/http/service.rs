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
/// This function processes the addition of a new model by interacting with the shared `Manager` in the application state.
/// It asynchronously attempts to add the model based on the provided `model_name` in the request payload.
///
/// # Arguments
///
/// - `State(app_state)`: The shared application state (`Arc<AppState>`), which contains the `Manager` responsible for
///   managing models in the system.
/// - `Json(payload)`: The JSON payload containing the `AddModelRequest`, which includes the `model_name` and any additional
///   relevant information (such as the model's file path).
///
/// # Returns
///
/// - `Result<StatusCode, (StatusCode, String)>`:
///   - If the model is successfully added, it returns `StatusCode::OK`.
///   - If an error occurs during the addition process, it returns `StatusCode::INTERNAL_SERVER_ERROR` along with an error message.
/// 
/// # Error Handling
/// If there is an error while adding the model, such as a failure to interact with the file system or any underlying issues
/// with the model manager, the function returns an `INTERNAL_SERVER_ERROR` status code along with a generic error message. Detailed
/// error information is expected to be logged on the server side.
pub async fn add_model(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<AddModelRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    match app_state.manager.add_model(payload.model_name).await {
        Ok(_) => Ok(StatusCode::OK),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to add model ❌. Please check server logs".to_string(),
        )),
    }
}

/// Updates an existing model in the model store.
///
/// This function processes the update of a model by interacting with the shared `Manager` in the application state.
/// It asynchronously attempts to update the specified model based on the provided `model_name` in the request payload.
///
/// # Arguments
///
/// - `State(app_state)`: The shared application state (`Arc<AppState>`), which contains the `Manager` responsible for
///   managing models in the system.
/// - `Json(payload)`: The JSON payload containing the `UpdateModelRequest`, which includes the `model_name` 
///   of the model to be updated and any additional relevant information (such as updated model parameters).
///
/// # Returns
///
/// - `Result<StatusCode, (StatusCode, String)>`:
///   - If the model is successfully updated, it returns `StatusCode::OK`.
///   - If an error occurs during the update process, it returns `StatusCode::INTERNAL_SERVER_ERROR` along with an error message.
///
/// # Error Handling
/// If there is an error while updating the model, such as failure to find the model or issues with underlying data,
/// the function returns an `INTERNAL_SERVER_ERROR` status code along with a generic error message. Detailed
/// error information is expected to be logged on the server side.
pub async fn update_model(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<UpdateModelRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    match app_state.manager.update_model(payload.model_name).await {
        Ok(_) => Ok(StatusCode::OK),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to update model ❌. Please check server logs".to_string(),
        )),
    }
}

/// Deletes an existing model from the model store.
///
/// This function handles the deletion of a model based on the provided model name in the request query.
/// It interacts with the shared `Manager` from the application state (`AppState`) to attempt to delete
/// the specified model.
///
/// # Arguments
///
/// - `State(app_state)`: The shared application state (`Arc<AppState>`), which contains the `Manager`
///   responsible for managing the models.
/// - `Query(request)`: The query parameters containing the `DeleteModelRequest`, which includes the model name
///   to be deleted. This is extracted from the query string of the request URL.
///
/// # Returns
///
/// - `Result<StatusCode, (StatusCode, String)>`:
///   - If the model is successfully deleted, returns `StatusCode::OK`.
///   - If an error occurs during the deletion process or if the model does not exist, it returns
///     `StatusCode::INTERNAL_SERVER_ERROR` along with a detailed error message.
///
/// # Error Handling
/// If the model cannot be found or an error occurs during the deletion process (e.g., file system error or database error),
/// the function returns an `INTERNAL_SERVER_ERROR` status code along with a string that provides a detailed description of the error.
pub async fn delete_model(
    State(app_state): State<Arc<AppState>>,
    request: Query<DeleteModelRequest>,
) -> Result<StatusCode, (StatusCode, String)> {
    match app_state.manager.delete_model(request.0.model_name) {
        Ok(_) => Ok(StatusCode::OK),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to delete model ❌: {}", e),
        )),
    }
}

/// Retrieves the list of models available in the server.
///
/// This endpoint fetches the list of models and their metadata from the server. 
/// It interacts with the shared `Manager` in the application state to retrieve the data.
///
/// # Arguments
///
/// - `State(app_state)`: The application state (`Arc<AppState>`), which contains the `Manager` responsible for
///   handling model operations.
///
/// # Returns
///
/// A `Result<(StatusCode, Json<GetModelsResponse>), (StatusCode, String)>`:
/// - On success, it returns:
///   - `StatusCode::OK` with a JSON response containing the total count of models and their metadata.
/// - On failure, it returns:
///   - `StatusCode::INTERNAL_SERVER_ERROR` with an error message indicating the reason for the failure.
///
/// # Error Handling
/// If there is an error retrieving the models (e.g., failure to access the underlying storage or an unexpected exception),
/// the function returns an `INTERNAL_SERVER_ERROR` status code along with a descriptive error message.
pub async fn get_models(
    State(app_state): State<Arc<AppState>>,
) -> Result<(StatusCode, Json<GetModelsResponse>), (StatusCode, String)> {
    match app_state.manager.get_models() {
        Ok(models) => Ok((
            StatusCode::OK,
            Json(GetModelsResponse {
                total: models.len() as i32,
                models,
            }),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to get models ❌: {}", e),
        )),
    }
}

/// Prediction endpoint handler.
///
/// This function asynchronously processes prediction requests by utilizing a worker thread
/// to perform the prediction. It takes a `PredictRequest` payload and leverages the shared
/// `Manager` to make predictions on a separate thread pool (`cpu_pool`). The result is sent
/// back to the main async flow via a oneshot channel.
///
/// # Arguments
/// - `State(app_state)`: Shared state containing an `Arc<AppState>`, which holds the `Manager` responsible for
///   managing models and the `cpu_pool` for running blocking operations in a thread pool.
/// - `Json(payload)`: The JSON payload which contains the prediction request, including the `model_name` (the name of
///   the model to be used) and `input` (the input data for the model in serialized form).
///
/// # Returns
/// - `Result<(StatusCode, Json<PredictResponse>), (StatusCode, String)>`:
///   - On success, it returns `StatusCode::OK` with the prediction result wrapped in a `PredictResponse` struct.
///   - On failure, it returns `StatusCode::INTERNAL_SERVER_ERROR` with an error message in plain text.
///
/// # Example Request
// ```
// POST /predict
// {
//   "model_name": "example_model",
//   "input": "{\"key\": \"value\"}"
// }
// ```
///
/// # Example Success Response
// ```
// StatusCode: 200 OK
// {
//   "error": "",
//   "output": "{\"result_key\": \"[[result_value_1], [result_value_2], [result_value_3]]\"}"
// }
// ```

/// # Example Error Response
// ```
// StatusCode: 500 INTERNAL SERVER ERROR
// {
//   "output": "",
//   "error": "Failed to predict ❌: specific error message"
// }
// ```
///
/// # Error Handling
/// If the prediction fails either due to an error from the `Manager` or failure to receive a response from the oneshot
/// channel, an appropriate error message is returned along with the `INTERNAL_SERVER_ERROR` status code.
///
/// This handler ensures that any blocking operation (like model prediction) is offloaded to the `cpu_pool` to avoid
/// blocking the main async runtime.
pub async fn predict(
    State(app_state): State<Arc<AppState>>,
    Json(payload): Json<PredictRequest>,
) -> Result<(StatusCode, Json<PredictResponse>), (StatusCode, String)> {
    let (tx, rx) = oneshot::channel();

    let cpu_pool = &app_state.cpu_pool;
    let manager = Arc::clone(&app_state.manager);
    let model_name = payload.model_name;
    let model_input = payload.input;

    cpu_pool.spawn(move || worker::predict_and_send(manager, model_name, model_input, tx));

    match rx.await {
        Ok(predictions) => match predictions {
            Ok(output) => Ok((StatusCode::OK, Json(PredictResponse { output }))),
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to predict ❌: {}", e),
            )),
        },
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to predict ❌: {}", e),
        )),
    }
}
