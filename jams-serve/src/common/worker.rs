use jams_core::manager::Manager;
use std::sync::Arc;
use tokio::sync::oneshot::Sender;

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
pub fn predict_and_send(
    manager: Arc<Manager>,
    model_name: String,
    input: String,
    tx: Sender<anyhow::Result<String>>,
) {
    // we do not handle the result here
    let predictions = manager.predict(model_name, input.as_str());
    let _ = tx.send(predictions);
}
