use serde::Serialize;
use std::collections::HashMap;

pub const DEFAULT_OUTPUT_KEY: &str = "predictions";

/// Struct representing the output of a prediction.
#[derive(Debug, Serialize)]
pub struct ModelOutput {
    /// The predictions made by the model.
    /// We use a hashmap because we can have models with multiple outputs
    /// The client is responsible for selecting the correct field for their respective purpose
    /// For the models which do not support multiple outputs, the default key will be 'predictions'
    pub predictions: HashMap<String, Vec<Vec<f64>>>,
}
