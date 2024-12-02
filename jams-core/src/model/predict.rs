use crate::model::input::ModelInput;
use crate::model::output::ModelOutput;

/// Trait for making predictions using a model.
pub trait Predict: Send + Sync + 'static {
    /// Predicts the output for the given model input.
    ///
    /// # Arguments
    /// * `input` - The input data for the model.
    ///
    /// # Returns
    /// * `Ok(Output)` - The prediction output.
    /// * `Err(anyhow::Error)` - If there was an error during prediction.
    fn predict(&self, input: ModelInput) -> anyhow::Result<ModelOutput>;
}
