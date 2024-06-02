use std::sync::Arc;
use crate::model::predictor::Predictor;

pub type ModelName = String;

/// Trait representing a storage system for machine learning models.
pub trait Storage {
    /// Fetches all available models from the storage.
    fn fetch_models(&self);

    /// Retrieves a specific machine learning/deep learning model by its name.
    fn get_model(&self, _: ModelName) -> Arc<dyn Predictor>;
}