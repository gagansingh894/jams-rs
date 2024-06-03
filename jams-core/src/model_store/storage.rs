use crate::model::predictor::Predictor;
use dashmap::mapref::one::Ref;
use std::sync::Arc;

pub type ModelName = String;

/// Trait representing a storage system for machine learning models.
pub trait Storage {
    /// Fetches all available models from the storage.
    fn fetch_models(&self) -> anyhow::Result<()>;

    /// Retrieves a specific machine learning/deep learning model by its name.
    fn get_model(&self, _: ModelName) -> Option<Ref<ModelName, Arc<dyn Predictor>>>;
}
