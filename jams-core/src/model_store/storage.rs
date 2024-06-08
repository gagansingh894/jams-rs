use crate::model::predictor::Predictor;
use crate::model::frameworks::ModelFramework;
use dashmap::mapref::one::Ref;
use std::sync::Arc;
use serde::Serialize;

pub type ModelName = String;

/// Trait representing a storage system for machine learning models.
pub trait Storage: Send + Sync + 'static {
    /// Fetches all available models from the storage.
    fn fetch_models(&self) -> anyhow::Result<()>;

    /// Retrieves a specific machine learning/deep learning model by its name.
    fn get_model(&self, model_name: ModelName) -> Option<Ref<ModelName, Arc<Model>>>;

    /// Retrieves metadata for models which are currently loaded in memory
    fn get_models(&self) -> anyhow::Result<Vec<Metadata>>;
}


/// Represents a machine learning model.
///
/// This struct holds the essential information and the actual predictive model
/// used in the application.
/// It includes metadata such as the model's name, path,
/// framework, and last updated timestamp.
///
/// # Fields
///
/// * `predictor` - An instance of a type that implements the `Predictor` trait.
/// This represents the predictive model.
/// * `info` - Metadata about the model.
///
pub struct Model {
    pub predictor: Arc<dyn Predictor>,
    pub info: Metadata,
}

/// Metadata for a machine learning model.
///
/// This struct holds metadata information about a model, including its name,
/// framework, filesystem path, and the last updated timestamp.
///
/// # Fields
///
/// * `name` - The name of the model.
/// * `framework` - The machine learning framework used to build the model (e.g., TensorFlow, PyTorch).
/// * `path` - The filesystem path to the model's file or directory.
/// * `last_updated` - The timestamp of when the model was last updated.
///
#[derive(Clone, Serialize)]
pub struct Metadata {
    name: String,
    framework: ModelFramework,
    path: String,
    last_updated: String,
}

impl Model {
    /// Creates a new `Model` instance.
    ///
    /// This function initializes a new `Model` with the given predictor and metadata.
    ///
    /// # Parameters
    ///
    /// * `predictor` - An instance of a type that implements the `Predictor` trait, wrapped in an `Arc`.
    /// * `model_name` - The name of the model.
    /// * `framework` - The machine learning framework used to build the model.
    /// * `path` - The filesystem path to the model's file or directory.
    /// * `last_updated` - The timestamp of when the model was last updated.
    ///
    /// # Returns
    ///
    /// A `Model` instance containing the predictor and metadata.
    ///
    pub fn new(
        predictor: Arc<dyn Predictor>,
        model_name: String,
        framework: ModelFramework,
        path: String,
        last_updated: String,
    ) -> Model {
        let info = Metadata {
            name: model_name,
            framework,
            path,
            last_updated,
        };

        Model {
            predictor,
            info,
        }
    }
}