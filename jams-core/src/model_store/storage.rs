use crate::model::predictor::Predictor;
use crate::model::frameworks::{ModelFramework, CATBOOST, LIGHTGBM, PYTORCH, TENSORFLOW, TORCH};
use dashmap::mapref::one::Ref;
use std::sync::Arc;
use serde::Serialize;
use crate::model;

pub type ModelName = String;

/// Trait representing a storage system for machine learning models.
pub trait Storage: Send + Sync + 'static {
    /// Fetches all available models from the storage.
    fn fetch_models(&self) -> anyhow::Result<()>;

    /// Adds a specific machine learning/deep learning model
    fn add_model(&self, model_name: ModelName, model_path: &str) -> anyhow::Result<()>;

    /// Updates a specific machine learning/deep learning model based on model name
    fn update_model(&self, model_name: ModelName) -> anyhow::Result<()>;

    /// Retrieves a specific machine learning/deep learning model by its name.
    fn get_model(&self, model_name: ModelName) -> Option<Ref<ModelName, Arc<Model>>>;

    /// Retrieves metadata for models which are currently loaded in memory
    fn get_models(&self) -> anyhow::Result<Vec<Metadata>>;

    /// Removes a specific machine learning/deep learning model by its name.
    fn delete_model(&self, model_name: ModelName) -> anyhow::Result<()>;
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
    pub name: String,
    pub framework: ModelFramework,
    pub path: String,
    pub last_updated: String,
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

/// Loads a machine learning model based on the specified framework and model path.
///
/// # Arguments
///
/// * `model_framework` - An enum representing the framework of the model to be loaded.
/// * `model_path` - A string slice that holds the path to the model file.
///
/// # Returns
///
/// This function returns a Result containing an `Arc<dyn Predictor>` on success, or an error of type `anyhow::Error` on failure.
///
/// # Errors
///
/// This function will return an error if:
/// * The model framework is unsupported.
/// * The model fails to load due to an internal error specific to the framework.
///
pub fn load_model(model_framework: ModelFramework, model_path: &str) -> anyhow::Result<Arc<dyn Predictor>> {
    if model_framework == TENSORFLOW {
        match model::tensorflow::Tensorflow::load(model_path) {
            Ok(predictor) => {
                Ok(Arc::new(predictor))
            }
            Err(e) => { anyhow::bail!("Failed to load Tensorflow model: {}", e) }
        }
    } else if (model_framework == TORCH) || (model_framework == PYTORCH) {
        match model::torch::Torch::load(model_path) {
            Ok(predictor) => { Ok(Arc::new(predictor)) }
            Err(e) => { anyhow::bail!("Failed to load Torch model: {}", e) }
        }
    } else if model_framework == CATBOOST {
        match model::catboost::Catboost::load(model_path) {
            Ok(predictor) => { Ok(Arc::new(predictor)) }
            Err(e) => { anyhow::bail!("Failed to load Catboost model: {}", e) }
        }
    } else if model_framework == LIGHTGBM {
        match model::lightgbm::LightGBM::load(model_path) {
            Ok(predictor) => { Ok(Arc::new(predictor)) }
            Err(e) => { anyhow::bail!("Failed to load LightGBM model: {}", e) }
        }
    } else {
        anyhow::bail!("unsupported model framework: {}", model_framework)
    }
}