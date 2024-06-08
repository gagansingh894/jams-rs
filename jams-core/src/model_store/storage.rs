use crate::model;
use crate::model::frameworks::{ModelFramework, CATBOOST, LIGHTGBM, PYTORCH, TENSORFLOW, TORCH};
use crate::model::predictor::Predictor;
use dashmap::mapref::one::Ref;
use serde::Serialize;
use std::sync::Arc;

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

        Model { predictor, info }
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
pub fn load_predictor(
    model_framework: ModelFramework,
    model_path: &str,
) -> anyhow::Result<Arc<dyn Predictor>> {
    if model_framework == TENSORFLOW {
        match model::tensorflow::Tensorflow::load(model_path) {
            Ok(predictor) => Ok(Arc::new(predictor)),
            Err(e) => {
                anyhow::bail!("Failed to load Tensorflow model: {}", e)
            }
        }
    } else if (model_framework == TORCH) || (model_framework == PYTORCH) {
        match model::torch::Torch::load(model_path) {
            Ok(predictor) => Ok(Arc::new(predictor)),
            Err(e) => {
                anyhow::bail!("Failed to load Torch model: {}", e)
            }
        }
    } else if model_framework == CATBOOST {
        match model::catboost::Catboost::load(model_path) {
            Ok(predictor) => Ok(Arc::new(predictor)),
            Err(e) => {
                anyhow::bail!("Failed to load Catboost model: {}", e)
            }
        }
    } else if model_framework == LIGHTGBM {
        match model::lightgbm::LightGBM::load(model_path) {
            Ok(predictor) => Ok(Arc::new(predictor)),
            Err(e) => {
                anyhow::bail!("Failed to load LightGBM model: {}", e)
            }
        }
    } else {
        anyhow::bail!("unsupported model framework: {}", model_framework)
    }
}

/// Extracts the model framework from the given model path.
///
/// This function checks the provided model path for the presence of specific framework identifiers and returns the corresponding `ModelFramework` enum if a match is found.
///
/// # Arguments
///
/// * `model_path` - A string containing the path to the model file.
///
/// # Returns
///
/// This function returns an `Option<ModelFramework>`:
/// * `Some(ModelFramework)` if a matching framework identifier is found in the model path.
/// * `None` if no matching framework identifier is found.
///
pub fn extract_framework_from_path(model_path: String) -> Option<ModelFramework> {
    if model_path.contains(TENSORFLOW) {
        Some(TENSORFLOW)
    } else if model_path.contains(PYTORCH) {
        Some(PYTORCH)
    } else if model_path.contains(TORCH) {
        Some(TORCH)
    } else if model_path.contains(CATBOOST) {
        Some(CATBOOST)
    } else if model_path.contains(LIGHTGBM) {
        Some(LIGHTGBM)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_create_model_artefact() {
        // Setup
        let model_name = "pytorch-my_awesome_californiahousing_model".to_string(); // notice that model name is exactly the same in path
        let path =
            "tests/model_storage/local_model_store/pytorch-my_awesome_californiahousing_model.pt";
        let framework = PYTORCH;
        let timestamp = chrono::Utc::now();
        let predictor = model::torch::Torch::load(path).unwrap();

        // Create
        let model = Model::new(
            Arc::new(predictor),
            model_name.clone(),
            framework,
            path.to_string(),
            timestamp.to_rfc3339(),
        );

        // Assert
        assert_eq!(model_name, model.info.name);
        assert_eq!(path, model.info.path.as_str());
        assert_eq!(framework, model.info.framework);
        assert_eq!(timestamp.to_rfc3339(), model.info.last_updated);
    }

    #[test]
    fn successfully_load_tensorflow_framework_model() {
        let path = "tests/model_storage/local_model_store/tensorflow-my_awesome_penguin_model";
        let framework = TENSORFLOW;

        // Load Model
        let model = load_predictor(framework, path);

        // Assert
        assert!(model.is_ok());
    }

    #[test]
    fn successfully_load_torch_framework_model() {
        let path = "tests/model_storage/local_model_store/torch-my_awesome_penguin_model.pt";
        let framework = TORCH;

        // Load Model
        let model = load_predictor(framework, path);

        // Assert
        assert!(model.is_ok());
    }

    #[test]
    fn successfully_load_pytorch_framework_model() {
        let path =
            "tests/model_storage/local_model_store/pytorch-my_awesome_californiahousing_model.pt";
        let framework = PYTORCH;

        // Load Model
        let model = load_predictor(framework, path);

        // Assert
        assert!(model.is_ok());
    }

    #[test]
    fn successfully_load_lightgbm_model() {
        let path = "tests/model_storage/local_model_store/lightgbm-my_awesome_binary_model_2.txt";
        let framework = LIGHTGBM;

        // Load Model
        let model = load_predictor(framework, path);

        // Assert
        assert!(model.is_ok());
    }

    #[test]
    fn successfully_load_catboost_model() {
        let path = "tests/model_storage/local_model_store/catboost-titanic_model";
        let framework = CATBOOST;

        // Load Model
        let model = load_predictor(framework, path);

        // Assert
        assert!(model.is_ok());
    }

    #[test]
    fn successfully_extract_framework_from_path_when_tensorflow_framework() {
        let path = "model/directory/tensorflow-my_model";

        let result = extract_framework_from_path(path.to_string());

        // assert
        assert!(result.is_some());
        assert_eq!(result.unwrap(), TENSORFLOW)
    }

    #[test]
    fn successfully_extract_framework_from_path_when_torch_framework() {
        let path = "model/directory/torch-my_model";

        let result = extract_framework_from_path(path.to_string());

        // assert
        assert!(result.is_some());
        assert_eq!(result.unwrap(), TORCH)
    }

    #[test]
    fn successfully_extract_framework_from_path_when_pytorch_framework() {
        let path = "model/directory/pytorch-my_model";

        let result = extract_framework_from_path(path.to_string());

        // assert
        assert!(result.is_some());
        assert_eq!(result.unwrap(), PYTORCH)
    }

    #[test]
    fn successfully_extract_framework_from_path_when_lightgbm_framework() {
        let path = "model/directory/lightgbm-my_model";

        let result = extract_framework_from_path(path.to_string());

        // assert
        assert!(result.is_some());
        assert_eq!(result.unwrap(), LIGHTGBM)
    }

    #[test]
    fn successfully_extract_framework_from_path_when_catboost_framework() {
        let path = "model/directory/catboost-my_model";

        let result = extract_framework_from_path(path.to_string());

        // assert
        assert!(result.is_some());
        assert_eq!(result.unwrap(), CATBOOST)
    }

    #[test]
    fn fails_to_extract_framework_from_path_when_unknown_framework() {
        let path = "model/directory/fbprophet-my_model";

        let result = extract_framework_from_path(path.to_string());

        // assert
        assert!(result.is_none());
    }
}
