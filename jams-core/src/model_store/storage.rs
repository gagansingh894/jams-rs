use crate::model;
use crate::model::frameworks::{ModelFramework, CATBOOST, LIGHTGBM, PYTORCH, TENSORFLOW, TORCH};
use crate::model::predictor::Predictor;
use async_trait::async_trait;
use chrono::Utc;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use serde::Serialize;
use std::sync::Arc;
use tokio::fs;

pub type ModelName = String;

/// Trait representing a storage system for machine learning models.
#[async_trait]
pub trait Storage: Send + Sync + 'static {
    /// Adds a specific machine learning/deep learning model
    async fn add_model(&self, model_name: ModelName, model_path: &str) -> anyhow::Result<()>;

    /// Updates a specific machine learning/deep learning model based on model name
    async fn update_model(&self, model_name: ModelName) -> anyhow::Result<()>;

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

/// Loads models from the specified directory and returns a DashMap containing the models.
///
/// The models have a specific name format which allows us to identify the model framework
/// `<model_framework>-<model_name>`
/// The `model_framework` and `model_name` are separated by `-`
///
/// This function reads the contents of the provided directory, identifies model files based on
/// their filenames, and loads them into a `DashMap` where the keys are model names and the values
/// are `Arc`-wrapped `Model` instances.
///
/// # Arguments
///
/// * `model_dir` - A `String` specifying the path to the directory containing model files.
///
/// # Returns
///
/// A `Result` containing a `DashMap` with model names as keys and `Arc<Model>` as values.
/// On failure, returns an `anyhow::Error` with details about the error.
///
/// # Errors
///
/// This function will return an error if it fails to read the directory, convert file paths, or
/// load any of the models.
pub async fn load_models(model_dir: String) -> anyhow::Result<DashMap<ModelName, Arc<Model>>> {
    let models: DashMap<ModelName, Arc<Model>> = DashMap::new();

    match fs::read_dir(model_dir.clone()).await {
        Ok(mut dir) => {
            while let Ok(Some(entry)) = dir.next_entry().await {
                let file_path = match entry.path().to_str() {
                    None => {
                        anyhow::bail!("Failed to convert PathBuf to str ❌")
                    }
                    Some(file_path) => file_path.to_string(),
                };
                let file_name = match entry.file_name().into_string() {
                    Ok(file_name) => file_name,
                    Err(_) => {
                        anyhow::bail!("Failed to convert OsString to String ❌")
                    }
                };

                if file_name.contains(TENSORFLOW) {
                    let prefix = format!("{}-", TENSORFLOW);
                    match file_name.to_string().strip_prefix(&prefix) {
                        None => {
                            anyhow::bail!(
                                "Failed to strip prefix {} from file name {}",
                                prefix,
                                file_name
                            )
                        }
                        Some(model_name) => {
                            let predictor =
                                model::tensorflow::Tensorflow::load(file_path.as_str())?;
                            let now = Utc::now();
                            let sanitised_model_name = sanitize_model_name(model_name);
                            let model = Model::new(
                                Arc::new(predictor),
                                sanitised_model_name.clone(),
                                TENSORFLOW,
                                file_path.clone(),
                                now.to_rfc3339(),
                            );
                            models.insert(sanitised_model_name, Arc::new(model));
                            log::info!("Successfully loaded model from path: {} ✅", file_path);
                        }
                    }
                } else if file_name.contains(TORCH) {
                    // Torch and PyTorch are same models. PyTorch is a python wrapper around Torch
                    let prefix = format!("{}-", TORCH);
                    match file_name.to_string().strip_prefix(&prefix) {
                        None => {
                            let prefix = format!("{}-", PYTORCH);
                            match file_name.to_string().strip_prefix(&prefix) {
                                None => {
                                    anyhow::bail!(
                                        "Failed to strip prefix {} from file name {}",
                                        prefix,
                                        file_name
                                    )
                                }
                                Some(model_name) => {
                                    let predictor = model::torch::Torch::load(file_path.as_str())?;
                                    let now = Utc::now();
                                    let sanitised_model_name = sanitize_model_name(model_name);
                                    let model = Model::new(
                                        Arc::new(predictor),
                                        sanitised_model_name.clone(),
                                        PYTORCH, // TORCH can also be used, but they are aliases
                                        file_path.clone(),
                                        now.to_rfc3339(),
                                    );
                                    models.insert(sanitised_model_name, Arc::new(model));
                                    log::info!(
                                        "Successfully loaded model from path: {} ✅",
                                        file_path
                                    );
                                }
                            }
                        }
                        Some(model_name) => {
                            let predictor = model::torch::Torch::load(file_path.as_str())?;
                            let now = Utc::now();
                            let sanitised_model_name = sanitize_model_name(model_name);
                            let model = Model::new(
                                Arc::new(predictor),
                                sanitised_model_name.clone(),
                                PYTORCH, // TORCH can also be used, but they are aliases
                                file_path.clone(),
                                now.to_rfc2822(),
                            );
                            models.insert(sanitised_model_name, Arc::new(model));
                            log::info!("Successfully loaded model from path: {} ✅", file_path);
                        }
                    }
                } else if file_name.contains(CATBOOST) {
                    let prefix = format!("{}-", CATBOOST);
                    match file_name.to_string().strip_prefix(&prefix) {
                        None => {
                            anyhow::bail!(
                                "Failed to strip prefix {} from file name {}",
                                prefix,
                                file_name
                            )
                        }
                        Some(model_name) => {
                            let predictor = model::catboost::Catboost::load(file_path.as_str())?;
                            let now = Utc::now();
                            let sanitised_model_name = sanitize_model_name(model_name);
                            let model = Model::new(
                                Arc::new(predictor),
                                sanitised_model_name.clone(),
                                CATBOOST,
                                file_path.clone(),
                                now.to_rfc2822(),
                            );
                            models.insert(sanitised_model_name, Arc::new(model));
                            log::info!("Successfully loaded model from path: {} ✅", file_path);
                        }
                    }
                } else if file_name.contains(LIGHTGBM) {
                    let prefix = format!("{}-", LIGHTGBM);
                    match file_name.to_string().strip_prefix(&prefix) {
                        None => {
                            anyhow::bail!(
                                "Failed to strip prefix {} from file name {}",
                                prefix,
                                file_name
                            )
                        }
                        Some(model_name) => {
                            let predictor = model::lightgbm::LightGBM::load(file_path.as_str())?;
                            let now = Utc::now();
                            let sanitised_model_name = sanitize_model_name(model_name);
                            let model = Model::new(
                                Arc::new(predictor),
                                sanitised_model_name.clone(),
                                LIGHTGBM,
                                file_path.clone(),
                                now.to_rfc2822(),
                            );
                            models.insert(sanitised_model_name, Arc::new(model));
                            log::info!("Successfully loaded model from path: {} ✅", file_path);
                        }
                    }
                } else {
                    log::warn!(
                        "Unexpected model framework encountered in file ⚠️. \n File: {} \n",
                        file_name
                    );
                }
            }
        }
        Err(e) => {
            anyhow::bail!(
                "Failed to read directory {} ❌: {}",
                model_dir,
                e.to_string()
            )
        }
    }

    Ok(models)
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
pub async fn load_predictor(
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

/// Removes everything after the first dot ('.') from the input string.
///
/// # Arguments
///
/// * `input` - A reference to a string slice (`&str`) that contains the input string.
///
/// # Returns
///
/// A new `String` that contains the substring of `input` up to (but not including) the first dot ('.').
/// If no dot is found in `input`, the function returns a new `String` containing the entire `input`.
///
fn sanitize_model_name(input: &str) -> String {
    if let Some(index) = input.find('.') {
        let result = &input[..index];
        result.to_string()
    } else {
        input.to_string()
    }
}

/// Appends a file extension to the `model_path` based on the specified `model_framework`.
///
/// # Arguments
///
/// * `model_framework` - The framework type of the model.
/// * `model_path` - The base path of the model file.
///
/// # Returns
///
/// A `String` representing the `model_path` appended with the appropriate file extension
/// based on the `model_framework`.
///
pub fn append_model_format(model_framework: ModelFramework, model_path: String) -> String {
    if (model_framework == TORCH) || (model_framework == PYTORCH) {
        return format!("{}.pt", model_path);
    }

    if model_framework == LIGHTGBM {
        return format!("{}.txt", model_path);
    }

    model_path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_create_model_artefact() {
        // Setup
        let model_name = "pytorch-my_awesome_californiahousing_model".to_string(); // notice that model name is exactly the same in path
        let path =
            "tests/model_storage/models/pytorch-my_awesome_californiahousing_model.pt";
        let framework = PYTORCH;
        let timestamp = Utc::now();
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

    #[tokio::test]
    async fn successfully_load_tensorflow_framework_model() {
        let path = "tests/model_storage/models/tensorflow-my_awesome_penguin_model";
        let framework = TENSORFLOW;

        // Load Model
        let model = load_predictor(framework, path).await;

        // Assert
        assert!(model.is_ok());
    }

    #[tokio::test]
    async fn successfully_load_torch_framework_model() {
        let path = "tests/model_storage/models/torch-my_awesome_penguin_model.pt";
        let framework = TORCH;

        // Load Model
        let model = load_predictor(framework, path).await;

        // Assert
        assert!(model.is_ok());
    }

    #[tokio::test]
    async fn successfully_load_pytorch_framework_model() {
        let path =
            "tests/model_storage/models/pytorch-my_awesome_californiahousing_model.pt";
        let framework = PYTORCH;

        // Load Model
        let model = load_predictor(framework, path).await;

        // Assert
        assert!(model.is_ok());
    }

    #[tokio::test]
    async fn successfully_load_lightgbm_model() {
        let path = "tests/model_storage/models/lightgbm-my_awesome_binary_model_2.txt";
        let framework = LIGHTGBM;

        // Load Model
        let model = load_predictor(framework, path).await;

        // Assert
        assert!(model.is_ok());
    }

    #[tokio::test]
    async fn successfully_load_catboost_model() {
        let path = "tests/model_storage/models/catboost-titanic_model";
        let framework = CATBOOST;

        // Load Model
        let model = load_predictor(framework, path).await;

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

    #[test]
    fn sanitize_model_name_when_the_name_has_period() {
        let model_name = "my_torch_model.pt";

        let result = sanitize_model_name(model_name);

        // assert
        assert_eq!(result, "my_torch_model");
    }

    #[test]
    fn do_not_sanitize_model_name_when_the_name_is_already_in_correct_form() {
        let model_name = "my_torch_model";

        let result = sanitize_model_name(model_name);

        // assert
        assert_eq!(result, "my_torch_model");
    }

    #[test]
    fn append_model_format_when_model_framework_is_torch() {
        let path = "model/directory/my_torch_model";

        let result = append_model_format(TORCH, path.to_string());

        // assert
        assert_eq!(result, "model/directory/my_torch_model.pt")
    }

    #[test]
    fn append_model_format_when_model_framework_is_lightgbm() {
        let path = "model/directory/my_lightgbm_model";

        let result = append_model_format(LIGHTGBM, path.to_string());

        // assert
        assert_eq!(result, "model/directory/my_lightgbm_model.txt")
    }

    #[test]
    fn do_not_append_model_format_when_model_framework_not_torch_or_lightgbm() {
        let path = "model/directory/catboost-my_model";

        let result = append_model_format(CATBOOST, path.to_string());

        // assert
        assert_eq!(result, "model/directory/catboost-my_model")
    }
}
