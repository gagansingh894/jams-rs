use crate::model;
use crate::model::frameworks::{CATBOOST, LIGHTGBM, PYTORCH, TENSORFLOW, TORCH};
use crate::model::predictor::Predictor;
use crate::model_store::storage::{ModelName, Storage};
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use std::fs;
use std::sync::Arc;

/// A local model store that manages models stored in a specified directory.
///
/// The `LocalModelStore` struct is responsible for loading models from the local filesystem
/// and providing access to them. It supports models in various frameworks like TensorFlow,
/// Torch, CatBoost, and LightGBM.
///
/// # Fields
/// - `models` (DashMap<ModelName, Arc&ltdyn Predictor&gt>): A thread-safe map of model names to their respective predictor instances.
/// - `model_dir` (String): The directory where models are stored.
pub struct LocalModelStore {
    pub models: DashMap<ModelName, Arc<dyn Predictor>>,
    pub model_dir: String,
}

impl LocalModelStore {
    /// Creates a new `LocalModelStore` instance.
    ///
    /// This method initializes the local model store with the specified directory.
    ///
    /// # Arguments
    /// - `model_dir` (String): The directory where models are stored.
    ///
    /// # Returns
    /// - `Ok(LocalModelStore)`: If the instance was successfully created.
    /// - `Err(anyhow::Error)`: If there was an error creating the instance.
    ///
    pub fn new(model_dir: String) -> anyhow::Result<Self> {
        let models: DashMap<ModelName, Arc<dyn Predictor>> = DashMap::new();
        Ok(LocalModelStore { models, model_dir })
    }
}

impl Storage for LocalModelStore {
    /// Fetches and loads all models from the specified directory.
    ///
    /// The models have a specific name format which allows us to identify the model framework
    /// `<model_framework>-<model_name>`
    /// The `model_framework` and `model_name` are seperated by `-`
    ///
    /// This method iterates through the models in the directory, identifies the model framework based on the file name,
    /// and loads each model into the `models` map.
    ///
    /// # Returns
    /// - `Ok(())`: If all models were successfully fetched and loaded.
    /// - `Err(anyhow::Error)`: If there was an error fetching or loading the models.
    ///
    fn fetch_models(&self) -> anyhow::Result<()> {
        let dir = match fs::read_dir(&self.model_dir) {
            Ok(dir) => dir,
            Err(e) => {
                anyhow::bail!("Failed to read dir: {}", e)
            }
        };

        for file in dir {
            let file = match file {
                Ok(file) => file,
                Err(e) => {
                    anyhow::bail!("Failed to read file: {}", e)
                }
            };
            let (file_path, file_name) = (file.path(), file.file_name());
            let full_path = match file_path.into_os_string().into_string() {
                Ok(full_path) => full_path,
                Err(_) => {
                    anyhow::bail!("Failed to convert OsString to String")
                }
            };
            let file_name = match file_name.to_str() {
                None => {
                    anyhow::bail!("Failed to convert OsString to str")
                }
                Some(file_name) => file_name,
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
                        let model =
                            model::tensorflow::Tensorflow::load(full_path.as_str()).unwrap();
                        self.models.insert(model_name.to_string(), Arc::new(model));
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
                                let model = model::torch::Torch::load(full_path.as_str()).unwrap();
                                self.models.insert(model_name.to_string(), Arc::new(model));
                            }
                        }
                    }
                    Some(model_name) => {
                        let model = model::torch::Torch::load(full_path.as_str()).unwrap();
                        self.models.insert(model_name.to_string(), Arc::new(model));
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
                        let model = model::catboost::Catboost::load(full_path.as_str()).unwrap();
                        self.models.insert(model_name.to_string(), Arc::new(model));
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
                        let model = model::lightgbm::LightGBM::load(full_path.as_str()).unwrap();
                        self.models.insert(model_name.to_string(), Arc::new(model));
                    }
                }
            } else {
                log::warn!("Unexpected model framework encountered in file ⚠️. \n File:{} \n", file_name);
            }
        }
        Ok(())
    }

    fn get_model(&self, model_name: ModelName) -> Option<Ref<ModelName, Arc<dyn Predictor>>> {
        self.models.get(model_name.as_str())
    }

    fn get_model_names(&self) -> anyhow::Result<Vec<String>> {
        let model_names: Vec<String> = self.models.iter().map(|f| f.key().to_string()).collect();
        Ok(model_names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_load_models_from_different_frameworks_into_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();

        // load models
        let result = local_model_store.fetch_models();

        // assert
        assert!(result.is_ok());
        assert_ne!(local_model_store.models.len(), 0);
    }

    #[test]
    fn successfully_get_model_from_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();

        // load models
        let result = local_model_store.fetch_models();
        let model = local_model_store.get_model("my_awesome_autompg_model".to_string());

        // assert
        assert!(result.is_ok());
        assert!(model.is_some());
    }

    #[test]
    fn successfully_get_model_names_from_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();

        // load models
        let result = local_model_store.fetch_models();
        let model_names = local_model_store.get_model_names();

        // assert
        assert!(result.is_ok());
        assert!(model_names.is_ok());
        assert_ne!(model_names.unwrap().len(), 0);
    }
}
