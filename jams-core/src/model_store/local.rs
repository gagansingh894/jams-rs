use crate::model;
use crate::model::frameworks::{CATBOOST, LIGHTGBM, PYTORCH, TENSORFLOW, TORCH};
use crate::model_store::storage::{
    extract_framework_from_path, load_predictor, Metadata, Model, ModelName, Storage,
};
use chrono::Utc;
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
/// - `models` (DashMap<ModelName, Arc&ltModel&gt>): A thread-safe map of model names to their respective Model struct instances.
/// - `model_dir` (String): The directory where models are stored.
pub struct LocalModelStore {
    pub models: DashMap<ModelName, Arc<Model>>,
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
        let models: DashMap<ModelName, Arc<Model>> = DashMap::new();
        Ok(LocalModelStore { models, model_dir })
    }

    /// Loads all models from the specified directory.
    ///
    /// The models have a specific name format which allows us to identify the model framework
    /// `<model_framework>-<model_name>`
    /// The `model_framework` and `model_name` are separated by `-`
    ///
    /// This method iterates through the models in the directory, identifies the model framework based on the file name,
    /// and loads each model into the `models` map.
    ///
    /// # Returns
    /// - `Ok(())`: If all models were successfully fetched and loaded.
    /// - `Err(anyhow::Error)`: If there was an error fetching or loading the models.
    ///
    fn load_models(&self) -> anyhow::Result<()> {
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
                        let predictor =
                            model::tensorflow::Tensorflow::load(full_path.as_str()).unwrap();
                        let now = Utc::now();
                        let model = Model::new(
                            Arc::new(predictor),
                            model_name.to_string(),
                            TENSORFLOW,
                            full_path.clone(),
                            now.to_rfc3339(),
                        );
                        self.models.insert(model_name.to_string(), Arc::new(model));
                        log::info!("Successfully loaded model from path: {} ✅", full_path);
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
                                let predictor =
                                    model::torch::Torch::load(full_path.as_str()).unwrap();
                                let now = Utc::now();
                                let model = Model::new(
                                    Arc::new(predictor),
                                    model_name.to_string(),
                                    PYTORCH, // TORCH can also be used, but they are aliases
                                    full_path.clone(),
                                    now.to_rfc3339(),
                                );
                                self.models.insert(model_name.to_string(), Arc::new(model));
                                log::info!("Successfully loaded model from path: {} ✅", full_path);
                            }
                        }
                    }
                    Some(model_name) => {
                        let predictor = model::torch::Torch::load(full_path.as_str()).unwrap();
                        let now = Utc::now();
                        let model = Model::new(
                            Arc::new(predictor),
                            model_name.to_string(),
                            PYTORCH, // TORCH can also be used, but they are aliases
                            full_path.clone(),
                            now.to_rfc2822(),
                        );
                        self.models.insert(model_name.to_string(), Arc::new(model));
                        log::info!("Successfully loaded model from path: {} ✅", full_path);
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
                        let predictor =
                            model::catboost::Catboost::load(full_path.as_str()).unwrap();
                        let now = Utc::now();
                        let model = Model::new(
                            Arc::new(predictor),
                            model_name.to_string(),
                            CATBOOST,
                            full_path.clone(),
                            now.to_rfc2822(),
                        );
                        self.models.insert(model_name.to_string(), Arc::new(model));
                        log::info!("Successfully loaded model from path: {} ✅", full_path);
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
                        let predictor =
                            model::lightgbm::LightGBM::load(full_path.as_str()).unwrap();
                        let now = Utc::now();
                        let model = Model::new(
                            Arc::new(predictor),
                            model_name.to_string(),
                            LIGHTGBM,
                            full_path.clone(),
                            now.to_rfc2822(),
                        );
                        self.models.insert(model_name.to_string(), Arc::new(model));
                        log::info!("Successfully loaded model from path: {} ✅", full_path);
                    }
                }
            } else {
                log::warn!(
                    "Unexpected model framework encountered in file ⚠️. \n File: {} \n",
                    file_name
                );
            }
        }
        Ok(())
    }
}

impl Storage for LocalModelStore {
    /// Fetches and loads all models from the specified directory.
    ///
    /// The models have a specific name format which allows us to identify the model framework
    /// `<model_framework>-<model_name>`
    /// The `model_framework` and `model_name` are separated by `-`
    ///
    /// This method iterates through the models in the directory, identifies the model framework based on the file name,
    /// and loads each model into the `models` map.
    ///
    /// # Returns
    /// - `Ok(())`: If all models were successfully fetched and loaded.
    /// - `Err(anyhow::Error)`: If there was an error fetching or loading the models.
    ///
    fn fetch_models(&self) -> anyhow::Result<()> {
        match self.model_dir.is_empty() {
            true => {
                log::warn!("No model directory specified, hence no models will be loaded ⚠️");
                Ok(())
            }
            false => match self.load_models() {
                Ok(_) => {
                    log::info!("Successfully fetched valid models from directory ✅");
                    Ok(())
                }
                Err(e) => {
                    anyhow::bail!("Failed to fetch models - {}", e.to_string());
                }
            },
        }
    }

    ///
    /// This function attempts to extract the framework from the given model path,
    /// load the model using the identified framework, and then store the model
    /// in the model store along with its metadata.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be added.
    /// * `model_path` - The path to the model file.
    ///
    /// # Errors
    ///
    /// This function returns an error if:
    /// * The framework cannot be extracted from the model path.
    /// * The model fails to load.
    ///
    fn add_model(&self, model_name: ModelName, model_path: &str) -> anyhow::Result<()> {
        match extract_framework_from_path(model_path.to_string()) {
            None => {
                anyhow::bail!("Failed to extract framework from path")
            }
            Some(framework) => match load_predictor(framework, model_path) {
                Ok(predictor) => {
                    let now = Utc::now();
                    let model = Model::new(
                        predictor,
                        model_name.clone(),
                        framework,
                        model_path.to_string(),
                        now.to_rfc2822(),
                    );
                    self.models.insert(model_name, Arc::from(model));
                }
                Err(e) => {
                    anyhow::bail!("Failed to add new model: {e}")
                }
            },
        };
        Ok(())
    }

    /// Updates an existing model in the model store.
    ///
    /// This function retrieves the framework and location of an already loaded model,
    /// reloads the model, and updates its metadata in the model store.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be updated.
    ///
    /// # Errors
    ///
    /// This function returns an error if:
    /// * The specified model does not exist in the model store.
    /// * The model fails to load.
    ///
    fn update_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        // By calling remove on the hashmap, the object is returned on success/
        // We use the returned object, in this case the model to extract the framework and model path
        match self.models.remove(model_name.as_str()) {
            None => {
                anyhow::bail!(
                    "Failed to update as the specified model {} does not exist",
                    model_name
                )
            }
            Some(model) => {
                let (model_framework, model_path) =
                    (model.1.info.framework, model.1.info.path.as_str());
                match load_predictor(model_framework, model_path) {
                    Ok(predictor) => {
                        let now = Utc::now();
                        let model = Model::new(
                            predictor,
                            model_name.clone(),
                            model_framework,
                            model_path.to_string(),
                            now.to_rfc2822(),
                        );
                        self.models.insert(model_name.clone(), Arc::new(model));
                        Ok(())
                    }
                    Err(e) => {
                        anyhow::bail!("Failed to update the specified model {}: {}", model_name, e)
                    }
                }
            }
        }
    }

    /// Retrieves a model from the model store.
    ///
    /// This function returns a reference to the model with the specified name if it exists.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be retrieved.
    ///
    /// # Returns
    ///
    /// This function returns an `Option`:
    /// * `Some(Ref<ModelName, Arc<Model>>)` if the model exists.
    /// * `None` if the model does not exist.
    ///
    fn get_model(&self, model_name: ModelName) -> Option<Ref<ModelName, Arc<Model>>> {
        self.models.get(model_name.as_str())
    }

    /// Retrieves metadata for all models in the model store.
    ///
    /// This function returns a vector of `Metadata` containing information about all the models in the store.
    ///
    /// # Returns
    ///
    /// This function returns an `anyhow::Result` containing a vector of `Metadata`.
    ///
    fn get_models(&self) -> anyhow::Result<Vec<Metadata>> {
        let model_names: Vec<Metadata> = self
            .models
            .iter()
            .map(|f| f.value().info.to_owned())
            .collect();
        Ok(model_names)
    }

    /// Deletes a model from the model store.
    ///
    /// This function removes the model with the specified name from the store.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be deleted.
    ///
    /// # Errors
    ///
    /// This function returns an error if the specified model does not exist in the store.
    fn delete_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        match self.models.remove(&model_name) {
            None => {
                anyhow::bail!(
                    "Failed to delete model as the specified model {} does not exist",
                    model_name
                )
            }
            Some(_) => Ok(()),
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn successfully_load_models_from_different_frameworks_into_local_model_store() {
//         let model_dir = "tests/model_storage/local_model_store";
//         let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
//
//         // load models
//         let result = local_model_store.fetch_models();
//
//         // assert
//         assert!(result.is_ok());
//         assert_ne!(local_model_store.models.len(), 0);
//     }
//
//     #[test]
//     fn successfully_get_model_from_local_model_store() {
//         let model_dir = "tests/model_storage/local_model_store";
//         let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
//
//         // load models
//         let result = local_model_store.fetch_models();
//         let model = local_model_store.get_model("my_awesome_autompg_model".to_string());
//
//         // assert
//         assert!(result.is_ok());
//         assert!(model.is_some());
//     }
//
//     #[test]
//     fn fails_to_get_model_from_local_model_store_when_model_name_is_wrong() {
//         let model_dir = "tests/model_storage/local_model_store";
//         let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
//
//         // load models
//         let result = local_model_store.fetch_models();
//         let model = local_model_store.get_model("model_which_does_not_exist".to_string());
//
//         // assert
//         assert!(result.is_ok());
//         assert!(model.is_none());
//     }
//
//     #[test]
//     fn successfully_get_models_from_local_model_store() {
//         let model_dir = "tests/model_storage/local_model_store";
//         let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
//
//         // load models
//         let result = local_model_store.fetch_models();
//         let models = local_model_store.get_models();
//
//         // assert
//         assert!(result.is_ok());
//         assert!(models.is_ok());
//         assert_ne!(models.unwrap().len(), 0);
//     }
//
//     #[test]
//     fn successfully_deletes_model_in_the_local_model_store() {
//         let model_dir = "tests/model_storage/local_model_store";
//         let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
//
//         // load models
//         let result = local_model_store.fetch_models();
//         let deletion = local_model_store.delete_model("my_awesome_autompg_model".to_string());
//
//         // assert
//         assert!(result.is_ok());
//         assert!(deletion.is_ok());
//     }
//
//     #[test]
//     fn fails_to_deletes_model_in_the_local_model_store() {
//         let model_dir = "tests/model_storage/local_model_store";
//         let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
//
//         // load models
//         let result = local_model_store.fetch_models();
//         let deletion = local_model_store.delete_model("model_which_does_not_exist".to_string());
//
//         // assert
//         assert!(result.is_ok());
//         assert!(deletion.is_err());
//     }
//
//     #[test]
//     fn successfully_update_model_in_the_local_model_store() {
//         let model_dir = "tests/model_storage/local_model_store";
//         let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
//         let model_name = "my_awesome_autompg_model".to_string();
//
//         // load models
//         let result = local_model_store.fetch_models();
//         assert!(result.is_ok());
//
//         // retrieve timestamp from existing to model for assertion
//         let model = local_model_store
//             .get_model(model_name.clone())
//             .unwrap()
//             .to_owned();
//
//         // update model
//         let update = local_model_store.update_model(model_name.clone());
//         assert!(update.is_ok());
//         let updated_model = local_model_store
//             .get_model(model_name.clone())
//             .unwrap()
//             .to_owned();
//
//         // assert
//         assert_eq!(model.info.name, updated_model.info.name);
//         assert_eq!(model.info.path, updated_model.info.path);
//         assert_ne!(model.info.last_updated, updated_model.info.last_updated); // as model will be updated
//     }
//
//     #[test]
//     fn fails_to_update_model_in_the_local_model_store_when_model_name_is_incorrect() {
//         let model_dir = "tests/model_storage/local_model_store";
//         let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
//         let incorrect_model_name = "my_awesome_autompg_model_incorrect".to_string();
//
//         // load models
//         let result = local_model_store.fetch_models();
//         assert!(result.is_ok());
//
//         // update model with incorrect model name
//         let update = local_model_store.update_model(incorrect_model_name);
//
//         // assert
//         assert!(update.is_err());
//     }
//
//     #[test]
//     fn successfully_add_model_in_the_local_model_store() {
//         let model_dir = "tests/model_storage/local_model_store";
//         let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
//
//         // load models
//         let result = local_model_store.fetch_models();
//         assert!(result.is_ok());
//         // delete model to set up test
//         local_model_store
//             .delete_model("my_awesome_penguin_model".to_string())
//             .unwrap();
//         // assert that model is not present
//         let model = local_model_store.get_model("my_awesome_penguin_model".to_string());
//         assert!(model.is_none());
//         let num_models = local_model_store.get_models().unwrap().len();
//
//         // add model
//         let add = local_model_store.add_model(
//             "my_awesome_penguin_model".to_string(),
//             "tests/model_storage/local_model_store/tensorflow-my_awesome_penguin_model",
//         );
//         let num_models_after_add = local_model_store.get_models().unwrap().len();
//
//         // assert
//         assert!(add.is_ok());
//         let model = local_model_store.get_model("my_awesome_penguin_model".to_string());
//         assert!(model.is_some());
//         assert_eq!(num_models_after_add - num_models, 1);
//     }
//
//     #[test]
//     fn fails_to_add_model_in_the_local_model_store_when_the_model_path_is_wrong() {
//         let model_dir = "tests/model_storage/local_model_store";
//         let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
//
//         // load models
//         let result = local_model_store.fetch_models();
//         assert!(result.is_ok());
//
//         // add model
//         let add = local_model_store.add_model(
//             "my_awesome_penguin_model".to_string(),
//             "tests/model_storage/local_model_store/model_which_does_not_exist",
//         );
//
//         // assert
//         assert!(add.is_err());
//     }
// }
