use crate::model_store::storage::{
    extract_framework_from_path, load_models, load_predictor, Metadata, Model, ModelName, Storage,
};
use async_trait::async_trait;
use chrono::Utc;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use std::sync::Arc;
use tokio::task;

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
        let models = match model_dir.is_empty() {
            true => {
                log::warn!("No model directory specified, hence no models will be loaded ⚠️");
                let models: DashMap<ModelName, Arc<Model>> = DashMap::new();
                models
            }
            false => match load_models(model_dir.clone()) {
                Ok(models) => {
                    log::info!("Successfully fetched valid models from directory ✅");
                    models
                }
                Err(e) => {
                    anyhow::bail!("Failed to fetch models - {}", e.to_string());
                }
            },
        };

        Ok(LocalModelStore { models, model_dir })
    }
}

#[async_trait]
impl Storage for LocalModelStore {
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
    async fn add_model(&self, model_name: ModelName, model_path: &str) -> anyhow::Result<()> {
        let model_name = model_name.clone();
        let model_path = model_path.to_string();
        let models = self.models.clone();

        match task::spawn_blocking(move || {
            // Blocking code inside spawn_blocking closure
            match extract_framework_from_path(model_path.to_string()) {
                None => {
                    anyhow::bail!("Failed to extract framework from path");
                }
                Some(framework) => match load_predictor(framework, model_path.as_str()) {
                    Ok(predictor) => {
                        let now = Utc::now();
                        let model = Model::new(
                            predictor,
                            model_name.clone(),
                            framework,
                            model_path.to_string(),
                            now.to_rfc2822(),
                        );
                        models.insert(model_name, Arc::from(model));
                        Ok(())
                    }
                    Err(e) => {
                        anyhow::bail!("Failed to add new model: {e}")
                    }
                },
            }
        })
        .await
        {
            Ok(result) => match result {
                Ok(_) => Ok(()),
                Err(_) => {
                    anyhow::bail!("Failed to add model")
                }
            },
            Err(_) => {
                anyhow::bail!("Failed to add model due to error in task::spawn_blocking")
            }
        }
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
    async fn update_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        let model_name = model_name.clone();
        let models = self.models.clone();
        match task::spawn_blocking(move || {
            // By calling remove on the hashmap, the object is returned on success/
            // We use the returned object, in this case the model to extract the framework and model path
            match models.remove(model_name.as_str()) {
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
                            models.insert(model_name.clone(), Arc::new(model));
                            Ok(())
                        }
                        Err(e) => {
                            anyhow::bail!(
                                "Failed to update the specified model {}: {}",
                                model_name,
                                e
                            )
                        }
                    }
                }
            }
        })
        .await
        {
            Ok(result) => match result {
                Ok(_) => Ok(()),
                Err(_) => {
                    anyhow::bail!("Failed to update model")
                }
            },
            Err(_) => {
                anyhow::bail!("Failed to update model due to error in task::spawn_blocking")
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
        let model: Vec<Metadata> = self
            .models
            .iter()
            .map(|f| f.value().info.to_owned())
            .collect();
        Ok(model)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_load_models_from_different_frameworks_into_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";

        // initialize and load models
        let local_model_store = LocalModelStore::new(model_dir.to_string());

        // assert
        assert!(local_model_store.is_ok());
        assert_ne!(local_model_store.unwrap().models.len(), 0);
    }

    #[test]
    fn successfully_get_model_from_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";

        // load models
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
        let model = local_model_store.get_model("my_awesome_autompg_model".to_string());

        // assert
        assert!(model.is_some());
    }

    #[test]
    fn fails_to_get_model_from_local_model_store_when_model_name_is_wrong() {
        let model_dir = "tests/model_storage/local_model_store";

        // load models
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
        let model = local_model_store.get_model("model_which_does_not_exist".to_string());

        // assert
        assert!(model.is_none());
    }

    #[test]
    fn successfully_get_models_from_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";

        // load models
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
        let models = local_model_store.get_models();

        // assert
        assert!(models.is_ok());
        assert_ne!(models.unwrap().len(), 0);
    }

    #[test]
    fn successfully_deletes_model_in_the_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";

        // load models
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
        let deletion = local_model_store.delete_model("my_awesome_autompg_model".to_string());

        // assert
        assert!(deletion.is_ok());
    }

    #[test]
    fn fails_to_deletes_model_in_the_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";

        // load models
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
        let deletion = local_model_store.delete_model("model_which_does_not_exist".to_string());

        // assert
        assert!(deletion.is_err());
    }

    #[tokio::test]
    async fn successfully_update_model_in_the_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";
        let model_name = "my_awesome_autompg_model".to_string();

        // load models
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();

        // retrieve timestamp from existing to model for assertion
        let model = local_model_store
            .get_model(model_name.clone())
            .unwrap()
            .to_owned();

        // update model
        let update = local_model_store.update_model(model_name.clone()).await;
        assert!(update.is_ok());
        let updated_model = local_model_store
            .get_model(model_name.clone())
            .unwrap()
            .to_owned();

        // assert
        assert_eq!(model.info.name, updated_model.info.name);
        assert_eq!(model.info.path, updated_model.info.path);
        assert_ne!(model.info.last_updated, updated_model.info.last_updated); // as model will be updated
    }

    #[tokio::test]
    async fn fails_to_update_model_in_the_local_model_store_when_model_name_is_incorrect() {
        let model_dir = "tests/model_storage/local_model_store";
        let incorrect_model_name = "my_awesome_autompg_model_incorrect".to_string();

        // load models
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();

        // update model with incorrect model name
        let update = local_model_store.update_model(incorrect_model_name).await;

        // assert
        assert!(update.is_err());
    }

    #[tokio::test]
    async fn successfully_add_model_in_the_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";

        // load models
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();

        // delete model to set up test
        local_model_store
            .delete_model("my_awesome_penguin_model".to_string())
            .unwrap();
        // assert that model is not present
        let model = local_model_store.get_model("my_awesome_penguin_model".to_string());
        assert!(model.is_none());
        let num_models = local_model_store.get_models().unwrap().len();

        // add model
        let add = local_model_store
            .add_model(
                "my_awesome_penguin_model".to_string(),
                "tests/model_storage/local_model_store/tensorflow-my_awesome_penguin_model",
            )
            .await;
        let num_models_after_add = local_model_store.get_models().unwrap().len();

        // assert
        assert!(add.is_ok());
        let model = local_model_store.get_model("my_awesome_penguin_model".to_string());
        assert!(model.is_some());
        assert_eq!(num_models_after_add - num_models, 1);
    }

    #[tokio::test]
    async fn fails_to_add_model_in_the_local_model_store_when_the_model_path_is_wrong() {
        let model_dir = "tests/model_storage/local_model_store";

        // load models
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();

        // add model
        let add = local_model_store
            .add_model(
                "my_awesome_penguin_model".to_string(),
                "tests/model_storage/local_model_store/model_which_does_not_exist",
            )
            .await;

        // assert
        assert!(add.is_err());
    }
}
