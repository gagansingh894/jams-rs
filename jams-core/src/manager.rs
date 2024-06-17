use crate::model::predictor::ModelInput;
use crate::model_store::storage::{Metadata, ModelName, Storage};
use std::sync::Arc;

/// Manages model storage and prediction requests.
///
/// The `Manager` struct is responsible for managing access to stored models and handling prediction requests.
/// It interacts with the model storage to fetch models and execute predictions.
///
/// # Fields
/// - `model_store` (Arc&ltdyn Storage&gt): A shared reference to the model storage.
pub struct Manager {
    model_store: Arc<dyn Storage>,
}

impl Manager {
    /// Creates a new `Manager` instance.
    ///
    /// This method initializes the `Manager` by fetching the available models from the model storage.
    ///
    /// # Arguments
    /// - `model_store` (Arc&ltdyn Storageglt): A shared reference to the model storage.
    ///
    /// # Returns
    /// - `Ok(Manager)`: If the models were successfully fetched.
    /// - `Err(anyhow::Error)`: If there was an error fetching the models.
    ///
    pub fn new(model_store: Arc<dyn Storage>) -> anyhow::Result<Self> {
        Ok(Manager { model_store })
    }

    /// Retrieves the names of all models stored in the model store.
    ///
    /// This method fetches and returns a vector of strings representing the names
    /// of all models currently stored in the model store.
    ///
    /// # Errors
    /// Returns an error if there are issues fetching the model names from the store.
    ///
    pub fn get_models(&self) -> anyhow::Result<Vec<Metadata>> {
        self.model_store.get_models()
    }

    /// Adds a new model to the model store.
    ///
    /// # Arguments
    ///
    /// * `model_name` - A `ModelName` representing the name of the model to be added.
    /// * `model_path` - A string slice representing the path to the model file.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the model is successfully added.
    /// * `Err(anyhow::Error)` if there is an error during the addition process.
    pub async fn add_model(&self, model_name: ModelName, model_path: &str) -> anyhow::Result<()> {
        self.model_store.add_model(model_name, model_path).await
    }

    /// Updates an existing model in the model store.
    ///
    /// # Arguments
    ///
    /// * `model_name` - A `ModelName` representing the name of the model to be updated.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the model is successfully updated.
    /// * `Err(anyhow::Error)` if there is an error during the update process or if the model does not exist.
    pub async fn update_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        self.model_store.update_model(model_name).await
    }

    /// Deletes an existing model from the model store.
    ///
    /// # Arguments
    ///
    /// * `model_name` - A `ModelName` representing the name of the model to be deleted.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the model is successfully deleted.
    /// * `Err(anyhow::Error)` if there is an error during the deletion process or if the model does not exist.
    pub fn delete_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        self.model_store.delete_model(model_name)
    }

    /// Predicts using the specified model and input data.
    ///
    /// This method fetches the specified model from the storage, parses the input data,
    /// and executes the prediction.
    ///
    /// # Arguments
    /// - `model_name` (ModelName): The name of the model to use for the prediction.
    /// - `input_json` (&str): The input data for the prediction, formatted as a JSON string.
    ///
    /// # Returns
    /// - `Ok(())`: If the prediction was successfully made.
    /// - `Err(anyhow::Error)`: If there was an error fetching the model, parsing the input, or making the prediction.
    ///
    pub fn predict(&self, model_name: ModelName, input_json: &str) -> anyhow::Result<String> {
        let model = self.model_store.get_model(model_name.clone());
        match model {
            None => {
                anyhow::bail!("No model exists for model name: {}", &model_name);
            }
            Some(model) => {
                // parse input
                match ModelInput::from_str(input_json) {
                    Ok(input) => {
                        // make predictions
                        let output = match model.predictor.predict(input) {
                            Ok(output) => output,
                            Err(e) => {
                                anyhow::bail!("Failed to make predictions: {}", e.to_string());
                            }
                        };

                        // parse output
                        match serde_json::to_string(&output) {
                            Ok(json) => Ok(json),
                            Err(e) => {
                                anyhow::bail!("Failed to parse predictions: {}", e.to_string());
                            }
                        }
                    }
                    Err(e) => {
                        anyhow::bail!("Failed to parse input: {}", e.to_string());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::local::LocalModelStore;

    #[tokio::test]
    async fn successfully_create_manager_with_local_model_store() {
        let model_dir = "./tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).await.unwrap();
        let manager = Manager::new(Arc::new(local_model_store));

        // assert
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn successfully_make_predictions_via_manager_with_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).await.unwrap();
        let manager = Manager::new(Arc::new(local_model_store)).unwrap();

        // dummy input
        let input = "{\"pclass\":[\"1\",\"3\",\"3\",\"3\",\"3\",\"3\",\"3\",\"3\",\"3\",\"1\"],\"sex\":[\"female\",\"female\",\"male\",\"female\",\"male\",\"female\",\"male\",\"male\",\"male\",\"male\"],\"age\":[22.0,23.79929292929293,32.0,23.79929292929293,14.0,2.0,22.0,28.0,23.79929292929293,23.79929292929293],\"sibsp\":[\"0\",\"1\",\"0\",\"8\",\"5\",\"4\",\"0\",\"0\",\"0\",\"0\"],\"parch\":[\"0\",\"0\",\"0\",\"2\",\"2\",\"2\",\"0\",\"0\",\"0\",\"0\"],\"fare\":[151.55,14.4542,7.925,69.55,46.9,31.275,7.8958,7.8958,7.8958,35.5],\"embarked\":[\"S\",\"C\",\"S\",\"S\",\"S\",\"S\",\"S\",\"S\",\"S\",\"S\"],\"class\":[\"First\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"First\"],\"who\":[\"woman\",\"woman\",\"man\",\"woman\",\"child\",\"child\",\"man\",\"man\",\"man\",\"man\"],\"adult_male\":[\"True\",\"False\",\"True\",\"False\",\"False\",\"False\",\"True\",\"True\",\"True\",\"True\"],\"deck\":[\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"C\"],\"embark_town\":[\"Southampton\",\"Cherbourg\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\"],\"alone\":[\"True\",\"False\",\"True\",\"False\",\"False\",\"False\",\"True\",\"True\",\"True\",\"True\"]}";
        let model_name: ModelName = "titanic_model".to_string(); // catboost model

        // assert
        let prediction = manager.predict(model_name, input);
        assert!(prediction.is_ok());
    }

    #[tokio::test]
    #[should_panic]
    // todo: check if we can exit gracefully
    async fn fail_to_make_predictions_via_manager_with_local_model_store_when_input_shape_is_wrong()
    {
        let model_dir = "tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).await.unwrap();
        let manager = Manager::new(Arc::new(local_model_store)).unwrap();

        // dummy input
        let input = "{\"pclass\":[\"3\",\"3\",\"3\",\"3\",\"3\",\"3\",\"3\",\"3\",\"1\"],\"sex\":[\"female\",\"female\",\"male\",\"female\",\"male\",\"female\",\"male\",\"male\",\"male\",\"male\"],\"age\":[22.0,23.79929292929293,32.0,23.79929292929293,14.0,2.0,22.0,28.0,23.79929292929293,23.79929292929293],\"sibsp\":[\"0\",\"1\",\"0\",\"8\",\"5\",\"4\",\"0\",\"0\",\"0\",\"0\"],\"parch\":[\"0\",\"0\",\"0\",\"2\",\"2\",\"2\",\"0\",\"0\",\"0\",\"0\"],\"fare\":[151.55,14.4542,7.925,69.55,46.9,31.275,7.8958,7.8958,7.8958,35.5],\"embarked\":[\"S\",\"C\",\"S\",\"S\",\"S\",\"S\",\"S\",\"S\",\"S\",\"S\"],\"class\":[\"First\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"First\"],\"who\":[\"woman\",\"woman\",\"man\",\"woman\",\"child\",\"child\",\"man\",\"man\",\"man\",\"man\"],\"adult_male\":[\"True\",\"False\",\"True\",\"False\",\"False\",\"False\",\"True\",\"True\",\"True\",\"True\"],\"deck\":[\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"C\"],\"embark_town\":[\"Southampton\",\"Cherbourg\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\"],\"alone\":[\"True\",\"False\",\"True\",\"False\",\"False\",\"False\",\"True\",\"True\",\"True\",\"True\"]}";
        let model_name: ModelName = "titanic_model".to_string(); // catboost model

        // assert
        let prediction = manager.predict(model_name, input);
        assert!(prediction.is_err());
    }

    #[tokio::test]
    async fn successfully_get_models_via_manager_with_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).await.unwrap();
        let manager = Manager::new(Arc::new(local_model_store)).unwrap();

        // models
        let models = manager.get_models();

        // assert
        assert!(models.is_ok());
        assert_ne!(manager.get_models().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn successfully_add_model_via_manager_with_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).await.unwrap();
        let manager = Manager::new(Arc::new(local_model_store)).unwrap();
        let model_name: ModelName = "my_awesome_penguin_model".to_string();
        let model_path =
            "tests/model_storage/local_model_store/tensorflow-my_awesome_penguin_model";

        // delete a model to add it back
        let num_model_before_deletion = manager.get_models().unwrap().len();
        manager.delete_model(model_name.clone()).unwrap();

        // add model
        let add = manager.add_model(model_name, model_path).await;

        // assert
        assert!(add.is_ok());
        assert_eq!(
            num_model_before_deletion,
            manager.get_models().unwrap().len()
        );
    }

    #[tokio::test]
    async fn successfully_update_model_via_manager_with_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).await.unwrap();
        let manager = Manager::new(Arc::new(local_model_store)).unwrap();
        let model_name: ModelName = "my_awesome_penguin_model".to_string();

        // update model
        let update = manager.update_model(model_name).await;

        // assert
        assert!(update.is_ok())
    }

    #[tokio::test]
    async fn successfully_delete_model_via_manager_with_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).await.unwrap();
        let manager = Manager::new(Arc::new(local_model_store)).unwrap();
        let model_name: ModelName = "my_awesome_penguin_model".to_string();

        // delete model
        let num_model_before_deletion = manager.get_models().unwrap().len();
        let delete = manager.delete_model(model_name);

        // assert
        assert!(delete.is_ok());
        assert_eq!(
            num_model_before_deletion - manager.get_models().unwrap().len(),
            1
        )
    }
}
