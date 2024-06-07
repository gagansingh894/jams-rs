use crate::model::predictor::ModelInput;
use crate::model_store::storage::{ModelName, Storage};
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
        match model_store.fetch_models() {
            Ok(_) => {
                log::info!("Successfully fetched valid models from directory ✅")
            }
            Err(e) => {
                anyhow::bail!("Failed to fetch models ❌ - {}", e.to_string());
            }
        }
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
    pub fn get_models(&self) -> anyhow::Result<Vec<String>> {
        self.model_store.get_model_names()
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
                        let output = match model.predict(input) {
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

    #[test]
    fn successfully_create_manager_with_local_model_store() {
        let model_dir = "./tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
        let manager = Manager::new(Arc::new(local_model_store));

        // assert
        assert!(manager.is_ok());
    }

    #[test]
    fn successfully_make_predictions_via_manager_with_local_model_store() {
        let model_dir = "tests/model_storage/local_model_store";
        let local_model_store = LocalModelStore::new(model_dir.to_string()).unwrap();
        let manager = Manager::new(Arc::new(local_model_store)).unwrap();

        // dummy input
        let input = "{\"pclass\":[\"1\",\"3\",\"3\",\"3\",\"3\",\"3\",\"3\",\"3\",\"3\",\"1\"],\"sex\":[\"female\",\"female\",\"male\",\"female\",\"male\",\"female\",\"male\",\"male\",\"male\",\"male\"],\"age\":[22.0,23.79929292929293,32.0,23.79929292929293,14.0,2.0,22.0,28.0,23.79929292929293,23.79929292929293],\"sibsp\":[\"0\",\"1\",\"0\",\"8\",\"5\",\"4\",\"0\",\"0\",\"0\",\"0\"],\"parch\":[\"0\",\"0\",\"0\",\"2\",\"2\",\"2\",\"0\",\"0\",\"0\",\"0\"],\"fare\":[151.55,14.4542,7.925,69.55,46.9,31.275,7.8958,7.8958,7.8958,35.5],\"embarked\":[\"S\",\"C\",\"S\",\"S\",\"S\",\"S\",\"S\",\"S\",\"S\",\"S\"],\"class\":[\"First\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"Third\",\"First\"],\"who\":[\"woman\",\"woman\",\"man\",\"woman\",\"child\",\"child\",\"man\",\"man\",\"man\",\"man\"],\"adult_male\":[\"True\",\"False\",\"True\",\"False\",\"False\",\"False\",\"True\",\"True\",\"True\",\"True\"],\"deck\":[\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"Unknown\",\"C\"],\"embark_town\":[\"Southampton\",\"Cherbourg\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\",\"Southampton\"],\"alone\":[\"True\",\"False\",\"True\",\"False\",\"False\",\"False\",\"True\",\"True\",\"True\",\"True\"]}";
        let model_name: ModelName = "titanic_model".to_string(); // catboost model

        // assert
        let prediction = manager.predict(model_name, input);
        assert!(prediction.is_ok());
    }
}
