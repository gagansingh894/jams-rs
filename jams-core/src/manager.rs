use crate::model::predictor::ModelInput;
use crate::model_store::storage::{ModelName, Storage};
use std::sync::Arc;

#[allow(dead_code)]
pub struct Manager {
    model_store: Arc<dyn Storage>,
}

impl Manager {
    #[allow(dead_code)]
    fn new(model_store: Arc<dyn Storage>) -> anyhow::Result<Self> {
        match model_store.fetch_models() {
            Ok(_) => {
                println!("successfully fetched models")
            }
            Err(_) => {
                anyhow::bail!("unable to fetch models");
            }
        }
        Ok(Manager { model_store })
    }

    #[allow(dead_code)]
    fn predict(&self, model_name: ModelName, input_json: &str) -> anyhow::Result<()> {
        let model = self.model_store.get_model(model_name.clone());
        match model {
            None => {
                anyhow::bail!("no model exists for model name: {}", &model_name);
            }
            Some(model) => {
                // parse input
                match ModelInput::from_str(input_json) {
                    Ok(input) => {
                        let _ = model.predict(input)?;
                        Ok(())
                    }
                    Err(e) => {
                        anyhow::bail!("failed to make predictions: {}", e.to_string());
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
        let model_dir = "tests/model_storage/local_model_store";
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
