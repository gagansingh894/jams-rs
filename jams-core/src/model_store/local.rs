use crate::model;
use crate::model::frameworks::{CATBOOST, LIGHTGBM, PYTORCH, TENSORFLOW, TORCH};
use crate::model::predictor::Predictor;
use crate::model_store::storage::{ModelName, Storage};
use dashmap::DashMap;
use std::fs;
use std::sync::Arc;

pub struct LocalModelStore {
    pub models: DashMap<ModelName, Arc<dyn Predictor>>,
    pub model_dir: String,
}

impl LocalModelStore {
    #[allow(dead_code)]
    fn new(model_dir: String) -> anyhow::Result<Self> {
        let models: DashMap<ModelName, Arc<dyn Predictor>> = DashMap::new();
        Ok(LocalModelStore { models, model_dir })
    }
}

impl Storage for LocalModelStore {
    fn fetch_models(&self) -> anyhow::Result<()> {
        // iterate through all the models in the dir
        // the models have a specif name format which would
        // allow us to identify model framework
        let dir = fs::read_dir(&self.model_dir).unwrap();
        for file in dir {
            let file = file.unwrap();
            let file_path = file.path();
            let file_name = file.file_name();
            let full_path = file_path.to_str().unwrap();
            if file_name.to_str().unwrap().to_string().contains(TENSORFLOW) {
                let prefix = format!("{}-", TENSORFLOW);
                let model_name = file_name.to_str().unwrap().to_string();
                let model_name = model_name.strip_prefix(prefix.as_str()).unwrap();
                let model = model::tensorflow::Tensorflow::load(full_path).unwrap();
                self.models.insert(model_name.to_string(), Arc::new(model));
            } else if file_name.to_str().unwrap().to_string().contains(TORCH) {
                let prefix_torch = format!("{}-", TORCH);
                let prefix_pytorch = format!("{}-", PYTORCH);
                let model_name = file_name.to_str().unwrap().to_string();
                let model_name = match model_name.strip_prefix(prefix_torch.as_str()) {
                    None => model_name.strip_prefix(prefix_pytorch.as_str()).unwrap(),
                    Some(v) => v,
                };

                let model = model::torch::Torch::load(full_path).unwrap();
                self.models.insert(model_name.to_string(), Arc::new(model));
            } else if file_name.to_str().unwrap().to_string().contains(CATBOOST) {
                let prefix = format!("{}-", CATBOOST);
                let model_name = file_name.to_str().unwrap().to_string();
                let model_name = model_name.strip_prefix(prefix.as_str()).unwrap();
                let model = model::catboost::Catboost::load(full_path).unwrap();
                self.models.insert(model_name.to_string(), Arc::new(model));
            } else if file_name.to_str().unwrap().to_string().contains(LIGHTGBM) {
                let prefix = format!("{}-", LIGHTGBM);
                let model_name = file_name.to_str().unwrap().to_string();
                let model_name = model_name.strip_prefix(prefix.as_str()).unwrap();
                let model = model::lightgbm::LightGBM::load(full_path).unwrap();
                self.models.insert(model_name.to_string(), Arc::new(model));
            } else {
                anyhow::bail!("unexpected model framework encountered")
            }
        }
        Ok(())
    }

    fn get_model(&self, _: ModelName) -> Arc<dyn Predictor> {
        todo!()
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
        println!("{:?}", local_model_store.models.len());
    }
}
