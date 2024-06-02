use std::sync::Arc;
use dashmap::DashMap;
use crate::model::predictor::Predictor;
use crate::model_store::storage::{ModelName, Storage};

pub struct LocalModelStore {
    pub models: DashMap<ModelName, Arc<dyn Predictor>>,
    pub model_dir: String
}

impl LocalModelStore {
    #[allow(dead_code)]
    fn new(model_dir: String) -> anyhow::Result<Self> {
        let models: DashMap<ModelName, Arc<dyn Predictor>> = DashMap::new();
        Ok(LocalModelStore {models, model_dir})
    }
}

impl Storage for LocalModelStore {
    fn fetch_models(&self) {
        // iterate through all the models in the dir
        // the models have a specif name format which would
        // allow us to identify model framework
        todo!()
    }

    fn get_model(&self, _: ModelName) -> Arc<dyn Predictor> {
        todo!()
    }
}