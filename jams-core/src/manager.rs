use crate::model_store::storage::Storage;
use std::sync::Arc;

#[allow(dead_code)]
pub struct Manager {
    pub model_store: Arc<dyn Storage>,
}

impl Manager {
    #[allow(dead_code)]
    fn new(model_store: Arc<dyn Storage>) -> anyhow::Result<Self> {
        Ok(Manager { model_store })
    }

    #[allow(dead_code)]
    fn predict(&self) {
        todo!()
    }
}
