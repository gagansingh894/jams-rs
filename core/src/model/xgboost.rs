use crate::model::predictor::{ModelInput, Output, Predictor};

pub struct XGBoost {}

impl XGBoost {
    #[allow(dead_code)]
    pub fn load(_: &str) -> anyhow::Result<Self> {
        todo!()
    }
}

impl Predictor for XGBoost {
    fn predict(&self, _: ModelInput) -> anyhow::Result<Output> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
