use crate::model::predictor::{ModelInput, Output, Predictor};

pub struct XGBoost {}

impl XGBoost {
    pub fn load() -> anyhow::Result<Self> {
        todo!()
    }
}

impl Predictor for XGBoost {
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
