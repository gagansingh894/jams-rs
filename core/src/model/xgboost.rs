use crate::model::predictor::{ModelInput, Predictor};

pub struct XGBoost {}

impl XGBoost {
    pub fn load() -> anyhow::Result<Self> {
        todo!()
    }
}

impl Predictor for XGBoost {
    fn predict(input: ModelInput) {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
