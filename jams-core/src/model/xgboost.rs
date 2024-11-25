use crate::model::predict::{ModelInput, Output, Predict};

pub struct XGBoost {}

impl XGBoost {
    #[allow(dead_code)]
    pub fn load(_: &str) -> anyhow::Result<Self> {
        todo!()
    }
}

impl Predict for XGBoost {
    fn predict(&self, _: ModelInput) -> anyhow::Result<Output> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
