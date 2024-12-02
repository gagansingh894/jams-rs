use crate::model::input::ModelInput;
use crate::model::output::ModelOutput;
use crate::model::predict::Predict;

pub struct XGBoost {}

impl XGBoost {
    #[allow(dead_code)]
    pub fn load(_: &str) -> anyhow::Result<Self> {
        todo!()
    }
}

impl Predict for XGBoost {
    fn predict(&self, _: ModelInput) -> anyhow::Result<ModelOutput> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
