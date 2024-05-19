use crate::model::predictor::{ModelInput, Predictor};

use tch::CModule;

pub struct Torch {
    model: CModule,
}

impl Torch {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let model = CModule::load(path).expect("failed to load pytorch model from file");
        Ok(Torch { model })
    }
}

impl Predictor for Torch {
    fn predict(input: ModelInput) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_load_pytorch_regression_model() {
        let path = "tests/model_artefacts/californiahousing_pytorch.pt";
        let model = Torch::load(path);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_load_pytorch_multiclass_classification_model() {
        let model_path = "tests/model_artefacts/mnist_pytorch.pt";
        let model = Torch::load(model_path);

        // assert the result is Ok
        assert!(model.is_ok())
    }
}
