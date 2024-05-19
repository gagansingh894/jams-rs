use crate::model::predictor::{ModelInput, Predictor};

use catboost_rs;

pub struct Catboost {
    model: catboost_rs::Model,
}

impl Catboost {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let model =
            catboost_rs::Model::load(path).expect("failed to load catboost model from .cbm file");
        Ok(Catboost { model })
    }
}

impl Predictor for Catboost {
    fn predict(&self, input: ModelInput) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_load_catboost_regressor_model() {
        let path = "tests/model_artefacts/catboost_regressor";
        let model = Catboost::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_load_catboost_binary_classification_model() {
        let path = "tests/model_artefacts/catboost_binary";
        let model = Catboost::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_load_catboost_multiclass_classification_model() {
        let path = "tests/model_artefacts/catboost_multiclass";
        let model = Catboost::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }
}
