use crate::model::predictor::{ModelInput, Predictor};
use lgbm;

pub struct LightGBM {
    booster: lgbm::Booster,
}

impl LightGBM {
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let model = lgbm::Booster::from_file(path.as_ref())
            .expect("failed to load lightGBM model fro  file");
        Ok(LightGBM { booster: model.0 })
    }
}

impl Predictor for LightGBM {
    fn predict(&self, input: ModelInput) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_load_lightgbm_regressor_model() {
        let path = "tests/model_artefacts/lgbm_reg.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_load_lightgbm_binary_classifier_model() {
        let path = "tests/model_artefacts/lgbm_binary.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_load_lightgbm_xentropy_binary_classifier_model() {
        let path = "tests/model_artefacts/lgbm_xen_binary.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_load_lightgbm_xentropy_probability_classifier_model() {
        let path = "tests/model_artefacts/lgbm_binary.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }
}
