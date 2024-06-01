use crate::model::predictor::{ModelInput, Output, Predictor, Value, Values};
use lgbm;
use lgbm::mat::MatLayouts;
use lgbm::mat::MatLayouts::ColMajor;
use lgbm::PredictType::Normal;
use lgbm::{MatBuf, Parameters};

struct LightGBMModelInput {
    matbuf: MatBuf<f32, MatLayouts>,
}

impl LightGBMModelInput {
    fn parse(input: ModelInput) -> Self {
        let mut numerical_features: Vec<Vec<f32>> = Vec::new();

        // extract the values from hashmap
        let input_matrix: Vec<Values> = input.values();

        for values in input_matrix {
            // get the value type
            let first = values.0.first().unwrap();

            // strings values are pushed to separate vector of type Vec<String>
            // int and float are pushed to separate of type Vec<f32>
            match first {
                Value::String(_) => {
                    panic!("not supported string as input features")
                }
                Value::Int(_) => {
                    let ints = values.to_ints();
                    // convert to float
                    let floats = ints.into_iter().map(|x| x as f32).collect();
                    numerical_features.push(floats);
                }
                Value::Float(_) => {
                    numerical_features.push(values.to_floats());
                }
            }
        }

        let (rows, cols) = (
            numerical_features.len(),
            numerical_features.first().unwrap().len(),
        );
        let flatten_values = numerical_features.into_iter().flatten().collect();
        // swapping rows and cols to meet input shape size
        let matbuf = MatBuf::from_vec(flatten_values, cols, rows, ColMajor);

        Self { matbuf }
    }
}

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
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output> {
        let input = LightGBMModelInput::parse(input);
        let p = Parameters::new();
        let preds = self
            .booster
            .predict_for_mat(&input.matbuf, Normal, 0, None, &p);
        match preds {
            Ok(predictions) => {
                let predictions: Vec<Vec<f64>> =
                    predictions.values().into_iter().map(|v| vec![*v]).collect();
                Ok(Output { predictions })
            }
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_utils;

    #[test]
    fn successfully_load_lightgbm_regressor_model() {
        let path = "tests/model_artefacts/lgbm_reg.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_lightgbm_regressor_model() {
        let path = "tests/model_artefacts/lgbm_reg.txt";
        let model = LightGBM::load(path);

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let model_inputs = test_utils::create_model_inputs(28, 0, 1);

        // make predictions
        let output = model.unwrap().predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok())
    }

    #[test]
    fn successfully_make_batch_predictions_using_lightgbm_regressor_model() {
        let path = "tests/model_artefacts/lgbm_reg.txt";
        let model = LightGBM::load(path);

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let model_inputs = test_utils::create_model_inputs(28, 0, 10);

        // make predictions
        let output = model.unwrap().predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok())
    }

    #[test]
    fn successfully_load_lightgbm_binary_classifier_model() {
        let path = "tests/model_artefacts/lgbm_binary.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_lightgbm_binary_classifier_model() {
        let path = "tests/model_artefacts/lgbm_binary.txt";
        let model = LightGBM::load(path);

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let model_inputs = test_utils::create_model_inputs(2, 0, 1);

        // make predictions
        let output = model.unwrap().predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok())
    }

    #[test]
    fn successfully_make_batch_predictions_using_lightgbm_binary_classifier_model() {
        let path = "tests/model_artefacts/lgbm_binary.txt";
        let model = LightGBM::load(path);

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let model_inputs = test_utils::create_model_inputs(2, 0, 10);

        // make predictions
        let output = model.unwrap().predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok())
    }

    #[test]
    fn successfully_load_lightgbm_xentropy_binary_classifier_model() {
        let path = "tests/model_artefacts/lgbm_xen_binary.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_lightgbm_xentropy_binary_classifier_model() {
        let path = "tests/model_artefacts/lgbm_xen_binary.txt";
        let model = LightGBM::load(path);

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let model_inputs = test_utils::create_model_inputs(2, 0, 1);

        // make predictions
        let output = model.unwrap().predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok())
    }

    #[test]
    fn successfully_make_batch_predictions_using_lightgbm_xentropy_binary_classifier_model() {
        let path = "tests/model_artefacts/lgbm_xen_binary.txt";
        let model = LightGBM::load(path);

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let model_inputs = test_utils::create_model_inputs(2, 0, 10);

        // make predictions
        let output = model.unwrap().predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok())
    }

    #[test]
    fn successfully_load_lightgbm_xentropy_probability_classifier_model() {
        let path = "tests/model_artefacts/lgbm_binary.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_lightgbm_xentropy_prob_classifier_model() {
        let path = "tests/model_artefacts/lgbm_xen_prob.txt";
        let model = LightGBM::load(path);

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let model_inputs = test_utils::create_model_inputs(2, 0, 1);

        // make predictions
        let output = model.unwrap().predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok())
    }

    #[test]
    fn successfully_make_batch_predictions_using_lightgbm_xentropy_prob_classifier_model() {
        let path = "tests/model_artefacts/lgbm_xen_prob.txt";
        let model = LightGBM::load(path);

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let model_inputs = test_utils::create_model_inputs(2, 0, 10);

        // make predictions
        let output = model.unwrap().predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok())
    }
}
