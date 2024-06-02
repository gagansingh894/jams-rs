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
    fn parse(input: ModelInput) -> anyhow::Result<Self> {
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
                    anyhow::bail!("not supported string as input features")
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

        Ok(Self { matbuf })
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
        let input = LightGBMModelInput::parse(input)?;
        let p = Parameters::new();
        let preds = self
            .booster
            .predict_for_mat(&input.matbuf, Normal, 0, None, &p);
        match preds {
            Ok(predictions) => {
                let predictions: Vec<Vec<f64>> =
                    predictions.values().iter().map(|v| vec![*v]).collect();
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
        let model = LightGBM::load(path).unwrap();

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let size = 1;
        let model_inputs = test_utils::utils::create_model_inputs(28, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[1]]. This is because this is a regression model with single output
        // and batch size of 1
        assert_eq!(predictions.first().unwrap().len(), 1);
    }

    #[test]
    fn successfully_make_batch_predictions_using_lightgbm_regressor_model() {
        let path = "tests/model_artefacts/lgbm_reg.txt";
        let model = LightGBM::load(path).unwrap();

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(28, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[1], [2], [3]]. This is because this is a regression model with single output
        // and batch size of 10
        assert_eq!(predictions.first().unwrap().len(), 1);
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
        let model = LightGBM::load(path).unwrap();

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let size = 1;
        let model_inputs = test_utils::utils::create_model_inputs(2, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[0,4]]. This is because this is a binary classification model with single output
        // and the output is probability value whose value lies between 0 and 1. The user can set
        // its own threshold.
        assert_eq!(predictions.first().unwrap().len(), 1);
    }

    #[test]
    fn successfully_make_batch_predictions_using_lightgbm_binary_classifier_model() {
        let path = "tests/model_artefacts/lgbm_binary.txt";
        let model = LightGBM::load(path).unwrap();

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(2, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[0,4], [0.1], [0.2]]. This is because this is a binary classification model with single output
        // and the output is probability value whose value lies between 0 and 1. The user can set
        // its own threshold.
        assert_eq!(predictions.first().unwrap().len(), 1);
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
        let model = LightGBM::load(path).unwrap();

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let size = 1;
        let model_inputs = test_utils::utils::create_model_inputs(2, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[0,4]]. This is because this is a binary classification model with single output
        // and the output is probability value whose value lies between 0 and 1. The user can set
        // its own threshold.
        assert_eq!(predictions.first().unwrap().len(), 1);
    }

    #[test]
    fn successfully_make_batch_predictions_using_lightgbm_xentropy_binary_classifier_model() {
        let path = "tests/model_artefacts/lgbm_xen_binary.txt";
        let model = LightGBM::load(path).unwrap();

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(2, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[0,4], [0.1], [0.2]]. This is because this is a binary classification model with single output
        // and the output is probability value whose value lies between 0 and 1. The user can set
        // its own threshold.
        assert_eq!(predictions.first().unwrap().len(), 1);
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
        let model = LightGBM::load(path).unwrap();

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let size = 1;
        let model_inputs = test_utils::utils::create_model_inputs(2, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[0,4]]. This is because this is a binary classification model with single output
        // and the output is probability value whose value lies between 0 and 1. The user can set
        // its own threshold.
        assert_eq!(predictions.first().unwrap().len(), 1);
    }

    #[test]
    fn successfully_make_batch_predictions_using_lightgbm_xentropy_prob_classifier_model() {
        let path = "tests/model_artefacts/lgbm_xen_prob.txt";
        let model = LightGBM::load(path).unwrap();

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(2, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert the result is ok
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[0,4], [0.1], [0.2]]. This is because this is a binary classification model with single output
        // and the output is probability value whose value lies between 0 and 1. The user can set
        // its own threshold.
        assert_eq!(predictions.first().unwrap().len(), 1);
    }
}
