use crate::model::predictor::{ModelInput, Output, Predictor, Value, Values};
use lgbm;
use lgbm::mat::MatLayouts;
use lgbm::mat::MatLayouts::ColMajor;
use lgbm::PredictType::Normal;
use lgbm::{MatBuf, Parameters};

/// Struct representing the input data format for a LightGBM model.
///
/// This struct encapsulates numerical features parsed from a `ModelInput`.
/// It prepares these features into a format suitable for feeding into a LightGBM model.
struct LightGBMModelInput {
    /// Matrix buffer containing numerical features in column-major order.
    pub matbuf: MatBuf<f32, MatLayouts>,
}

impl LightGBMModelInput {
    /// Parses the input `ModelInput` into a `LightGBMModelInput`.
    ///
    /// This method extracts numerical features from the `ModelInput` and converts them
    /// into a format suitable for a LightGBM model, represented by a `MatBuf`.
    ///
    /// # Arguments
    ///
    /// * `input` - The `ModelInput` containing the input values.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if the input contains unsupported string values or if there are
    /// issues converting or organizing the numerical features.
    #[tracing::instrument(skip(input))]
    pub fn parse(input: ModelInput) -> anyhow::Result<Self> {
        let mut numerical_features: Vec<Vec<f32>> = Vec::new();

        // extract the values from hashmap
        let input_matrix: Vec<Values> = input.values();

        for values in input_matrix {
            // get the value type
            match values.0.first() {
                Some(first) => {
                    // strings values are pushed to separate vector of type Vec<String>
                    // int and float are pushed to separate of type Vec<f32>
                    match first {
                        Value::String(_) => {
                            anyhow::bail!("string type as input feature is not supported")
                        }
                        Value::Int(_) => {
                            let ints = values.to_ints()?;
                            // convert to float
                            let floats = ints.into_iter().map(|x| x as f32).collect();
                            numerical_features.push(floats);
                        }
                        Value::Float(_) => {
                            numerical_features.push(values.to_floats()?);
                        }
                    }
                }
                None => {
                    anyhow::bail!("failed to get first value ❌")
                }
            }
        }

        if numerical_features.is_empty() {
            anyhow::bail!("input is empty")
        }

        // determine the number of rows and columns in the matrix
        let (rows, cols) = (
            numerical_features.len(),
            numerical_features.first().unwrap().len(),
        );

        // flatten and convert numerical features into a single vector
        let flatten_values = numerical_features.into_iter().flatten().collect();

        // create a MatBuf in column-major order
        let matbuf = MatBuf::from_vec(flatten_values, cols, rows, ColMajor);

        Ok(Self { matbuf })
    }
}

/// Struct representing a predictor using a LightGBM model.
///
/// This struct encapsulates a LightGBM model booster, allowing for loading and prediction.
pub struct LightGBM {
    /// The LightGBM model booster.
    pub booster: lgbm::Booster,
}

impl LightGBM {
    /// Loads a LightGBM model from the specified file path.
    ///
    /// # Arguments
    ///
    /// * `path` - A string slice that holds the path to the LightGBM model file.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if there is an issue loading the LightGBM model from the file.
    #[tracing::instrument]
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let model = match lgbm::Booster::from_file(path.as_ref()) {
            Ok(model) => model,
            Err(e) => {
                anyhow::bail!("Failed to load LightGBM model from file {}: {}", path, e);
            }
        };
        Ok(LightGBM { booster: model.0 })
    }
}

impl Predictor for LightGBM {
    /// Performs prediction using the loaded LightGBM model.
    ///
    /// # Arguments
    ///
    /// * `input` - The `ModelInput` containing the input values for prediction.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if there is an issue parsing the input or performing the prediction.
    #[tracing::instrument(skip(self, input))]
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output> {
        let input = LightGBMModelInput::parse(input)?;
        let p = Parameters::new();
        let preds = self
            .booster
            .predict_for_mat(input.matbuf, Normal, 0, None, &p);
        match preds {
            Ok(predictions) => {
                // Convert predictions into Vec<Vec<f64>> format
                let predictions: Vec<Vec<f64>> =
                    predictions.values().iter().map(|v| vec![*v]).collect();
                Ok(Output { predictions })
            }
            Err(e) => anyhow::bail!("Failed to make predictions using LightGBM: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_utils;

    #[test]
    fn fails_to_load_lightgbm_model() {
        let model_dir = "incorrect/path";
        let model = LightGBM::load(model_dir);

        // assert the result is Ok
        assert!(model.is_err())
    }

    #[test]
    fn successfully_load_lightgbm_regressor_model() {
        let path = "tests/model_storage/models/lightgbm-my_awesome_reg_model.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_lightgbm_regressor_model() {
        let path = "tests/model_storage/models/lightgbm-my_awesome_reg_model.txt";
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
    fn fails_to_make_prediction_using_lightgbm_when_input_is_empty() {
        let path = "tests/model_storage/models/lightgbm-my_awesome_reg_model.txt";
        let model = LightGBM::load(path).unwrap();

        // lightgbm models do not support string input features. They have to preprocessed if the
        // model is using a string feature
        let size = 0;
        let model_inputs = test_utils::utils::create_model_inputs(28, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert the result is err
        assert!(output.is_err());
    }

    #[test]
    fn successfully_make_batch_predictions_using_lightgbm_regressor_model() {
        let path = "tests/model_storage/models/lightgbm-my_awesome_reg_model.txt";
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
        let path = "tests/model_storage/models/lightgbm-my_awesome_binary_model_2.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_lightgbm_binary_classifier_model() {
        let path = "tests/model_storage/models/lightgbm-my_awesome_binary_model_2.txt";
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
        let path = "tests/model_storage/models/lightgbm-my_awesome_binary_model_2.txt";
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
        let path = "tests/model_storage/models/lightgbm-my_awesome_xen_binary_model.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_lightgbm_xentropy_binary_classifier_model() {
        let path = "tests/model_storage/models/lightgbm-my_awesome_xen_binary_model.txt";
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
        let path = "tests/model_storage/models/lightgbm-my_awesome_xen_binary_model.txt";
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
        let path = "tests/model_storage/models/lightgbm-my_awesome_binary_model_2.txt";
        let model = LightGBM::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_lightgbm_xentropy_prob_classifier_model() {
        let path = "tests/model_storage/models/lightgbm-my_awesome_xen_prob_model.txt";
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
        let path = "tests/model_storage/models/lightgbm-my_awesome_xen_prob_model.txt";
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
