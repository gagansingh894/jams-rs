use crate::model::predict::{ModelInput, Output, Predict, DEFAULT_OUTPUT_KEY};
use std::collections::HashMap;

use catboost_rs;

/// Struct representing input data for a Catboost model.
struct CatboostModelInput {
    /// Numeric features as a 2D vector.
    pub numeric_features: Vec<Vec<f32>>,
    /// Categorical features as a 2D vector of strings.
    pub categorical_features: Vec<Vec<String>>,
}

impl CatboostModelInput {
    /// Parses the input `ModelInput` into `CatboostModelInput`.
    ///
    /// # Arguments
    ///
    /// * `model_input` - The `ModelInput` containing input values.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if there is an issue parsing the input data.
    #[tracing::instrument(skip(model_input))]
    pub fn parse(mut model_input: ModelInput) -> anyhow::Result<Self> {
        // only float features are supported, so we are converting Vec<i32> to Vec<f32>
        // we can use .1 of any features as the length is same.
        // only the number of features is changing
        let numerical_features_shape = (
            model_input.integer_features.shape.0 + model_input.float_features.shape.0,
            model_input.float_features.shape.1,
        );

        // convert integer to float
        let mut converted: Vec<f32> = model_input
            .integer_features
            .values
            .into_iter()
            .map(|x| x as f32)
            .collect();

        // reuse the float vector by appending new values
        model_input.float_features.values.append(&mut converted);

        // we will use nd array to perform transpose operation
        let categorical_nd = match model_input.string_features.values.is_empty() {
            true => ndarray::Array2::<String>::default((1, 1)),
            false => ndarray::Array2::<String>::from_shape_vec(
                model_input.string_features.shape,
                model_input.string_features.values,
            )?,
        };

        let numeric_nd = match model_input.float_features.values.is_empty() {
            true => ndarray::Array2::<f32>::default((1, 1)),
            false => ndarray::Array2::<f32>::from_shape_vec(
                numerical_features_shape,
                model_input.float_features.values,
            )?,
        };

        Ok(CatboostModelInput {
            numeric_features: numeric_nd
                .t()
                .outer_iter()
                .map(|row| row.to_vec())
                .collect(),
            categorical_features: categorical_nd
                .t()
                .outer_iter()
                .map(|row| row.to_vec())
                .collect(),
        })
    }
}

/// Struct representing a Catboost model.
pub struct Catboost {
    /// The loaded Catboost model.
    model: catboost_rs::Model,
}

impl Catboost {
    /// Loads a Catboost model from the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - A string slice that holds the path to the Catboost model file (.cbm).
    ///
    /// # Errors
    ///
    /// Returns an `Err` if loading the Catboost model fails.
    #[tracing::instrument]
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let model = match catboost_rs::Model::load(path) {
            Ok(model) => model,
            Err(e) => {
                tracing::error!("Failed to load Catboost model from file {}: {}", path, e);
                anyhow::bail!("Failed to load Catboost model from file {}: {}", path, e)
            }
        };
        Ok(Catboost { model })
    }
}

impl Predict for Catboost {
    /// Predicts output based on the given input using the Catboost model.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data for prediction.
    ///
    /// # Errors
    ///
    /// Returns an `Err` if there is an issue with parsing the input or making predictions.
    #[tracing::instrument(skip(self, input))]
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output> {
        let input = CatboostModelInput::parse(input)?;
        let preds = self
            .model
            .calc_model_prediction(input.numeric_features, input.categorical_features);
        match preds {
            Ok(preds) => {
                let mut predictions: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
                let values: Vec<Vec<f64>> = preds.into_iter().map(|v| vec![v]).collect();
                predictions.insert(DEFAULT_OUTPUT_KEY.to_string(), values);
                Ok(Output { predictions })
            }
            Err(e) => {
                tracing::error!(
                    "Failed to make predictions using Catboost model: {}",
                    e.to_string()
                );

                anyhow::bail!(
                    "Failed to make predictions using Catboost model: {}",
                    e.to_string()
                )
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::test_utils;

    #[test]
    fn fails_to_load_catboost_model() {
        let model_dir = "incorrect/path";
        let model = Catboost::load(model_dir);

        // assert the result is Ok
        assert!(model.is_err())
    }

    #[test]
    fn successfully_load_catboost_regressor_model() {
        let path = "tests/model_storage/models/catboost-my_awesome_regressor_model";
        let model = Catboost::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_catboost_regressor_model() {
        let path = "tests/model_storage/models/catboost-my_awesome_regressor_model";
        let model = Catboost::load(path).unwrap();

        let size = 1;
        let model_inputs = test_utils::utils::create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            size,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;
        let predictions = predictions.get(DEFAULT_OUTPUT_KEY).unwrap();

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[1]]. This is because this is a regression model with single output
        // and batch size of 1
        assert_eq!(predictions.first().unwrap().len(), size);
    }

    #[test]
    fn successfully_make_batch_prediction_using_catboost_regressor_model() {
        let path = "tests/model_storage/models/catboost-my_awesome_regressor_model";
        let model = Catboost::load(path).unwrap();

        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            size,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;
        let predictions = predictions.get(DEFAULT_OUTPUT_KEY).unwrap();

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[1], [2], [3]]. This is because this is a regression model with single output
        // and batch size of 10
        assert_eq!(predictions.first().unwrap().len(), 1);
    }

    #[test]
    fn successfully_load_catboost_binary_classification_model() {
        let path = "tests/model_storage/models/catboost-my_awesome_binary_model";
        let model = Catboost::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_catboost_binary_model() {
        let path = "tests/model_storage/models/catboost-my_awesome_binary_model";
        let model = Catboost::load(path).unwrap();

        let size = 1;
        let model_inputs = test_utils::utils::create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            size,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;
        let predictions = predictions.get(DEFAULT_OUTPUT_KEY).unwrap();

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[0.5]]. This is because this is a binary classification model with single output
        // and the output is probability value whose value lies between 0 and 1. The user can set
        // its own threshold.
        assert_eq!(predictions.first().unwrap().len(), 1);
    }

    #[test]
    fn successfully_make_batch_prediction_using_catboost_binary_model() {
        let path = "tests/model_storage/models/catboost-my_awesome_binary_model";
        let model = Catboost::load(path).unwrap();

        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            size,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;
        let predictions = predictions.get(DEFAULT_OUTPUT_KEY).unwrap();

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[0,4], [0.1], [0.2]]. This is because this is a binary classification model with single output
        // and the output is probability value whose value lies between 0 and 1. The user can set
        // its own threshold.
        assert_eq!(predictions.first().unwrap().len(), 1);
    }

    #[test]
    fn successfully_load_catboost_multiclass_classification_model() {
        let path = "tests/model_storage/models/catboost-my_awesome_multiclass_model";
        let model = Catboost::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    // todo: investigate why multiclass classification fails for single prediction
    // #[test]
    // fn successfully_make_single_prediction_using_catboost_multiclass_classification_model() {
    //     let path = "tests/model_artefacts/catboost_multiclass";
    //     let model = Catboost::load(path).unwrap();
    //
    //     let model_inputs = create_model_inputs(
    //         model.model.get_float_features_count(),
    //         model.model.get_cat_features_count(),
    //         1,
    //     );
    //
    //     // make predictions
    //     let output = model.predict(model_inputs);
    //
    //     // assert
    //     assert!(output.is_ok());
    //     assert_eq!(output.unwrap().predictions.len(), 1)
    // }

    // todo: investigate why multiclass classification fails for batch prediction
    // #[test]
    // fn successfully_make_batch_prediction_using_catboost_multiclass_classification_model() {
    //     let path = "tests/model_artefacts/catboost_multiclass";
    //     let model = Catboost::load(path).unwrap();
    //
    //     let model_inputs = create_model_inputs(
    //         model.model.get_float_features_count(),
    //         model.model.get_cat_features_count(),
    //         10,
    //     );
    //
    //     // make predictions
    //     let output = model.predict(model_inputs);
    //
    //     // assert
    //     assert!(output.is_ok());
    //     assert_eq!(output.unwrap().predictions.len(), 1)
    // }
}
