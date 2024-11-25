use crate::model::predict::{ModelInput, Output, Predict, Values, DEFAULT_OUTPUT_KEY};
use std::collections::HashMap;

use crate::MAX_CAPACITY;
use tch::CModule;

/// Struct representing the input for a Torch model.
///
/// # Fields
/// * `tensor` - The tensor representation of the model input.
struct TorchModelInput {
    tensor: tch::Tensor,
}

impl TorchModelInput {
    /// Parses a `ModelInput` into a `TorchModelInput`.
    ///
    /// # Arguments
    /// * `model_input` - The `ModelInput` to be parsed.
    ///
    /// # Returns
    /// * `Ok(TorchModelInput)` - If parsing was successful.
    /// * `Err(anyhow::Error)` - If there was an error during parsing.
    #[tracing::instrument(skip(model_input))]
    fn parse(model_input: ModelInput) -> anyhow::Result<Self> {
        let mut numerical_features: Vec<f32> = Vec::with_capacity(MAX_CAPACITY);

        // extract the values from hashmap
        let input_matrix: Vec<Values> = model_input.values();
        let num_cols = input_matrix.len();
        // it is okay to overwrite this on each loop as all the length of each row is same
        let mut num_rows: usize = 0;

        for values in input_matrix {
            match values {
                Values::String(_) => {
                    tracing::error!("string type as input feature is not supported");
                    anyhow::bail!("string type as input feature is not supported")
                }
                Values::Int(_) => {
                    let ints = match values.into_ints() {
                        None => {
                            tracing::error!("failed to convert input values to int vector");
                            anyhow::bail!("failed to convert input values to int vector")
                        }
                        Some(ints) => ints,
                    };
                    // convert to float
                    let floats: Vec<f32> = ints.into_iter().map(|x| x as f32).collect();
                    num_rows = floats.len();
                    numerical_features.extend(floats);
                }
                Values::Float(_) => {
                    let floats = match values.into_floats() {
                        None => {
                            tracing::error!("failed to convert input values to float vector");
                            anyhow::bail!("failed to convert input values to float vector")
                        }
                        Some(ints) => ints,
                    };
                    num_rows = floats.len();
                    numerical_features.extend(floats);
                }
            }
        }

        let tensor =
            tch::Tensor::from_slice(&numerical_features).view([num_rows as i64, num_cols as i64]);

        Ok(Self { tensor })
    }
}

/// Struct representing a Torch model.
///
/// # Fields
/// * `model` - The compiled Torch model.
pub struct Torch {
    model: CModule,
}

impl Torch {
    /// Loads a Torch model from the specified file path.
    ///
    /// # Arguments
    /// * `path` - The file path to the Torch model.
    ///
    /// # Returns
    /// * `Ok(Torch)` - If the model was successfully loaded.
    /// * `Err(anyhow::Error)` - If there was an error during loading.
    #[tracing::instrument(skip(path))]
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let model = match CModule::load(path) {
            Ok(model) => model,
            Err(e) => {
                tracing::error!(
                    "Failed to load pytorch model from file {}: {}",
                    path,
                    e.to_string()
                );

                anyhow::bail!(
                    "Failed to load pytorch model from file {}: {}",
                    path,
                    e.to_string()
                )
            }
        };
        Ok(Torch { model })
    }
}

impl Predict for Torch {
    /// Predicts the output for the given model input.
    ///
    /// # Arguments
    /// * `input` - The input data for the model.
    ///
    /// # Returns
    /// * `Ok(Output)` - The prediction output.
    /// * `Err(anyhow::Error)` - If there was an error during prediction.
    #[tracing::instrument(skip(self, input))]
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output> {
        let input = TorchModelInput::parse(input)?;
        let preds = self.model.forward_ts(&[input.tensor]);
        match preds {
            Ok(preds) => {
                let mut predictions: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
                let values: Vec<Vec<f64>> = preds.try_into()?;
                predictions.insert(DEFAULT_OUTPUT_KEY.to_string(), values);
                Ok(Output { predictions })
            }
            Err(e) => {
                tracing::error!(
                    "Failed to make predictions using Torch model: {}",
                    e.to_string()
                );

                anyhow::bail!(
                    "Failed to make predictions using Torch model: {}",
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
    fn fails_to_load_torch_model() {
        let model_dir = "incorrect/path";
        let model = Torch::load(model_dir);

        // assert the result is Ok
        assert!(model.is_err())
    }

    #[test]
    fn successfully_load_pytorch_regression_model() {
        let path = "tests/model_storage/models/pytorch-my_awesome_californiahousing_model.pt";
        let model = Torch::load(path);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_pytorch_regression_model_when_input_is_tabular_data() {
        let path = "tests/model_storage/models/pytorch-my_awesome_californiahousing_model.pt";
        let model = Torch::load(path).unwrap();

        // torch models do not support string input features. They have to preprocessed if the
        // model is using string features
        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(8, 0, size);

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
        assert_eq!(predictions.first().unwrap().len(), 1);
    }

    #[test]
    fn successfully_load_pytorch_multiclass_classification_model() {
        let model_path = "tests/model_storage/models/torch-my_awesome_penguin_model.pt";
        let model = Torch::load(model_path);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_pytorch_multiclass_classification_model_when_input_is_tabular_data(
    ) {
        let path = "tests/model_storage/models/torch-my_awesome_penguin_model.pt";
        let model = Torch::load(path).unwrap();

        // torch models do not support string input features. They have to preprocessed if the
        // model is using string features
        let size = 10;
        let model_inputs = test_utils::utils::create_model_inputs(4, 0, size);

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;
        let predictions = predictions.get(DEFAULT_OUTPUT_KEY).unwrap();

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 3 i.e [[1,2,3], [1,2,3], [1,2,3]]. This is because this is multi class classification
        // model with 3 classes
        assert_eq!(predictions.first().unwrap().len(), 3);
    }
}
