use crate::model::predictor::{ModelInput, Output, Predictor, Value, Values};

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
    /// * `input` - The `ModelInput` to be parsed.
    ///
    /// # Returns
    /// * `Ok(TorchModelInput)` - If parsing was successful.
    /// * `Err(anyhow::Error)` - If there was an error during parsing.
    #[tracing::instrument(skip(input))]
    fn parse(input: ModelInput) -> anyhow::Result<Self> {
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
                            anyhow::bail!("not supported string tensors")
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

        let tensor = tch::Tensor::from_slice2(&numerical_features).tr();

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

impl Predictor for Torch {
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
            Ok(predictions) => {
                let predictions: Vec<Vec<f64>> = predictions.try_into()?;
                Ok(Output { predictions })
            }
            Err(e) => anyhow::bail!(
                "Failed to make predictions using Torch model: {}",
                e.to_string()
            ),
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

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 3 i.e [[1,2,3], [1,2,3], [1,2,3]]. This is because this is multi class classification
        // model with 3 classes
        assert_eq!(predictions.first().unwrap().len(), 3);
    }
}
