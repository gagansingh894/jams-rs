use crate::model::predictor::{ModelInput, Output, Predictor, Value, Values};

use tch::CModule;

struct TorchModelInput {
    tensor: tch::Tensor,
}

impl TorchModelInput {
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
                    anyhow::bail!("not supported string tensors")
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

        let tensor = tch::Tensor::from_slice2(&numerical_features).tr();

        Ok(Self { tensor })
    }
}

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
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output> {
        let input = TorchModelInput::parse(input)?;
        let preds = self.model.forward_ts(&[input.tensor]);
        match preds {
            Ok(predictions) => {
                let predictions: Vec<Vec<f64>> = predictions.try_into().unwrap();
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
    fn successfully_load_pytorch_regression_model() {
        let path = "tests/model_storage/local_model_store/pytorch-my_awesome_californiahousing_model.pt";
        let model = Torch::load(path);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_pytorch_regression_model_when_input_is_tabular_data() {
        let path = "tests/model_storage/local_model_store/pytorch-my_awesome_californiahousing_model.pt";
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
        let model_path = "tests/model_storage/local_model_store/torch-my_awesome_penguin_model.pt";
        let model = Torch::load(model_path);

        // assert the result is Ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_pytorch_multiclass_classification_model_when_input_is_tabular_data(
    ) {
        let path = "tests/model_storage/local_model_store/torch-my_awesome_penguin_model.pt";
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
