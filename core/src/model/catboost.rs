use crate::model::predictor::{ModelInput, Output, Predictor, Value, Values};

use catboost_rs;
use ndarray::Axis;

struct CatboostModelInput {
    numeric_features: Vec<Vec<f32>>,
    categorical_features: Vec<Vec<String>>,
}

impl CatboostModelInput {
    fn parse(input: ModelInput) -> Self {
        let mut categorical_features: Vec<Vec<String>> = Vec::new();
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
                    categorical_features.push(values.to_strings());
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

        create_catboost_model_inputs(categorical_features, numerical_features)
    }
}

fn create_catboost_model_inputs(
    categorical_features: Vec<Vec<String>>,
    numeric_features: Vec<Vec<f32>>,
) -> CatboostModelInput {
    // parse the 2d vecs to ndarrau
    let mut categorical_nd = ndarray::Array2::<String>::default(get_shape(&categorical_features));
    let mut numeric_nd = ndarray::Array2::<f32>::default(get_shape(&numeric_features));

    // populate if values are present
    if !categorical_features.is_empty() {
        for (i, mut row) in categorical_nd.axis_iter_mut(Axis(0)).enumerate() {
            for (j, col) in row.iter_mut().enumerate() {
                *col = categorical_features[i][j].clone();
            }
        }
    }

    // populate if values are present
    if !numeric_features.is_empty() {
        for (i, mut row) in numeric_nd.axis_iter_mut(Axis(0)).enumerate() {
            for (j, col) in row.iter_mut().enumerate() {
                *col = numeric_features[i][j];
            }
        }
    }

    CatboostModelInput {
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
    }
}

fn get_shape<T>(vector: &Vec<Vec<T>>) -> (usize, usize) {
    if vector.is_empty() {
        (1, 1)
    } else {
        (vector.len(), vector.first().unwrap().len())
    }
}

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
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output> {
        let input = CatboostModelInput::parse(input);
        let preds = self
            .model
            .calc_model_prediction(input.numeric_features, input.categorical_features);
        match preds {
            Ok(predictions) => {
                let predictions: Vec<Vec<f64>> = predictions.into_iter().map(|v| vec![v]).collect();
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
    fn successfully_load_catboost_regressor_model() {
        let path = "tests/model_artefacts/catboost_regressor";
        let model = Catboost::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_catboost_regressor_model() {
        let path = "tests/model_artefacts/catboost_regressor";
        let model = Catboost::load(path).unwrap();

        let size = 1;
        let model_inputs = test_utils::create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            size,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

        // asserts the output length of predictions is equal to input length
        assert_eq!(predictions.len(), size);
        // since the predictions is of type Vec<Vec<f64>>, we will assert that inner vec is of
        // length 1 i.e [[1]]. This is because this is a regression model with single output
        // and batch size of 1
        assert_eq!(predictions.first().unwrap().len(), size);
    }

    #[test]
    fn successfully_make_batch_prediction_using_catboost_regressor_model() {
        let path = "tests/model_artefacts/catboost_regressor";
        let model = Catboost::load(path).unwrap();

        let size = 10;
        let model_inputs = test_utils::create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            size,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
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
    fn successfully_load_catboost_binary_classification_model() {
        let path = "tests/model_artefacts/catboost_binary";
        let model = Catboost::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_single_prediction_using_catboost_binary_model() {
        let path = "tests/model_artefacts/catboost_binary";
        let model = Catboost::load(path).unwrap();

        let size = 1;
        let model_inputs = test_utils::create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            size,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        let predictions = output.unwrap().predictions;

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
        let path = "tests/model_artefacts/catboost_binary";
        let model = Catboost::load(path).unwrap();

        let size = 10;
        let model_inputs = test_utils::create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            size,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
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
    fn successfully_load_catboost_multiclass_classification_model() {
        let path = "tests/model_artefacts/catboost_multiclass";
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
