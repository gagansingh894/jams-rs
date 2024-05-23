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
        let input_matrix: Vec<Values> = input.get_values();

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
            Ok(predictions) => Ok(Output { predictions }),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::predictor::FeatureName;
    use std::collections::HashMap;

    fn create_model_inputs(
        num_numeric_features: usize,
        num_string_features: usize,
        size: usize,
    ) -> ModelInput {
        let mut model_input: HashMap<FeatureName, Values> = HashMap::new();
        let mut rng = rand::thread_rng();
        let random_string_values: Vec<String> = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];

        // create string features
        for i in 0..num_string_features {
            let mut string_features: Vec<Value> = Vec::new();

            for _ in 0..size {
                let value = random_string_values
                    .get(rng.gen_range(0..random_string_values.len()))
                    .unwrap()
                    .to_string();
                string_features.push(Value::String(value));
            }

            let feature_name = format!("string_feature_{}", i);
            model_input.insert(feature_name, Values(string_features));
        }

        // create numeric features
        for i in 0..num_numeric_features {
            let mut number_features: Vec<Value> = Vec::new();

            for _ in 0..size {
                let value = rng.gen::<f32>();
                number_features.push(Value::Float(value));
            }

            let feature_name = format!("numeric_feature_{}", i);
            model_input.insert(feature_name, Values(number_features));
        }

        ModelInput::from_hashmap(model_input).unwrap()
    }

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

        let model_inputs = create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            1,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        assert_eq!(output.unwrap().predictions.len(), 1)
    }

    #[test]
    fn successfully_make_batch_prediction_using_catboost_regressor_model() {
        let path = "tests/model_artefacts/catboost_regressor";
        let model = Catboost::load(path).unwrap();

        let model_inputs = create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            10,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        assert_eq!(output.unwrap().predictions.len(), 10)
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

        let model_inputs = create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            1,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        assert_eq!(output.unwrap().predictions.len(), 1)
    }

    #[test]
    fn successfully_make_batch_prediction_using_catboost_binary_model() {
        let path = "tests/model_artefacts/catboost_binary";
        let model = Catboost::load(path).unwrap();

        let model_inputs = create_model_inputs(
            model.model.get_float_features_count(),
            model.model.get_cat_features_count(),
            10,
        );

        // make predictions
        let output = model.predict(model_inputs);

        // assert
        assert!(output.is_ok());
        assert_eq!(output.unwrap().predictions.len(), 10)
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
