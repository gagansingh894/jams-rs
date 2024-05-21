use crate::model::predictor::{ModelInput, Predictor, Value};
use std::vec;

use catboost_rs;
use rand::Rng;

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
    fn predict(&self, input: ModelInput) {
        let (float_features, cat_features) = parse_model_input(input);
        let pred = self.model.calc_model_prediction(float_features, cat_features);
    }
}

fn parse_model_input(input: ModelInput) -> (Vec<Vec<f32>>, Vec<Vec<String>>) {
    let mut float_features: Vec<Vec<f32>> = Vec::new();
    let mut cat_features: Vec<Vec<String>> = Vec::new();
    // num of features
    let num_features = input.iter().len();
    // initialize a new vector to hold the transposed data
    let mut transposed_features_values: Vec<Vec<Value>> = Vec::with_capacity(num_features);

    println!("creating a 2d vector");

    // convert to 2D vector
    let features_values: Vec<Vec<Value>> = input.clone().into_iter()
        .map(|(_, v)| v)
        .collect();

    println!("{:?}", features_values);
    println!("{}", num_features);
    println!("{}", features_values.len());

    // iterate over each column index
    for i in 0..num_features {
        println!("iteration {}", i);
        // create a new row vector for each column index
        let mut row: Vec<Value> = Vec::new();
        // iterate over each row (inner vector) to collect values from the current column
        for inner_vec in &features_values {
            println!("inner vec length{:?}", inner_vec.len());
            let value = inner_vec.get(i).unwrap();
            println!("{:?}", value);
            row.push(value.clone());
        }

        // push the completed row vector into the transposed_vec
        transposed_features_values.push(row);
    }

    for feature_values in transposed_features_values.iter() {
        // check the type of Value
        match feature_values.first().unwrap() {
            // create categorical features
            Value::String(_) => {
                let mut strings: Vec<String> = Vec::new();
                for value in feature_values {
                    if let Value::String(s) = value {
                        strings.push(s.to_string());
                    }
                }
                cat_features.push(strings);
            }
            // create numerical features. the Value::Int variant gets parsed as float32
            Value::Int(_) => {
                let mut numbers: Vec<f32> = Vec::new();
                for value in feature_values {
                    if let Value::Int(i) = value {
                        numbers.push(i.clone() as f32)
                    }
                }
                float_features.push(numbers);
            }
            // create numerical features. the Value::Float variant gets parsed as float32
            Value::Float(_) => {
                let mut numbers: Vec<f32> = Vec::new();
                for value in feature_values {
                    if let Value::Float(f) = value {
                        numbers.push(f.clone() as f32)
                    }
                }
                float_features.push(numbers);
            }
        }
    }

    // ensure that the shape is correct as the catboost expects a 2D vec even if we do not have
    // those features
    if cat_features.is_empty() {
        cat_features.push(vec![]);
    }

    if float_features.is_empty() {
        float_features.push(vec![]);
    }

    (float_features, cat_features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::predictor::FeatureName;
    use std::collections::HashMap;

    // fn create_random_2d_float_vec(rows: usize, cols: usize) -> Vec<Vec<f32>> {
    //     let mut rng = rand::thread_rng();
    //
    //     (0..rows)
    //         .map(|_| (0..cols).map(|_| rng.gen::<f64>() as f32).collect())
    //         .collect()
    // }
    //
    // fn create_random_2d_int_vec(rows: usize, cols: usize) -> Vec<Vec<i32>> {
    //     let mut rng = rand::thread_rng();
    //
    //     (0..rows)
    //         .map(|_| (0..cols).map(|_| rng.gen::<i64>() as i32).collect())
    //         .collect()
    // }
    //
    // fn create_random_2d_string_vec(rows: usize, cols: usize) -> Vec<Vec<String>> {
    //     let mut rng = rand::thread_rng();
    //     let values = vec!["a", "b"];
    //
    //     (0..rows)
    //         .map(|_| {
    //             (0..cols)
    //                 .map(|_| values.get(rng.gen_range(0..2)).unwrap().to_string())
    //                 .collect()
    //         })
    //         .collect()
    // }

    fn create_model_inputs(num_numeric_features: usize, num_string_features: usize, size: usize) -> ModelInput {
        let mut model_input: HashMap<FeatureName, Vec<Value>> = HashMap::new();
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
                    .get(rng.gen_range(0..size))
                    .unwrap()
                    .to_string();
                string_features.push(Value::String(value));
            }

            let feature_name = format!("string_feature_{}", i);
            model_input.insert(feature_name, string_features);
        }

        // create numeric features
        for i in 0..num_numeric_features {
            let mut number_features: Vec<Value> = Vec::new();

            for _ in 0..size {
                let value = rng.gen::<f64>();
                number_features.push(Value::Float(value));
            }

            let feature_name = format!("numeric_feature_{}", i);
            model_input.insert(feature_name, number_features);
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
            model.model.get_float_features_count(), // 4
            model.model.get_cat_features_count(), // 0
            1,
        );

        println!("{:?}", model_inputs);

        // inspect model
        let _ = model.predict(model_inputs);
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

        // inspect model
        let _ = model.predict(model_inputs);
    }

    #[test]
    fn successfully_load_catboost_binary_classification_model() {
        let path = "tests/model_artefacts/catboost_binary";
        let model = Catboost::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_catboost_binary_model() {
        let path = "tests/model_artefacts/catboost_binary";
        let model = Catboost::load(path).unwrap();

        fn create_random_2d_float_vec(rows: usize, cols: usize) -> Vec<Vec<f32>> {
            let mut rng = rand::thread_rng();

            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen::<f64>() as f32).collect())
                .collect()
        }

        fn create_random_2d_string_vec(rows: usize, cols: usize) -> Vec<Vec<String>> {
            let mut rng = rand::thread_rng();
            let values = vec!["aa", "bb"];

            (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| values.get(rng.gen_range(0..2)).unwrap().to_string())
                        .collect()
                })
                .collect()
        }

        let cat_features = create_random_2d_string_vec(1, model.model.get_cat_features_count());
        let float_features = create_random_2d_float_vec(1, model.model.get_float_features_count());

        println!(
            "prediction dimension: {}",
            model.model.get_dimensions_count()
        );
        println!(
            "numeric feature count: {}",
            model.model.get_float_features_count()
        );
        println!(
            "categoric feature count: {}",
            model.model.get_cat_features_count()
        );

        println!("{:?}", cat_features);
        println!("{:?}", float_features);

        // inspect model
        let pred = model
            .model
            .calc_model_prediction(float_features.clone(), cat_features.clone())
            .unwrap();
        let prob = model
            .model
            .calc_predict_proba(float_features, cat_features)
            .unwrap();

        fn sigmoid(x: f64) -> f64 {
            1. / (1. + (-x).exp())
        }

        println!("{:?}", pred);
        println!("{:?}", prob);
        println!("{:?}", sigmoid(pred[0]));
    }

    #[test]
    fn successfully_load_catboost_multiclass_classification_model() {
        let path = "tests/model_artefacts/catboost_multiclass";
        let model = Catboost::load(path);

        // assert the result is ok
        assert!(model.is_ok())
    }

    #[test]
    fn successfully_make_prediction_using_catboost_multiclass_classification_model() {
        let path = "tests/model_artefacts/catboost_multiclass";
        let model = Catboost::load(path).unwrap();

        fn create_random_2d_float_vec(rows: usize, cols: usize) -> Vec<Vec<f32>> {
            let mut rng = rand::thread_rng();

            (0..rows)
                .map(|_| (0..cols).map(|_| rng.gen::<f64>() as f32).collect())
                .collect()
        }

        fn create_random_2d_string_vec(rows: usize, cols: usize) -> Vec<Vec<String>> {
            let mut rng = rand::thread_rng();
            let values = vec!["summer", "winter"];

            (0..rows)
                .map(|_| {
                    (0..cols)
                        .map(|_| values.get(rng.gen_range(0..2)).unwrap().to_string())
                        .collect()
                })
                .collect()
        }

        let cat_features = create_random_2d_string_vec(1, model.model.get_cat_features_count());
        let float_features = create_random_2d_float_vec(1, model.model.get_float_features_count());

        println!(
            "prediction dimension: {}",
            model.model.get_dimensions_count()
        );
        println!(
            "numeric feature count: {}",
            model.model.get_float_features_count()
        );
        println!(
            "categoric feature count: {}",
            model.model.get_cat_features_count()
        );

        println!("{:?}", cat_features);
        println!("{:?}", float_features);

        // inspect model
        // let pred = model.model.calc_model_prediction(float_features.clone(), cat_features.clone()).unwrap();
        let prob = model
            .model
            .calc_predict_proba(float_features, cat_features)
            .unwrap();

        // fn sigmoid(x: f64) -> f64 {
        //     1. / (1. + (-x).exp())
        // }
        //
        // println!("{:?}", pred);
        // println!("{:?}", prob);
        // println!("{:?}", sigmoid(pred[0]));
    }
}
