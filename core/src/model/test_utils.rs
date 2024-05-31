use crate::model::predictor::{FeatureName, ModelInput, Value, Values};
use rand::Rng;
use std::collections::HashMap;

pub fn create_model_inputs(
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
