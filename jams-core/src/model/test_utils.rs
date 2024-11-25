#[cfg(test)]
pub mod utils {
    use crate::model::predict::{FeatureName, ModelInput, Values};
    use crate::MAX_CAPACITY;
    use rand::Rng;
    use std::collections::HashMap;

    #[cfg(test)]
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
            let mut string_features: Vec<String> = Vec::with_capacity(MAX_CAPACITY);

            for _ in 0..size {
                let value = random_string_values
                    .get(rng.gen_range(0..random_string_values.len()))
                    .unwrap()
                    .to_string();
                string_features.push(value);
            }

            let feature_name = format!("string_feature_{}", i);
            model_input.insert(feature_name, Values::String(string_features));
        }

        // create numeric features
        for i in 0..num_numeric_features {
            let mut number_features: Vec<f32> = Vec::with_capacity(MAX_CAPACITY);

            for _ in 0..size {
                let value = rng.gen::<f32>();
                number_features.push(value);
            }

            let feature_name = format!("numeric_feature_{}", i);
            model_input.insert(feature_name, Values::Float(number_features));
        }

        ModelInput::from_hashmap(model_input).unwrap()
    }

    #[cfg(test)]
    pub fn create_model_inputs_with_names(
        numeric_features_names: Vec<String>,
        string_features_names: Vec<String>,
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
        for feature_name in string_features_names {
            let mut string_features: Vec<String> = Vec::with_capacity(MAX_CAPACITY);

            for _ in 0..size {
                let value = random_string_values
                    .get(rng.gen_range(0..random_string_values.len()))
                    .unwrap()
                    .to_string();
                string_features.push(value);
            }

            model_input.insert(feature_name, Values::String(string_features));
        }

        // create numeric features
        for feature_name in numeric_features_names {
            let mut number_features: Vec<f32> = Vec::with_capacity(MAX_CAPACITY);

            for _ in 0..size {
                let value = rng.gen::<f32>();
                number_features.push(value);
            }

            model_input.insert(feature_name, Values::Float(number_features));
        }

        ModelInput::from_hashmap(model_input).unwrap()
    }
}
