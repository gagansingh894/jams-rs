#[cfg(test)]
pub mod utils {
    use crate::model::input::ModelInput;
    use rand::Rng;

    #[cfg(test)]
    pub fn create_model_inputs(
        num_numeric_features: usize,
        num_string_features: usize,
        size: usize,
    ) -> ModelInput {
        let mut model_input = ModelInput::default();
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
            let mut string_values = Vec::new();
            for _ in 0..size {
                let value = random_string_values
                    .get(rng.gen_range(0..random_string_values.len()))
                    .unwrap()
                    .to_string();
                string_values.push(value);
            }
            model_input.string_features.values.extend(string_values);
            model_input.string_features.shape.1 = size;

            let feature_name = format!("string_feature_{}", i);
            model_input.string_features.names.push(feature_name);
            model_input.string_features.shape.0 += 1;
        }

        // create numeric features
        for i in 0..num_numeric_features {
            let mut float_values = Vec::new();
            for _ in 0..size {
                let value = rng.gen::<f32>();
                float_values.push(value);
            }
            model_input.float_features.values.extend(float_values);
            model_input.float_features.shape.1 = size;

            let feature_name = format!("numeric_feature_{}", i);
            model_input.float_features.names.push(feature_name);
            model_input.float_features.shape.0 += 1;
        }

        model_input
    }

    #[cfg(test)]
    pub fn create_model_inputs_with_names(
        numeric_features_names: Vec<String>,
        string_features_names: Vec<String>,
        size: usize,
    ) -> ModelInput {
        let mut model_input = ModelInput::default();
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
            let mut string_values = Vec::new();
            for _ in 0..size {
                let value = random_string_values
                    .get(rng.gen_range(0..random_string_values.len()))
                    .unwrap()
                    .to_string();
                string_values.push(value);
            }
            model_input.string_features.values.extend(string_values);
            model_input.string_features.shape.1 = size;

            model_input.string_features.names.push(feature_name);
            model_input.string_features.shape.0 += 1;
        }

        // create numeric features
        for feature_name in numeric_features_names {
            let mut float_values = Vec::new();
            for _ in 0..size {
                let value = rng.gen::<f32>();
                float_values.push(value);
            }
            model_input.float_features.values.extend(float_values);
            model_input.float_features.shape.1 = size;

            model_input.float_features.names.push(feature_name);
            model_input.float_features.shape.0 += 1;
        }

        model_input
    }
}
