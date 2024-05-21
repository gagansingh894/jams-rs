use serde::Deserialize;
use std::collections::HashMap;

pub trait Predictor {
    fn predict(&self, input: ModelInput);
}

// Feature Name is an alias to string which refers to the feature name which is being fed into the model
pub(crate) type FeatureName = String;

// Value defines valid types which the model can accept as input
#[derive(Deserialize, Debug, Clone)]
pub enum Value {
    String(String),
    Int(i64),
    Float(f64),
}

// ModelInput is the core type which Predictor accepts.
#[derive(Deserialize, Debug, Clone)]
pub struct ModelInput(HashMap<FeatureName, Vec<Value>>);

impl ModelInput {
    pub fn from_str(json: &str) -> anyhow::Result<ModelInput> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        let model_input = parse_json_serde_value(value)?;
        Ok(Self(model_input))
    }

    pub fn from_serde_json(value: serde_json::Value) -> anyhow::Result<Self> {
        let model_input = parse_json_serde_value(value)?;
        Ok(Self(model_input))
    }

    pub fn from_hashmap(value: HashMap<FeatureName, Vec<Value>>) -> anyhow::Result<Self> {
        Ok(Self(value))
    }

    pub fn inner(self) -> HashMap<FeatureName, Vec<Value>> {
        self.0
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, FeatureName, Vec<Value>> {
        self.0.iter()
    }

    pub fn into_iter(self) -> std::collections::hash_map::IntoIter<FeatureName, Vec<Value>> {
        self.0.into_iter()
    }
}

fn parse_json_serde_value(
    json: serde_json::Value,
) -> anyhow::Result<HashMap<FeatureName, Vec<Value>>> {
    // create an empty hashmap to store the features
    let mut model_input: HashMap<FeatureName, Vec<Value>> = HashMap::new();

    for (key, values) in json.as_object().unwrap().into_iter() {
        let feature_name: FeatureName = key.to_string();

        // validate value is an array
        let vec = values
            .as_array()
            .expect("failed to cast serde json value to array");
        let first = vec
            .first()
            .expect("failed to get the first element from the array");

        if first.is_i64() {
            // todo: can fail, simplifying with unwrap
            let feature_values = vec
                .iter()
                .map(|v| Value::Int(v.as_i64().unwrap()))
                .collect();
            model_input.insert(feature_name, feature_values);
        } else if first.is_f64() {
            // todo: can fail, simplifying with unwrap
            let feature_values = vec
                .iter()
                .map(|v| Value::Float(v.as_f64().unwrap()))
                .collect();
            model_input.insert(feature_name, feature_values);
        } else if first.is_string() {
            // todo: can fail, simplifying with unwrap
            let feature_values = vec
                .iter()
                .map(|v| Value::String(v.as_str().unwrap().to_owned()))
                .collect();
            model_input.insert(feature_name, feature_values);
        } else {
            // todo: error out
            println!("unexpected format");
        }
    }

    Ok(model_input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_parses_model_input_from_str() {
        let json_data = r#"{
        "feature_1": [42, 42],
        "feature_2": [3.14, 3.14],
        "feature_3": ["a", "a"]
    }"#;

        let model_input = ModelInput::from_str(json_data);

        // assert result is ok
        assert!(model_input.is_ok())
    }

    #[test]
    fn successfully_parses_model_input_serde_json_value() {
        let json_data = r#"{
        "feature_1": [42, 42],
        "feature_2": [3.14, 3.14],
        "feature_3": ["a", "a"]
    }"#;

        let serde_json_value = serde_json::from_str(json_data).unwrap();

        let model_input = ModelInput::from_serde_json(serde_json_value);

        // assert result is ok
        assert!(model_input.is_ok())
    }
}
