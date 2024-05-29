use serde::Deserialize;
use std::collections::HashMap;

pub trait Predictor {
    // todo: work out how to output multiclass output
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output>;
}

pub struct Output {
    pub predictions: Vec<f64>,
}

// ModelInput is the core type which Predictor accepts.
#[derive(Deserialize, Debug, Clone)]
pub struct ModelInput(HashMap<FeatureName, Values>);

impl ModelInput {
    pub fn get_values(self) -> Vec<Values> {
        self.inner().into_values().collect()
    }

    pub fn from_str(json: &str) -> anyhow::Result<ModelInput> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        let model_input = parse_json_serde_value(value)?;
        Ok(Self(model_input))
    }

    pub fn from_serde_json(value: serde_json::Value) -> anyhow::Result<Self> {
        let model_input = parse_json_serde_value(value)?;
        Ok(Self(model_input))
    }

    pub fn from_hashmap(value: HashMap<FeatureName, Values>) -> anyhow::Result<Self> {
        Ok(Self(value))
    }

    pub fn inner(self) -> HashMap<FeatureName, Values> {
        self.0
    }

    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, FeatureName, Values> {
        self.0.iter()
    }

    pub fn into_iter(self) -> std::collections::hash_map::IntoIter<FeatureName, Values> {
        self.0.into_iter()
    }
}

// Feature Name is an alias to string which refers to the feature name which is being fed into the model
pub type FeatureName = String;

#[derive(Deserialize, Debug, Clone)]
pub struct Values(pub Vec<Value>);

impl Values {
    pub fn iter(&self) -> std::slice::Iter<'_, Value> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Value> {
        self.0.iter_mut()
    }

    pub fn from_strings(strings: Vec<String>) -> Self {
        let values = strings.into_iter().map(|s| Value::String(s)).collect();
        Values(values)
    }

    pub fn from_ints(ints: Vec<i32>) -> Self {
        let values = ints.into_iter().map(|i| Value::Int(i)).collect();
        Values(values)
    }

    pub fn from_floats(floats: Vec<f32>) -> Self {
        let values = floats.into_iter().map(|f| Value::Float(f)).collect();
        Values(values)
    }

    pub fn to_strings(&self) -> Vec<String> {
        self.iter()
            .map(|v| v.as_string().unwrap().to_string())
            .collect()
    }

    pub fn to_ints(&self) -> Vec<i32> {
        self.iter().map(|v| v.as_int().unwrap()).collect()
    }

    pub fn to_floats(&self) -> Vec<f32> {
        self.iter().map(|v| v.as_float().unwrap()).collect()
    }
}

// Value defines valid types which the model can accept as input
#[derive(Deserialize, Debug, Clone)]
pub enum Value {
    String(String),
    Int(i32),
    Float(f32),
}

impl Value {
    // Method to convert the Value to a String
    pub fn as_string(&self) -> Option<&String> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    // Method to convert the Value to an i32
    pub fn as_int(&self) -> Option<i32> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    // Method to convert the Value to a f64
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }
}

fn parse_json_serde_value(json: serde_json::Value) -> anyhow::Result<HashMap<FeatureName, Values>> {
    // create an empty hashmap to store the features
    let mut model_input: HashMap<FeatureName, Values> = HashMap::new();

    for (key, values) in json.as_object().unwrap() {
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
                .map(|v| Value::Int(v.as_i64().unwrap() as i32))
                .collect();
            model_input.insert(feature_name, Values(feature_values));
        } else if first.is_f64() {
            // todo: can fail, simplifying with unwrap
            let feature_values = vec
                .iter()
                .map(|v| Value::Float(v.as_f64().unwrap() as f32))
                .collect();
            model_input.insert(feature_name, Values(feature_values));
        } else if first.is_string() {
            // todo: can fail, simplifying with unwrap
            let feature_values = vec
                .iter()
                .map(|v| Value::String(v.as_str().unwrap().to_owned()))
                .collect();
            model_input.insert(feature_name, Values(feature_values));
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
