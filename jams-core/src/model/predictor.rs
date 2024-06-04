use serde::Deserialize;
use std::collections::HashMap;

/// Trait for making predictions using a model.
pub trait Predictor: Send + Sync + 'static {
    /// Predicts the output for the given model input.
    ///
    /// # Arguments
    /// * `input` - The input data for the model.
    ///
    /// # Returns
    /// * `Ok(Output)` - The prediction output.
    /// * `Err(anyhow::Error)` - If there was an error during prediction.
    fn predict(&self, input: ModelInput) -> anyhow::Result<Output>;
}

/// Struct representing the output of a prediction.
#[derive(Debug)]
pub struct Output {
    /// The predictions made by the model.
    pub predictions: Vec<Vec<f64>>,
}

/// Struct representing the input to a model.
///
/// # Fields
/// * `0` - A hashmap where the keys are feature names and the values are feature values.
#[derive(Deserialize, Debug, Clone)]
pub struct ModelInput(HashMap<FeatureName, Values>);

impl ModelInput {
    /// Retrieves a reference to the values associated with the given feature name.
    ///
    /// # Arguments
    /// * `key` - The feature name.
    ///
    /// # Returns
    /// * `Some(&Values)` - If the feature name exists.
    /// * `None` - If the feature name does not exist.
    pub fn get(&self, key: &FeatureName) -> Option<&Values> {
        self.0.get(key)
    }

    /// Returns a vector of all values.
    pub fn values(self) -> Vec<Values> {
        self.inner().into_values().collect()
    }

    /// Parses a JSON string to create a `ModelInput` instance.
    ///
    /// # Arguments
    /// * `json` - The JSON string representing the model input.
    ///
    /// # Returns
    /// * `Ok(ModelInput)` - If parsing was successful.
    /// * `Err(anyhow::Error)` - If there was an error during parsing.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(json: &str) -> anyhow::Result<ModelInput> {
        let value: serde_json::Value = serde_json::from_str(json)?;
        let model_input = parse_json_serde_value(value)?;
        Ok(Self(model_input))
    }

    /// Creates a `ModelInput` instance from a `serde_json::Value`.
    ///
    /// # Arguments
    /// * `value` - The `serde_json::Value` representing the model input.
    ///
    /// # Returns
    /// * `Ok(ModelInput)` - If the conversion was successful.
    /// * `Err(anyhow::Error)` - If there was an error during the conversion.
    pub fn from_serde_json(value: serde_json::Value) -> anyhow::Result<Self> {
        let model_input = parse_json_serde_value(value)?;
        Ok(Self(model_input))
    }

    /// Creates a `ModelInput` instance from a `HashMap`.
    ///
    /// # Arguments
    /// * `value` - The `HashMap` representing the model input.
    ///
    /// # Returns
    /// * `Ok(ModelInput)` - If the conversion was successful.
    /// * `Err(anyhow::Error)` - If there was an error during the conversion.
    pub fn from_hashmap(value: HashMap<FeatureName, Values>) -> anyhow::Result<Self> {
        Ok(Self(value))
    }

    /// Consumes the `ModelInput` and returns the inner `HashMap`.
    pub fn inner(self) -> HashMap<FeatureName, Values> {
        self.0
    }

    /// Returns an iterator over the feature names and values.
    pub fn iter(&self) -> std::collections::hash_map::Iter<'_, FeatureName, Values> {
        self.0.iter()
    }

    /// Consumes the `ModelInput` and returns an iterator over the feature names and values.
    #[allow(clippy::should_implement_trait)]
    pub fn into_iter(self) -> std::collections::hash_map::IntoIter<FeatureName, Values> {
        self.0.into_iter()
    }
}

/// Type alias for the feature name, which is a string.
pub type FeatureName = String;

/// Struct representing a collection of feature values.
#[derive(Deserialize, Debug, Clone)]
pub struct Values(pub Vec<Value>);

impl Values {
    /// Returns an iterator over the feature values.
    pub fn iter(&self) -> std::slice::Iter<'_, Value> {
        self.0.iter()
    }

    /// Returns a mutable iterator over the feature values.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Value> {
        self.0.iter_mut()
    }

    /// Creates a `Values` instance from a vector of strings.
    ///
    /// # Arguments
    /// * `strings` - A vector of strings.
    ///
    /// # Returns
    /// * `Values` - The created `Values` instance.
    pub fn from_strings(strings: Vec<String>) -> Self {
        let values = strings.into_iter().map(Value::String).collect();
        Values(values)
    }

    /// Creates a `Values` instance from a vector of integers.
    ///
    /// # Arguments
    /// * `ints` - A vector of integers.
    ///
    /// # Returns
    /// * `Values` - The created `Values` instance.
    pub fn from_ints(ints: Vec<i32>) -> Self {
        let values = ints.into_iter().map(Value::Int).collect();
        Values(values)
    }

    /// Creates a `Values` instance from a vector of floats.
    ///
    /// # Arguments
    /// * `floats` - A vector of floats.
    ///
    /// # Returns
    /// * `Values` - The created `Values` instance.
    pub fn from_floats(floats: Vec<f32>) -> Self {
        let values = floats.into_iter().map(Value::Float).collect();
        Values(values)
    }

    /// Converts the feature values to a vector of strings.
    ///
    /// # Returns
    /// * `Vec<String>` - The converted values.
    pub fn to_strings(&self) -> Vec<String> {
        self.iter()
            .map(|v| v.as_string().unwrap().to_string())
            .collect()
    }

    /// Converts the feature values to a vector of integers.
    ///
    /// # Returns
    /// * `Vec<i32>` - The converted values.
    pub fn to_ints(&self) -> Vec<i32> {
        self.iter().map(|v| v.as_int().unwrap()).collect()
    }

    /// Converts the feature values to a vector of floats.
    ///
    /// # Returns
    /// * `Vec<f32>` - The converted values.
    pub fn to_floats(&self) -> Vec<f32> {
        self.iter().map(|v| v.as_float().unwrap()).collect()
    }
}
/// Enum representing a feature value, which can be a string, integer, or float.
#[derive(Deserialize, Debug, Clone)]
pub enum Value {
    String(String),
    Int(i32),
    Float(f32),
}

impl Value {
    /// Converts the `Value` to an `Option<&String>`.
    ///
    /// # Returns
    /// * `Some(&String)` - If the value is a string.
    /// * `None` - If the value is not a string.
    pub fn as_string(&self) -> Option<&String> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Converts the `Value` to an `Option<i32>`.
    ///
    /// # Returns
    /// * `Some(i32)` - If the value is an integer.
    /// * `None` - If the value is not an integer.
    pub fn as_int(&self) -> Option<i32> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Converts the `Value` to an `Option<f32>`.
    ///
    /// # Returns
    /// * `Some(f32)` - If the value is a float.
    /// * `None` - If the value is not a float.
    pub fn as_float(&self) -> Option<f32> {
        match self {
            Value::Float(f) => Some(*f),
            _ => None,
        }
    }
}

/// Parses a `serde_json::Value` into a `HashMap` of feature names and values.
///
/// # Arguments
/// * `json` - The JSON value representing the model input.
///
/// # Returns
/// * `Ok(HashMap<FeatureName, Values>)` - If parsing was successful.
/// * `Err(anyhow::Error)` - If there was an error during parsing.
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
