use serde::{Deserialize, Serialize};
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
#[derive(Debug, Serialize)]
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
    #[tracing::instrument(skip(json))]
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
    #[tracing::instrument(skip(value))]
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
    #[tracing::instrument(skip(strings))]
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
    #[tracing::instrument(skip(ints))]
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
    #[tracing::instrument(skip(floats))]
    pub fn from_floats(floats: Vec<f32>) -> Self {
        let values = floats.into_iter().map(Value::Float).collect();
        Values(values)
    }

    /// Converts the feature values to a vector of strings.
    ///
    /// # Returns
    /// * `anyhow::Result<Vec<String>>` - The converted values.
    #[tracing::instrument(skip(self))]
    pub fn to_strings(&self) -> anyhow::Result<Vec<String>> {
        self.iter()
            .map(|v| {
                Ok(match v.as_string() {
                    None => {
                        anyhow::bail!("Failed to convert Values to strings ❌")
                    }
                    Some(v) => v.to_string(),
                })
            })
            .collect()
    }

    /// Converts the feature values to a vector of integers.
    ///
    /// # Returns
    /// * `anyhow::Result<Vec<i32>>` - The converted values.
    #[tracing::instrument(skip(self))]
    pub fn to_ints(&self) -> anyhow::Result<Vec<i32>> {
        self.iter()
            .map(|v| {
                Ok(match v.as_int() {
                    None => {
                        anyhow::bail!("Failed to convert Values to ints ❌")
                    }
                    Some(v) => v,
                })
            })
            .collect()
    }

    /// Converts the feature values to a vector of floats.
    ///
    /// # Returns
    /// * `anyhow::Result<Vec<f32>>` - The converted values.
    #[tracing::instrument(skip(self))]
    pub fn to_floats(&self) -> anyhow::Result<Vec<f32>> {
        self.iter()
            .map(|v| {
                Ok(match v.as_float() {
                    None => {
                        anyhow::bail!("Failed to convert Values to floats ❌")
                    }
                    Some(v) => v,
                })
            })
            .collect()
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
#[tracing::instrument(skip(json))]
fn parse_json_serde_value(json: serde_json::Value) -> anyhow::Result<HashMap<FeatureName, Values>> {
    // create an empty hashmap to store the features
    let mut model_input: HashMap<FeatureName, Values> = HashMap::new();

    let json_object = match json.as_object() {
        None => {
            anyhow::bail!("Failed to convert JSON input to JSON map ❌")
        }
        Some(json_object) => json_object,
    };

    for (key, values) in json_object {
        let feature_name: FeatureName = key.to_string();

        // validate value is an array
        let vec = match values.as_array() {
            None => {
                anyhow::bail!("Failed to cast serde json value to array.");
            }
            Some(vec) => vec,
        };

        let first = match vec.first() {
            None => {
                anyhow::bail!("Failed to get the first element from the array");
            }
            Some(first) => first,
        };

        if first.is_i64() {
            let feature_values = vec
                .iter()
                .map(|v| Value::Int(v.as_i64().unwrap() as i32))
                .collect();
            model_input.insert(feature_name, Values(feature_values));
        } else if first.is_f64() {
            let feature_values = vec
                .iter()
                .map(|v| Value::Float(v.as_f64().unwrap() as f32))
                .collect();
            model_input.insert(feature_name, Values(feature_values));
        } else if first.is_string() {
            let feature_values = vec
                .iter()
                .map(|v| Value::String(v.as_str().unwrap().to_owned()))
                .collect();
            model_input.insert(feature_name, Values(feature_values));
        } else {
            anyhow::bail!("Unsupported format in input json");
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

    #[test]
    fn fails_to_parses_model_input_serde_json_value_due_to_empty_input() {
        let json_data = r#"{
        "feature_1": [],
        "feature_2": [],
        "feature_3": []
    }"#;

        let serde_json_value = serde_json::from_str(json_data).unwrap();

        let model_input = ModelInput::from_serde_json(serde_json_value);

        // assert result is ok
        assert!(model_input.is_err())
    }

    #[test]
    fn fails_to_parses_model_input_serde_json_value_due_to_value_being_not_an_array() {
        let json_data = r#"{
        "feature_1": {
                "a": ""
            },
        "feature_2": [],
        "feature_3": []
    }"#;

        let serde_json_value = serde_json::from_str(json_data).unwrap();

        let model_input = ModelInput::from_serde_json(serde_json_value);

        // assert result is ok
        assert!(model_input.is_err())
    }

    // Adding these tests as these are not currently used anywhere in the code
    // but are useful methods to have
    #[test]
    fn successfully_convert_to_value_enum_to_option_i32() {
        let value = Value::Int(2147);

        let convert = value.as_int();

        assert!(convert.is_some())
    }

    #[test]
    fn successfully_convert_to_values_enum_from_vec_i32() {
        let vec = vec![1, 2, 3];

        let values = Values::from_ints(vec.clone());

        assert_eq!(vec.len(), values.iter().len())
    }

    #[test]
    fn successfully_convert_to_values_enum_from_vec_f32() {
        let vec: Vec<f32> = vec![1.0, 2.0, 3.0];

        let values = Values::from_floats(vec.clone());

        assert_eq!(vec.len(), values.iter().len())
    }

    #[test]
    fn successfully_convert_to_values_enum_from_vec_string() {
        let vec: Vec<String> = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let values = Values::from_strings(vec.clone());

        assert_eq!(vec.len(), values.iter().len())
    }

    #[test]
    fn successfully_convert_the_values_enum_of_int_to_vec_int() {
        let vec: Vec<i32> = vec![1, 2, 3];

        let values = Values::from_ints(vec.clone());

        let values_vec = values.to_ints().unwrap();

        assert_eq!(vec, values_vec)
    }

    #[test]
    fn successfully_returns_a_mutable_iterator_for_values_enum() {
        let vec: Vec<f32> = vec![1.0, 2.0, 3.0];

        let mut values = Values::from_floats(vec.clone());

        // change all values to 10
        for element in values.iter_mut() {
            *element = Value::Int(10);
        }

        // assertion
        for element in values.iter() {
            assert_eq!(element.as_int().unwrap(), 10)
        }
    }
}
