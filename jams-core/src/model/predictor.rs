use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt::Formatter;

pub const DEFAULT_OUTPUT_KEY: &str = "predictions";

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
    /// We use a hashmap because we can have models with multiple outputs
    /// The client is responsible for selecting the correct field for their respective purpose
    /// For the models which do not support multiple outputs, the default key will be 'predictions'
    pub predictions: HashMap<String, Vec<Vec<f64>>>,
}

/// Struct representing the input to a model.
///
/// # Fields
/// * `0` - A hashmap where the keys are feature names and the values are feature values.
#[derive(Debug, Clone)]
pub struct ModelInput(HashMap<FeatureName, Values>);

impl<'de> Deserialize<'de> for ModelInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ModelInputVisitor;

        impl<'de> Visitor<'de> for ModelInputVisitor {
            type Value = ModelInput;

            fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
                formatter.write_str("a map with string keys and homogeneous arrays of integers, floats, or strings as values")
            }

            fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
            where
                M: MapAccess<'de>,
            {
                let mut data = HashMap::new();

                while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                    let values = match value {
                        serde_json::Value::Array(arr) => {
                            // Try to interpret the array as a vector of strings
                            if let Some(first_elem) = arr.first() {
                                // Check if the array is empty
                                if arr.is_empty() {
                                    return Err(serde::de::Error::custom(format!(
                                        "Empty array found for key '{}'",
                                        key
                                    )));
                                }
                                if first_elem.is_string() {
                                    let vec: Vec<String> = arr
                                        .into_iter()
                                        .map(|v| v.as_str().unwrap().to_string())
                                        .collect();
                                    Values::String(vec)
                                } else if first_elem.is_i64() {
                                    // Try to interpret the array as a vector of integers
                                    let vec: Vec<i32> = arr
                                        .into_iter()
                                        .map(|v| v.as_i64().unwrap() as i32)
                                        .collect();
                                    Values::Int(vec)
                                } else if first_elem.is_f64() {
                                    // Try to interpret the array as a vector of floats
                                    let vec: Vec<f32> = arr
                                        .into_iter()
                                        .map(|v| v.as_f64().unwrap() as f32)
                                        .collect();
                                    Values::Float(vec)
                                } else {
                                    return Err(serde::de::Error::custom(
                                        "Unsupported value type in array",
                                    ));
                                }
                            } else {
                                return Err(serde::de::Error::custom("Empty array found"));
                            }
                        }
                        _ => return Err(serde::de::Error::custom("Expected an array as value")),
                    };
                    data.insert(key, values);
                }
                Ok(ModelInput(data))
            }
        }

        deserializer.deserialize_map(ModelInputVisitor)
    }
}

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
        let model_input: Result<ModelInput, serde_json::Error> = serde_json::from_str(json);
        match model_input {
            Ok(input) => Ok(input),
            Err(e) => {
                tracing::error!("Failed to parse json to model input: {} ❌", e.to_string());
                anyhow::bail!("Failed to parse json to model input: {} ❌", e.to_string())
            }
        }
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

#[derive(Debug, Clone)]
pub enum Values {
    String(Vec<String>),
    Int(Vec<i32>),
    Float(Vec<f32>),
}

impl Values {
    /// Consumes the `Values` enum and returns the inner `Vec<String>` if the variant is `Values::String`.
    ///
    /// # Returns
    ///
    /// * `Some(Vec<String>)` - if the variant is `Values::String`.
    /// * `None` - if the variant is not `Values::String`.
    ///
    pub fn into_strings(self) -> Option<Vec<String>> {
        if let Values::String(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns a clone of the inner `Vec<String>` if the `Values` variant is `Values::String`.
    ///
    /// # Returns
    ///
    /// * `Some(Vec<String>)` - if the variant is `Values::String`.
    /// * `None` - if the variant is not `Values::String`.
    ///
    pub fn to_strings(&self) -> Option<Vec<String>> {
        if let Values::String(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }

    /// Consumes the `Values` enum and returns the inner `Vec<f32>` if the variant is `Values::Float`.
    ///
    /// # Returns
    ///
    /// * `Some(Vec<f32>)` - if the variant is `Values::Float`.
    /// * `None` - if the variant is not `Values::Float`.
    ///
    pub fn into_floats(self) -> Option<Vec<f32>> {
        if let Values::Float(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns a clone of the inner `Vec<f32>` if the `Values` variant is `Values::Float`.
    ///
    /// # Returns
    ///
    /// * `Some(Vec<f32>)` - if the variant is `Values::Float`.
    /// * `None` - if the variant is not `Values::Float`.
    ///
    pub fn to_floats(&self) -> Option<Vec<f32>> {
        if let Values::Float(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }

    /// Consumes the `Values` enum and returns the inner `Vec<i32>` if the variant is `Values::Int`.
    ///
    /// # Returns
    ///
    /// * `Some(Vec<i32>)` - if the variant is `Values::Int`.
    /// * `None` - if the variant is not `Values::Int`.
    ///
    pub fn into_ints(self) -> Option<Vec<i32>> {
        if let Values::Int(v) = self {
            Some(v)
        } else {
            None
        }
    }

    /// Returns a clone of the inner `Vec<i32>` if the `Values` variant is `Values::Int`.
    ///
    /// # Returns
    ///
    /// * `Some(Vec<i32>)` - if the variant is `Values::Int`.
    /// * `None` - if the variant is not `Values::Int`.
    ///
    pub fn to_ints(&self) -> Option<Vec<i32>> {
        if let Values::Int(v) = self {
            Some(v.to_vec())
        } else {
            None
        }
    }
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
        println!("{:?}", model_input);

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

        let model_input = ModelInput::from_str(json_data);

        // assert result is err
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

        let model_input = ModelInput::from_str(json_data);

        // assert result is err
        assert!(model_input.is_err())
    }
}
