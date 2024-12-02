use crate::pool::MODEL_INPUT_POOL;
use crate::FEATURE_NAMES_CAPACITY;
use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::fmt::Formatter;
use std::sync::Arc;

/// Type alias for the feature name, which is a string.
pub type FeatureName = String;

#[derive(Debug, Clone)]
pub enum Values {
    String(Vec<String>),
    Int(Vec<i32>),
    Float(Vec<f32>),
}

impl Values {
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

    /// Converts the `Values` enum to a reference of the vector of `i32` values.
    ///
    /// This method checks if the `Values` enum is of the `Int` variant. If it is, it
    /// returns a reference to the vector of `i32` values inside it. If the enum is not
    /// of the `Int` variant, it returns `None`.
    ///
    /// # Returns
    /// - `Some(&Vec<i32>)`: A reference to the vector of `i32` values, if the enum is of type `Int`.
    /// - `None`: If the enum is not of type `Int`.
    pub fn as_ints(&self) -> Option<&Vec<i32>> {
        if let Values::Int(v) = self {
            Some(v) // Return a reference to the vector inside `Values::Int`
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

    /// Converts the `Values` enum to a reference of the vector of `f32` values.
    ///
    /// This method checks if the `Values` enum is of the `Float` variant. If it is, it
    /// returns a reference to the vector of `f32` values inside it. If the enum is not
    /// of the `Float` variant, it returns `None`.
    ///
    /// # Returns
    /// - `Some(&Vec<f32>)`: A reference to the vector of `f32` values, if the enum is of type `Float`.
    /// - `None`: If the enum is not of type `Float`.
    pub fn as_floats(&self) -> Option<&Vec<f32>> {
        if let Values::Float(v) = self {
            Some(v) // Return a reference to the vector inside `Values::Float`
        } else {
            None
        }
    }

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

    /// Converts the `Values` enum to a reference of the vector of `String` values.
    ///
    /// This method checks if the `Values` enum is of the `String` variant. If it is, it
    /// returns a reference to the vector of `String` values inside it. If the enum is not
    /// of the `String` variant, it returns `None`.
    ///
    /// # Returns
    /// - `Some(&Vec<f32>)`: A reference to the vector of `f32` values, if the enum is of type `Float`.
    /// - `None`: If the enum is not of type `Float`.
    pub fn as_strings(&self) -> Option<&Vec<String>> {
        if let Values::String(v) = self {
            Some(v) // Return a reference to the vector inside `Values::Float`
        } else {
            None
        }
    }

    /// Appends the contents of another `Values` enum to the current one.
    ///
    /// This method allows you to append a vector of the same type (e.g., `String` to `String`, `i32` to `i32`, etc.)
    ///
    /// # Panics
    /// Panics if the variant of the current `Values` enum doesn't match the variant of the other `Values` enum.
    ///
    pub fn append(&mut self, other: &mut Values) {
        match (self, other) {
            (Values::String(v1), Values::String(v2)) => v1.append(v2), // No cloning, move `v2` into `v1`
            (Values::Int(v1), Values::Int(v2)) => v1.append(v2), // No cloning, move `v2` into `v1`
            (Values::Float(v1), Values::Float(v2)) => v1.append(v2), // No cloning, move `v2` into `v1`
            _ => panic!("Cannot append values of different types"), // Panics if variants are not the same
        }
    }

    /// Clears the data inside the enum.
    ///
    /// This method empties the vector contained within the current variant of the `Values` enum.
    /// It does not change the variant itself. For example:
    /// - For `Values::String`, it clears the `Vec<String>`.
    /// - For `Values::Int`, it clears the `Vec<i32>`.
    /// - For `Values::Float`, it clears the `Vec<f32>`.
    ///
    fn clear(&mut self) {
        match self {
            Values::String(v) => v.clear(),
            Values::Int(v) => v.clear(),
            Values::Float(v) => v.clear(),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Values::String(v) => v.is_empty(),
            Values::Int(v) => v.is_empty(),
            Values::Float(v) => v.is_empty(),
        }
    }
}

impl Extend<f32> for Values {
    fn extend<T: IntoIterator<Item = f32>>(&mut self, iter: T) {
        match self {
            Values::Float(v) => v.extend(iter),
            _ => panic!("Cannot extend non-Float variant with a Float iterator"),
        }
    }
}

impl Extend<i32> for Values {
    fn extend<T: IntoIterator<Item = i32>>(&mut self, iter: T) {
        match self {
            Values::Int(v) => v.extend(iter),
            _ => panic!("Cannot extend non-Integer variant with a Integer iterator"),
        }
    }
}

impl Extend<String> for Values {
    fn extend<T: IntoIterator<Item = String>>(&mut self, iter: T) {
        match self {
            Values::String(v) => v.extend(iter),
            _ => panic!("Cannot extend non-String variant with a String iterator"),
        }
    }
}

/// A collection of features of a specific type.
///
/// This struct represents a generic feature set, where each feature has a name,
/// a collection of values, and a shape that describes the dimensions of the data.
#[derive(Debug, Clone)]
pub struct Features {
    /// The names of the features.
    pub names: Vec<FeatureName>,
    /// The values associated with the features.
    pub values: Values,
    /// The shape of the feature set, represented as (rows, columns).
    pub shape: (usize, usize),
}

impl Features {
    /// Clears the contents of the `Features` struct.
    ///
    /// This method resets the `Features` struct by:
    /// - Setting `shape` to `(0, 0)` to indicate an empty feature set.
    /// - Calling `clear` on the `names` vector to remove all feature names.
    /// - Calling `clear` on the `values` enum to remove all feature values.
    ///
    fn clear(&mut self) {
        self.shape = (0, 0);
        self.values.clear();
        self.names.clear()
    }
}

/// The input data for a machine learning model.
///
/// This struct contains three types of features: float, integer, and string.
/// Each type is stored in its own `Features` structure.
#[derive(Debug, Clone)]
pub struct ModelInput {
    /// The float features in the input.
    pub float_features: Features,
    /// The integer features in the input.
    pub integer_features: Features,
    /// The string features in the input.
    pub string_features: Features,
}

impl Default for ModelInput {
    fn default() -> Self {
        let float_features = Features {
            names: Vec::with_capacity(FEATURE_NAMES_CAPACITY),
            values: Values::Float(Vec::with_capacity(FEATURE_NAMES_CAPACITY)),
            shape: (0, 0),
        };
        let integer_features = Features {
            names: Vec::with_capacity(FEATURE_NAMES_CAPACITY),
            values: Values::Int(Vec::with_capacity(FEATURE_NAMES_CAPACITY)),
            shape: (0, 0),
        };
        let string_features = Features {
            names: Vec::with_capacity(FEATURE_NAMES_CAPACITY),
            values: Values::String(Vec::with_capacity(FEATURE_NAMES_CAPACITY)),
            shape: (0, 0),
        };

        Self {
            float_features,
            integer_features,
            string_features,
        }
    }
}

impl ModelInput {
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

    /// Clears the contents of the `ModelInput` struct.
    fn clear(&mut self) {
        self.integer_features.clear();
        self.float_features.clear();
        self.string_features.clear();
    }
}

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
                // get the ModelInput from the pool and detach from it
                // A background job ensures that the object pool is always filled
                let pool = Arc::clone(&MODEL_INPUT_POOL);
                let pool_object = pool.pull_owned(ModelInput::default);
                let (_, mut model_input) = pool_object.detach();
                model_input.clear();

                while let Some((key, value)) = map.next_entry::<String, serde_json::Value>()? {
                    match value {
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
                                let arr_length = arr.len();
                                if first_elem.is_string() {
                                    let vec: Vec<String> = arr
                                        .into_iter()
                                        .map(|v| v.as_str().unwrap().to_owned())
                                        .collect();
                                    model_input.string_features.names.push(key);
                                    model_input.string_features.values.extend(vec);
                                    model_input.string_features.shape.0 += 1;
                                    model_input.string_features.shape.1 = arr_length;
                                } else if first_elem.is_i64() {
                                    // Try to interpret the array as a vector of integers
                                    let vec: Vec<i32> = arr
                                        .into_iter()
                                        .map(|v| v.as_i64().unwrap() as i32)
                                        .collect();
                                    model_input.integer_features.names.push(key);
                                    model_input.integer_features.values.extend(vec);
                                    model_input.integer_features.shape.0 += 1;
                                    model_input.integer_features.shape.1 = arr_length;
                                } else if first_elem.is_f64() {
                                    // Try to interpret the array as a vector of floats
                                    let vec: Vec<f32> = arr
                                        .into_iter()
                                        .map(|v| v.as_f64().unwrap() as f32)
                                        .collect();
                                    model_input.float_features.names.push(key);
                                    model_input.float_features.values.extend(vec);
                                    model_input.float_features.shape.0 += 1;
                                    model_input.float_features.shape.1 = arr_length;
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
                    }
                }
                Ok(model_input)
            }
        }

        deserializer.deserialize_map(ModelInputVisitor)
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

        // assert result is ok
        assert!(model_input.is_ok())
    }

    #[test]
    fn successfully_parses_model_input_v2_from_str() {
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
