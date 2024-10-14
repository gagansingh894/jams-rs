use serde::Deserialize;
use std::slice::{Iter, IterMut};

#[derive(Deserialize)]
pub struct GetModelsResponse {
    /// Total number of models.
    pub total: i32,
    /// List of model names.
    pub models: Vec<Metadata>,
}
#[derive(Deserialize)]
pub struct Metadata {
    pub name: String,
    pub framework: String,
    pub path: String,
    pub last_updated: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Predictions(Vec<Vec<f64>>);

impl Predictions {
    #[allow(clippy::should_implement_trait)]
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let predictions: Predictions = serde_json::from_slice(bytes)?;
        Ok(predictions)
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_vec(self) -> Vec<Vec<f64>> {
        self.0
    }

    /// Returns an iterator over the feature values.
    pub fn iter(&self) -> Iter<'_, Vec<f64>> {
        self.0.iter()
    }

    /// Returns a mutable iterator over the feature values.
    pub fn iter_mut(&mut self) -> IterMut<'_, Vec<f64>> {
        self.0.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn successfully_parses_bytes_into_predictions() {}

    #[test]
    fn successfully_converts_predictions_into_2d_vector() {}

    #[test]
    fn successfully_returns_a_mutable_iterator_for_predictions() {}

    #[test]
    fn successfully_returns_an_iterator_for_predictions() {}

    #[test]
    fn fails_to_parse_bytes_into_predictions() {}
}
