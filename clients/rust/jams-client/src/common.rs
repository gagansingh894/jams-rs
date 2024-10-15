use serde::Deserialize;
use std::collections::HashMap;

#[derive(Deserialize, Debug)]
pub struct GetModelsResponse {
    /// Total number of models.
    pub total: i32,
    /// List of model names.
    pub models: Vec<Metadata>,
}
#[derive(Deserialize, Debug)]
pub struct Metadata {
    pub name: String,
    pub framework: String,
    pub path: String,
    pub last_updated: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Predictions(HashMap<String, Vec<Vec<f64>>>);

impl Predictions {
    #[allow(clippy::should_implement_trait)]
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let predictions: Predictions = serde_json::from_slice(bytes)?;
        Ok(predictions)
    }

    #[allow(clippy::wrong_self_convention)]
    pub fn to_vec(self) -> Vec<Vec<f64>> {
        self.0
            .values()
            .flat_map(|vecs| vecs.iter().cloned())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successfully_parses_bytes_into_predictions() {
        // Act
        let data= "{\"predictions\":[[0.7560540820707359],[1.152310804888906],[0.45694264204906754],[0.912618828350997],[0.08037521123549339],[0.8689713450910137],[0.4549892870109407],[0.5386298352854039],[0.471754086353748],[0.18414340024741896]]}".to_string();

        // Arrange
        let result = Predictions::from_bytes(data.as_bytes());

        // Assert
        assert!(result.is_ok())
    }

    #[test]
    fn successfully_converts_predictions_into_2d_vector() {
        // Act
        let data= "{\"predictions\":[[0.7560540820707359],[1.152310804888906],[0.45694264204906754],[0.912618828350997],[0.08037521123549339],[0.8689713450910137],[0.4549892870109407],[0.5386298352854039],[0.471754086353748],[0.18414340024741896]]}".to_string();

        // Arrange
        let result = Predictions::from_bytes(data.as_bytes());

        // Assert
        assert!(result.is_ok());
        let vec = result.unwrap().to_vec();
        assert_eq!(vec.len(), 10);
    }

    #[test]
    fn fails_to_parse_bytes_into_predictions() {
        // Act
        let data = "unsupported string value".to_string();

        // Arrange
        let result = Predictions::from_bytes(data.as_bytes());

        // Assert
        assert!(result.is_err())
    }
}
