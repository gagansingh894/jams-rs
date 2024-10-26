use crate::common::{get_url, GetModelsResponse, Predictions};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time;

#[derive(Serialize)]
struct PredictRequest {
    model_name: String,
    input: String,
}

#[derive(Deserialize)]
struct PredictResponse {
    output: String,
}

#[derive(Serialize)]
struct AddModelRequest {
    model_name: String,
}

#[derive(Serialize)]
struct UpdateModelRequest {
    model_name: String,
}

#[async_trait]
pub trait Client {
    async fn health_check(&self) -> anyhow::Result<()>;
    async fn predict(&self, model_name: String, model_input: String)
        -> anyhow::Result<Predictions>;
    async fn add_model(&self, model_name: String) -> anyhow::Result<()>;
    async fn update_model(&self, model_name: String) -> anyhow::Result<()>;
    async fn delete_model(&self, model_name: String) -> anyhow::Result<()>;
    async fn get_models(&self) -> anyhow::Result<GetModelsResponse>;
}

pub struct ApiClient {
    client: reqwest::Client,
    base_url: String,
    timeout: time::Duration,
}

impl ApiClient {
    pub fn builder() -> ApiClientBuilder {
        ApiClientBuilder::default()
    }
}

#[derive(Default)]
pub struct ApiClientBuilder {
    base_url: String,
    timeout: time::Duration,
}

impl ApiClientBuilder {
    pub fn new(base_url: String) -> ApiClientBuilder {
        ApiClientBuilder {
            base_url: get_url(base_url),
            timeout: time::Duration::from_secs(5),
        }
    }

    pub fn with_timeout(mut self, timeout: u64) -> ApiClientBuilder {
        self.timeout = time::Duration::from_secs(timeout);
        self
    }

    pub fn build(self) -> anyhow::Result<ApiClient> {
        let client = match reqwest::Client::builder().build() {
            Ok(client) => client,
            Err(err) => {
                anyhow::bail!("failed to create reqwest client: {}", err)
            }
        };
        Ok(ApiClient {
            client,
            base_url: self.base_url,
            timeout: self.timeout,
        })
    }
}

#[async_trait]
impl Client for ApiClient {
    async fn health_check(&self) -> anyhow::Result<()> {
        let url = format!("{}/{}", self.base_url, "healthcheck");
        match self.client.get(url).timeout(self.timeout).send().await {
            Ok(resp) => match resp.status().is_success() {
                true => Ok(()),
                false => {
                    anyhow::bail!(
                        "failed to health check J.A.M.S server ❌: {}",
                        resp.text().await.unwrap()
                    )
                }
            },
            Err(err) => {
                anyhow::bail!("failed to health check J.A.M.S server ❌: {}", err)
            }
        }
    }

    async fn predict(
        &self,
        model_name: String,
        model_input: String,
    ) -> anyhow::Result<Predictions> {
        let url = format!("{}/{}", self.base_url, "api/predict");
        match self
            .client
            .post(url)
            .json(&PredictRequest {
                model_name,
                input: model_input,
            })
            .timeout(self.timeout)
            .send()
            .await
        {
            Ok(resp) => match resp.status().is_success() {
                true => {
                    let predictions = resp.json::<PredictResponse>().await?;
                    match Predictions::from_bytes(predictions.output.as_ref()) {
                        Ok(predictions) => Ok(predictions),
                        Err(err) => {
                            anyhow::bail!(
                                "failed to parse response from bytes ❌: {}",
                                err.to_string()
                            )
                        }
                    }
                }
                false => {
                    anyhow::bail!(
                        "failed to get predictions ❌: {}",
                        resp.text().await.unwrap()
                    )
                }
            },
            Err(err) => {
                anyhow::bail!("failed to make predict request ❌: {}", err.to_string())
            }
        }
    }

    async fn add_model(&self, model_name: String) -> anyhow::Result<()> {
        let url = format!("{}/{}", self.base_url, "api/models");
        match self
            .client
            .post(url)
            .json(&AddModelRequest { model_name })
            .timeout(self.timeout)
            .send()
            .await
        {
            Ok(resp) => match resp.status().is_success() {
                true => Ok(()),
                false => {
                    anyhow::bail!("failed to add model ❌: {}", resp.text().await.unwrap())
                }
            },
            Err(err) => {
                anyhow::bail!("failed to make add_model request ❌: {}", err.to_string())
            }
        }
    }

    async fn update_model(&self, model_name: String) -> anyhow::Result<()> {
        let url = format!("{}/{}", self.base_url, "api/models");
        match self
            .client
            .put(url)
            .json(&UpdateModelRequest { model_name })
            .timeout(self.timeout)
            .send()
            .await
        {
            Ok(resp) => match resp.status().is_success() {
                true => Ok(()),
                false => {
                    anyhow::bail!("failed to update model ❌: {}", resp.text().await.unwrap())
                }
            },
            Err(err) => {
                anyhow::bail!(
                    "failed to make update_model request ❌: {}",
                    err.to_string()
                )
            }
        }
    }

    async fn delete_model(&self, model_name: String) -> anyhow::Result<()> {
        let url = format!(
            "{}/{}?model_name={}",
            self.base_url, "api/models", model_name
        );
        match self.client.delete(url).timeout(self.timeout).send().await {
            Ok(resp) => match resp.status().is_success() {
                true => Ok(()),
                false => {
                    anyhow::bail!("failed to delete model ❌: {}", resp.text().await.unwrap())
                }
            },
            Err(err) => {
                anyhow::bail!(
                    "failed to make delete_model request ❌: {}",
                    err.to_string()
                )
            }
        }
    }

    async fn get_models(&self) -> anyhow::Result<GetModelsResponse> {
        let url = format!("{}/{}", self.base_url, "api/models");
        match self.client.get(url).timeout(self.timeout).send().await {
            Ok(resp) => match resp.status().is_success() {
                true => Ok(resp.json::<GetModelsResponse>().await?),
                false => {
                    anyhow::bail!("failed to get models ❌: {}", resp.text().await.unwrap())
                }
            },
            Err(err) => {
                anyhow::bail!("failed to make get_models request ❌: {}", err.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    fn get_url() -> String {
        let hostname = env::var("JAMS_HTTP_HOSTNAME").unwrap_or("0.0.0.0".to_string());
        format!("{}:3000", hostname)
    }

    #[tokio::test]
    async fn successfully_sends_health_check_request() {
        // Arrange
        let client = ApiClientBuilder::new(get_url())
            .with_timeout(2)
            .build()
            .unwrap();

        // Act
        let resp = client.health_check().await;

        // Assert
        assert!(resp.is_ok())
    }

    #[tokio::test]
    async fn successfully_sends_get_model_request() {
        // Arrange
        let client = ApiClientBuilder::new(get_url())
            .with_timeout(2)
            .build()
            .unwrap();

        // Act
        let result = client.get_models().await;

        // Assert
        assert!(result.is_ok());
        let res = result.unwrap();
        // We will have some models as we are loading from assets/model_store
        assert!(!res.models.is_empty());
    }

    #[tokio::test]
    async fn successfully_sends_delete_model_request() {
        // Arrange
        let client = ApiClientBuilder::new(get_url())
            .with_timeout(2)
            .build()
            .unwrap();

        // Act
        client
            .add_model("pytorch-my_awesome_californiahousing_model".to_string())
            .await
            .unwrap();

        let resp = client
            .delete_model("my_awesome_californiahousing_model".to_string())
            .await;

        // Assert
        assert!(resp.is_ok())
    }

    #[tokio::test]
    async fn successfully_sends_add_model_request() {
        // Arrange
        let client = ApiClientBuilder::new(get_url())
            .with_timeout(2)
            .build()
            .unwrap();

        // Act
        client
            .delete_model("my_awesome_penguin_model".to_string())
            .await
            .unwrap();

        let resp = client
            .add_model("tensorflow-my_awesome_penguin_model".to_string())
            .await;

        // Assert
        assert!(resp.is_ok())
    }

    #[tokio::test]
    async fn successfully_sends_update_model_request() {
        // Arrange
        let client = ApiClientBuilder::new(get_url())
            .with_timeout(2)
            .build()
            .unwrap();

        // Act
        let resp = client.update_model("titanic_model".to_string()).await;

        // Assert
        assert!(resp.is_ok())
    }

    #[tokio::test]
    async fn successfully_sends_predict_model_request() {
        // Arrange
        let client = ApiClientBuilder::new(get_url())
            .with_timeout(2)
            .build()
            .unwrap();

        // Act
        let model_name = "titanic_model".to_string();
        let model_input = serde_json::json!(
                {
                    "pclass": ["1", "3"],
                    "sex": ["male", "female"],
                    "age": [22.0, 23.79929292929293],
                    "sibsp": ["0", "1", ],
                    "parch": ["0", "0"],
                    "fare": [151.55, 14.4542],
                    "embarked": ["S", "C"],
                    "class": ["First", "Third"],
                    "who": ["man", "woman"],
                    "adult_male": ["True", "False"],
                    "deck": ["Unknown", "Unknown"],
                    "embark_town": ["Southampton", "Cherbourg"],
                    "alone": ["True", "False"]
                }
        )
        .to_string();
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        let resp = client.predict(model_name, model_input).await;

        // Assert
        assert!(resp.is_ok())
    }
}
