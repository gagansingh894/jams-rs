use crate::common::{GetModelsResponse, Metadata, Predictions};
use async_trait::async_trait;
use jams_proto::jams_v1::model_server_client::ModelServerClient;
use jams_proto::jams_v1::{
    AddModelRequest, DeleteModelRequest, PredictRequest, UpdateModelRequest,
};
use tonic::transport::Channel;

#[async_trait]
pub trait Client {
    async fn health_check(&mut self) -> anyhow::Result<()>;
    async fn predict(
        &mut self,
        model_name: String,
        model_input: String,
    ) -> anyhow::Result<Predictions>;
    async fn add_model(&mut self, model_name: String) -> anyhow::Result<()>;
    async fn update_model(&mut self, model_name: String) -> anyhow::Result<()>;
    async fn delete_model(&mut self, model_name: String) -> anyhow::Result<()>;
    async fn get_models(&mut self) -> anyhow::Result<GetModelsResponse>;
}

pub struct ApiClient {
    client: ModelServerClient<Channel>,
    base_url: String,
}

impl ApiClient {
    async fn new(base_url: String) -> anyhow::Result<Self> {
        let base_url = format!("http://{}", base_url);
        let client = match ModelServerClient::connect(base_url.clone()).await {
            Ok(client) => client,
            Err(err) => {
                anyhow::bail!("failed to create grpc client ❌: {}", err.to_string())
            }
        };
        Ok(ApiClient { client, base_url })
    }
}

#[async_trait]
impl Client for ApiClient {
    async fn health_check(&mut self) -> anyhow::Result<()> {
        match self.client.health_check(tonic::Request::new(())).await {
            Ok(_) => Ok(()),
            Err(status) => {
                anyhow::bail!(
                    "failed to health check J.A.M.S server ❌: {}",
                    status.to_string()
                )
            }
        }
    }

    async fn predict(
        &mut self,
        model_name: String,
        model_input: String,
    ) -> anyhow::Result<Predictions> {
        match self
            .client
            .predict(PredictRequest {
                model_name,
                input: model_input,
            })
            .await
        {
            Ok(resp) => {
                let inner = resp.into_inner().output;
                match Predictions::from_bytes(inner.as_bytes()) {
                    Ok(predictions) => Ok(predictions),
                    Err(err) => {
                        anyhow::bail!(
                            "failed to parse response from bytes ❌: {}",
                            err.to_string()
                        )
                    }
                }
            }
            Err(status) => {
                anyhow::bail!("failed to get predictions ❌: {}", status.to_string())
            }
        }
    }

    async fn add_model(&mut self, model_name: String) -> anyhow::Result<()> {
        match self.client.add_model(AddModelRequest { model_name }).await {
            Ok(_) => Ok(()),
            Err(status) => {
                anyhow::bail!("failed to add model ❌: {}", status.to_string())
            }
        }
    }

    async fn update_model(&mut self, model_name: String) -> anyhow::Result<()> {
        match self
            .client
            .update_model(UpdateModelRequest { model_name })
            .await
        {
            Ok(_) => Ok(()),
            Err(status) => {
                anyhow::bail!("failed to update model ❌: {}", status.to_string())
            }
        }
    }

    async fn delete_model(&mut self, model_name: String) -> anyhow::Result<()> {
        match self
            .client
            .delete_model(DeleteModelRequest { model_name })
            .await
        {
            Ok(_) => Ok(()),
            Err(status) => {
                anyhow::bail!("failed to delete model ❌: {}", status.to_string())
            }
        }
    }

    async fn get_models(&mut self) -> anyhow::Result<GetModelsResponse> {
        match self.client.get_models(tonic::Request::new(())).await {
            Ok(response) => {
                let pb_response = response.into_inner();

                let models = pb_response
                    .models
                    .into_iter()
                    .map(|model| Metadata {
                        name: model.name,
                        framework: model.framework,
                        path: model.path,
                        last_updated: model.last_updated,
                    })
                    .collect();

                Ok(GetModelsResponse {
                    total: pb_response.total,
                    models,
                })
            }
            Err(status) => {
                anyhow::bail!("failed to get models ❌: {}", status.to_string())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    fn get_url() -> String {
        let hostname = env::var("JAMS_GRPC_HOSTNAME").unwrap_or("0.0.0.0".to_string());
        format!("{}:4000", hostname)
    }

    #[tokio::test]
    async fn successfully_sends_health_check_request() {
        // Arrange
        let mut client = ApiClient::new(get_url()).await.unwrap();

        // Act
        let resp = client.health_check().await;

        // Assert
        assert!(resp.is_ok())
    }

    #[tokio::test]
    async fn successfully_sends_get_model_request() {
        // Arrange
        let mut client = ApiClient::new(get_url()).await.unwrap();

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
        let mut client = ApiClient::new(get_url()).await.unwrap();

        // Act
        let resp = client
            .delete_model("my_awesome_californiahousing_model".to_string())
            .await;

        // Assert
        assert!(resp.is_ok())
    }

    #[tokio::test]
    async fn successfully_sends_add_model_request() {
        // Arrange
        let mut client = ApiClient::new(get_url()).await.unwrap();

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
        let mut client = ApiClient::new(get_url()).await.unwrap();

        // Act
        let resp = client.update_model("titanic_model".to_string()).await;

        // Assert
        assert!(resp.is_ok())
    }

    #[tokio::test]
    async fn successfully_sends_predict_model_request() {
        // Arrange
        let mut client = ApiClient::new(get_url()).await.unwrap();

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
