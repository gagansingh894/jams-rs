use crate::common::{GetModelsResponse, Predictions};
use async_trait::async_trait;
use serde::Serialize;

#[derive(Serialize)]
struct PredictRequest {
    model_name: String,
    input: String,
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
}

impl ApiClient {
    pub fn new(base_url: String) -> anyhow::Result<Self> {
        let client = match reqwest::Client::builder().build() {
            Ok(client) => client,
            Err(err) => {
                anyhow::bail!("failed to create reqwest client: {}", err)
            }
        };
        let base_url = format!("http://{}", base_url);

        Ok(ApiClient { client, base_url })
    }
}

#[async_trait]
impl Client for ApiClient {
    async fn health_check(&self) -> anyhow::Result<()> {
        let url = format!("{}/{}", self.base_url, "healthcheck");
        match self.client.get(url).send().await {
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
            .send()
            .await
        {
            Ok(resp) => match resp.status().is_success() {
                true => {
                    let bytes = resp.bytes().await?;
                    match Predictions::from_bytes(bytes.as_ref()) {
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
        match self.client.delete(url).send().await {
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
        match self.client.get(url).send().await {
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
