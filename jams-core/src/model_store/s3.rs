use async_trait::async_trait;
use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3 as s3;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use flate2::read::GzDecoder;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use tar::Archive;
use tokio::io::AsyncWriteExt;

use crate::model_store::storage::{load_models, Metadata, Model, ModelName, Storage};

/// A struct representing a model store that interfaces with S3.
#[allow(dead_code)]
pub struct S3ModelStore {
    /// A thread-safe map of model names to their corresponding models.
    pub models: DashMap<ModelName, Arc<Model>>,
    /// An S3 client for interacting with the S3 service.
    client: s3::Client,
    /// The name S3 bucket where models are stored.
    bucket_name: String,
    /// Directory, which stores the model artifacts downloaded from S3
    model_store_dir: String,
}

const DOWNLOADED_MODELS_DIRECTORY_NAME: &str = "model_store";

impl S3ModelStore {
    /// Creates a new `S3ModelStore`.
    ///
    /// # Arguments
    ///
    /// * `bucket_name` - Name of the S3 bucket where models are stored.
    ///
    /// # Returns
    ///
    /// A result containing the newly created `S3ModelStore` or an error if the initialization fails.
    pub async fn new(bucket_name: String) -> anyhow::Result<Self> {
        // Ensure model_dir_uri is not empty, return error if empty
        if bucket_name.is_empty() {
            anyhow::bail!("S3 bucket name must be specified ❌")
        }
        // Use the default region provider chain to load the region from the configuration file
        let region_provider = RegionProviderChain::default_provider();
        // Load the AWS configuration from the environment
        let config = aws_config::from_env().region(region_provider).load().await;
        // Create an S3 client with the loaded configuration
        let client = s3::Client::new(&config);
        // Directory which stores the models downloaded from S3
        let model_store_dir = format!(
            "{}/{}",
            std::env::var("HOME").unwrap(),
            DOWNLOADED_MODELS_DIRECTORY_NAME
        );
        // Fetch the models from S3
        let models = match fetch_models(&client, bucket_name.clone(), model_store_dir.clone()).await
        {
            Ok(models) => {
                log::info!("Successfully fetched valid models from S3 ✅");
                models
            }
            Err(e) => {
                anyhow::bail!("Failed to fetch models ❌ - {}", e.to_string());
            }
        };

        Ok(Self {
            models,
            client,
            bucket_name,
            model_store_dir,
        })
    }
}

#[async_trait]
impl Storage for S3ModelStore {
    async fn add_model(&self, _model_name: ModelName, _model_path: &str) -> anyhow::Result<()> {
        todo!()
    }

    async fn update_model(&self, _model_name: ModelName) -> anyhow::Result<()> {
        todo!()
    }

    /// Retrieves a model from the model store.
    ///
    /// This function returns a reference to the model with the specified name if it exists.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be retrieved.
    ///
    /// # Returns
    ///
    /// This function returns an `Option`:
    /// * `Some(Ref<ModelName, Arc<Model>>)` if the model exists.
    /// * `None` if the model does not exist.
    ///
    fn get_model(&self, model_name: ModelName) -> Option<Ref<ModelName, Arc<Model>>> {
        self.models.get(model_name.as_str())
    }

    /// Retrieves metadata for all models in the model store.
    ///
    /// This function returns a vector of `Metadata` containing information about all the models in the store.
    ///
    /// # Returns
    ///
    /// This function returns an `anyhow::Result` containing a vector of `Metadata`.
    ///
    fn get_models(&self) -> anyhow::Result<Vec<Metadata>> {
        let model: Vec<Metadata> = self
            .models
            .iter()
            .map(|f| f.value().info.to_owned())
            .collect();
        Ok(model)
    }

    /// Deletes a model from the model store.
    ///
    /// This function removes the model with the specified name from the store.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be deleted.
    ///
    /// # Errors
    ///
    /// This function returns an error if the specified model does not exist in the store.
    fn delete_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        match self.models.remove(&model_name) {
            None => {
                anyhow::bail!(
                    "Failed to delete model as the specified model {} does not exist",
                    model_name
                )
            }
            Some(_) => Ok(()),
        }
    }
}

async fn fetch_models(
    client: &s3::Client,
    bucket_name: String,
    model_store_dir: String,
) -> anyhow::Result<DashMap<ModelName, Arc<Model>>> {
    let keys = get_keys(client, bucket_name.clone()).await;

    match download_objects(client, bucket_name, keys.unwrap(), model_store_dir.as_str()).await {
        Ok(_) => {
            log::info!("Downloaded objects from s3 ✅")
        }
        Err(_) => {
            log::warn!("Failed to download objects from s3 ⚠️")
        }
    }

    let models = match load_models(model_store_dir) {
        Ok(models) => {
            log::info!("Successfully loaded models from directory ✅");
            models
        }
        Err(e) => {
            anyhow::bail!("Failed to load models - {}", e.to_string());
        }
    };

    Ok(models)
}

async fn get_keys(client: &s3::Client, bucket_name: String) -> anyhow::Result<Vec<String>> {
    let mut keys: Vec<String> = Vec::new();

    let mut response = client
        .list_objects_v2()
        .bucket(bucket_name.clone())
        .max_keys(10)
        .into_paginator()
        .send();

    while let Some(result) = response.next().await {
        match result {
            Ok(output) => match output.contents {
                None => {
                    log::warn!("No models found in the S3 bucket hence no models will be loaded ⚠️");
                }
                Some(objects) => {
                    for object in objects {
                        match object.key {
                            None => {
                                log::warn!("Object key is empty ⚠️");
                            }
                            Some(key) => {
                                keys.push(key);
                            }
                        }
                    }
                }
            },
            Err(e) => {
                anyhow::bail!(
                    "Failed to list objects in the {} bucket: {}",
                    bucket_name,
                    e.into_service_error()
                )
            }
        }
    }
    Ok(keys)
}

async fn download_objects(
    client: &s3::Client,
    bucket_name: String,
    object_keys: Vec<String>,
    out_dir: &str,
) -> anyhow::Result<()> {
    // Create a tempdir to hold downloaded models
    // This will be deleted when the program exits
    let dir = match tempfile::Builder::new().prefix("models").tempdir() {
        Ok(dir) => dir,
        Err(_) => {
            anyhow::bail!("Failed to create temporary directory ❌");
        }
    };
    let temp_path = dir.path().to_str().unwrap();

    for object_key in object_keys {
        let response = client
            .get_object()
            .bucket(bucket_name.clone())
            .key(object_key.clone())
            .send()
            .await;

        match response {
            Ok(output) => {
                match output.body.collect().await {
                    Ok(data) => {
                        match save_and_upack_tarball(
                            temp_path,
                            object_key.clone(),
                            data.into_bytes(),
                            out_dir,
                        )
                        .await
                        {
                            Ok(_) => {
                                // Do nothing
                            }
                            Err(_) => {
                                log::warn!("Failed to save artefact {} ⚠️", object_key)
                            }
                        }
                    }
                    Err(_) => {
                        log::warn!("Failed to download artefact {} ⚠️", object_key)
                    }
                }
            }
            Err(_) => {
                log::warn!("Failed to get object key: {} from S3 ⚠️", object_key)
            }
        }
    }
    Ok(())
}

async fn save_and_upack_tarball(
    path: &str,
    key: String,
    data: bytes::Bytes,
    out_dir: &str,
) -> anyhow::Result<()> {
    let file_path = Path::new(path).join(key);

    // Create parent directories if they do not exist
    if let Some(parent) = file_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut file = tokio::fs::File::create(&file_path).await?;
    file.write_all(&data).await?;
    log::info!("Saved file to: {:?}", file_path);

    unpack_tarball(file_path.to_str().unwrap(), out_dir).unwrap();
    Ok(())
}

fn unpack_tarball(tarball_path: &str, out_dir: &str) -> anyhow::Result<()> {
    let tar_gz = File::open(tarball_path)?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);

    match archive.unpack(out_dir) {
        Ok(_) => {
            log::info!(
                "Unpacked tarball: {:?} at location: {}",
                tarball_path,
                DOWNLOADED_MODELS_DIRECTORY_NAME
            );
            Ok(())
        }
        Err(_) => {
            log::warn!(
                "Failed to unpack tarball ⚠️: {:?} at location: {}",
                tarball_path,
                DOWNLOADED_MODELS_DIRECTORY_NAME
            );
            Ok(())
        }
    }
}

// todo: use Mocks
#[cfg(test)]
mod tests {
    // #[tokio::test]
    // async fn successfully_load_models_from_s3_model_store() {
    //     // Arrange
    //     let bucket_name = "jams-model-store".to_string();
    //
    //     // Act
    //     let model_store = S3ModelStore::new(bucket_name).await;
    //
    //     // Assert
    //     assert!(model_store.is_ok());
    //
    //     // Cleanup
    //     std::fs::remove_dir_all(model_store.unwrap().model_store_dir).unwrap();
    // }
}
