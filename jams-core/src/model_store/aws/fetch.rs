use std::sync::Arc;

use crate::model_store::aws::common::download_objects;
use async_trait::async_trait;
use aws_sdk_s3 as s3;
use dashmap::DashMap;

use crate::model_store::fetcher::Fetcher;
use crate::model_store::storage::{load_models, Model, ModelName};

#[async_trait]
impl Fetcher for aws_sdk_s3::client::Client {
    /// Implements the `Fetcher` trait for an S3 client, allowing it to check if the
    /// model store is empty and fetch models from an S3 bucket.
    ///
    async fn is_empty(&self, artefacts_dir_name: Option<String>) -> anyhow::Result<bool> {
        let keys = match artefacts_dir_name {
            None => {
                tracing::error!("S3 bucket name not provided ❌.");
                anyhow::bail!("S3 bucket name not provided ❌.")
            }
            Some(s3_bucket_name) => get_keys(self, s3_bucket_name).await?,
        };

        Ok(keys.is_empty())
    }

    /// Fetches models from the specified S3 bucket, downloads them to `output_dir`, and loads them into memory.
    ///
    /// # Parameters
    /// - `artefacts_dir_name`: The S3 bucket name where model artefacts are stored.
    /// - `output_dir`: The directory where models will be downloaded and stored.
    ///
    /// # Returns
    /// - `Ok(DashMap<ModelName, Arc<Model>>)` containing all loaded models mapped by their names.
    /// - `Err`: Returns an error if any step in fetching, saving, or loading models fails.
    ///
    async fn fetch_models(
        &self,
        artefacts_dir_name: Option<String>,
        output_dir: String,
    ) -> anyhow::Result<DashMap<ModelName, Arc<Model>>> {
        let s3_bucket_name = match artefacts_dir_name {
            None => {
                tracing::error!("S3 bucket name not provided ❌.");
                anyhow::bail!("S3 bucket name not provided ❌.")
            }
            Some(name) => name,
        };

        let keys = get_keys(self, s3_bucket_name.clone()).await?;

        match download_objects(self, s3_bucket_name, keys, output_dir.as_str()).await {
            Ok(_) => {
                tracing::info!("Downloaded objects from s3 ✅")
            }
            Err(e) => {
                tracing::warn!("Failed to download objects from s3: {e}. ⚠️")
            }
        }

        let models = load_models(output_dir).await?;

        Ok(models)
    }
}

/// Retrieves object keys from an S3 bucket.
///
/// This function uses an `s3::Client` instance to list object keys from the specified S3 bucket.
///
/// # Arguments
///
/// * `client` - An `s3::Client` instance for interacting with AWS S3.
/// * `bucket_name` - The name of the S3 bucket from which to retrieve object keys.
///
/// # Returns
///
/// * `Result<Vec<String>>` - A vector containing object keys retrieved from the S3 bucket,
///   or an error if object keys cannot be retrieved.
///
/// # Errors
///
/// This function will return an error if:
/// * Object keys cannot be listed from the S3 bucket.
///
#[tracing::instrument(skip(client))]
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
                    tracing::warn!(
                        "No models found in the S3 bucket hence no models will be loaded ⚠️"
                    );
                }
                Some(objects) => {
                    for object in objects {
                        match object.key {
                            None => {
                                tracing::warn!("Object key is empty ⚠️");
                            }
                            Some(key) => {
                                keys.push(key);
                            }
                        }
                    }
                }
            },
            Err(e) => {
                tracing::error!(
                    "Failed to list objects in the {} bucket: {}",
                    bucket_name,
                    e.into_service_error()
                );
                anyhow::bail!("Failed to list objects in the {} bucket", bucket_name,)
            }
        }
    }
    Ok(keys)
}

/// Fetches models from an S3 bucket, downloads them to a local directory, and loads them into memory.
///
/// This function first retrieves object keys from the S3 bucket, downloads corresponding objects,
/// saves and unpacks them into the specified local directory, and finally loads the models into memory.
///
/// # Arguments
///
/// * `client` - An `s3::Client` instance for interacting with AWS S3.
/// * `bucket_name` - The name of the S3 bucket from which to fetch models.
/// * `model_store_dir` - The local directory where downloaded models will be stored and loaded from.
///
/// # Returns
///
/// * `Result<DashMap<ModelName, Arc<Model>>>` - A `DashMap` containing loaded models mapped by their names,
///   or an error if fetching or loading models fails.
///
/// # Errors
///
/// This function will return an error if:
/// * The object keys cannot be retrieved from the S3 bucket.
/// * Objects cannot be downloaded from S3.
/// * Downloaded tarballs cannot be unpacked.
/// * Models cannot be loaded from the local directory.
///
#[tracing::instrument(skip(client))]
async fn fetch_models(
    client: &s3::Client,
    bucket_name: String,
    model_store_dir: String,
) -> anyhow::Result<DashMap<ModelName, Arc<Model>>> {
    let keys = get_keys(client, bucket_name.clone()).await?;

    match download_objects(client, bucket_name, keys, model_store_dir.as_str()).await {
        Ok(_) => {
            tracing::info!("Downloaded objects from s3 ✅")
        }
        Err(_) => {
            tracing::warn!("Failed to download objects from s3 ⚠️")
        }
    }

    let models = load_models(model_store_dir).await?;

    Ok(models)
}
