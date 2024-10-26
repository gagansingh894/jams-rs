use crate::model_store::azure::common::download_blob;
use crate::model_store::fetcher::Fetcher;
use crate::model_store::storage::{load_models, Model, ModelName};
use async_trait::async_trait;
use azure_storage_blobs::prelude::ContainerClient;
use dashmap::DashMap;
use futures::StreamExt;
use std::num::NonZeroU32;
use std::sync::Arc;

#[async_trait]
impl Fetcher for ContainerClient {
    /// Implements the `Fetcher` trait for an Azure client, allowing it to check if the
    /// model store is empty and fetch models from blob storage.
    ///
    async fn is_empty(&self, artefacts_dir_name: Option<String>) -> anyhow::Result<bool> {
        if artefacts_dir_name.is_some() {
            anyhow::bail!("Unexpected parameter 'artefacts_dir_name' provided ❌")
        }
        let max_results = NonZeroU32::new(1).unwrap();
        let mut stream = self.list_blobs().max_results(max_results).into_stream();

        match stream.next().await {
            None => {
                anyhow::bail!("Failed to iterate stream ❌.")
            }
            Some(resp) => match resp {
                Ok(result) => Ok(result.blobs.items.is_empty()),
                Err(e) => {
                    anyhow::bail!("Failed to collect data to bytes: {}", e)
                }
            },
        }
    }

    /// Asynchronously fetches models from an Azure Blob Storage container, unpacks them, and loads them into a `DashMap`.
    ///
    /// # Parameters
    /// - `artefacts_dir_name`: The storage container name  where model artefacts are stored.
    /// - `output_dir`: The directory where models will be downloaded and stored.
    ///
    /// # Returns
    /// - `Ok(DashMap<ModelName, Arc<Model>>)` containing all loaded models mapped by their names.
    /// - `Err`: Returns an error if any step in fetching, saving, or loading models fails.
    /// # Errors
    ///
    /// This function will return an error if:
    /// * A temporary directory cannot be created.
    /// * The temporary directory path cannot be converted to a string.
    /// * Listing blobs in the container fails.
    /// * Downloading a blob fails.
    /// * Converting downloaded bytes to a file fails.
    /// * Saving and unpacking the tarball fails.
    /// * Loading the models from the directory fails.
    ///
    async fn fetch_models(
        &self,
        artefacts_dir_name: Option<String>,
        output_dir: String,
    ) -> anyhow::Result<DashMap<ModelName, Arc<Model>>> {
        if artefacts_dir_name.is_some() {
            anyhow::bail!("Unexpected parameter 'artefacts_dir_name' provided ❌")
        }

        let max_results = NonZeroU32::new(10).unwrap();

        // List the blobs in the container
        let mut stream = self.list_blobs().max_results(max_results).into_stream();
        // For each blob, create a blob client and download the blob
        while let Some(result) = stream.next().await {
            match result {
                Ok(result) => {
                    for blob in result.blobs.blobs() {
                        // Download blob to model_store_dir
                        match download_blob(self, blob.clone().name, output_dir.clone()).await {
                            Ok(blob) => blob,
                            Err(e) => {
                                anyhow::bail!("Failed to download blob ❌: {}", e)
                            }
                        };
                    }
                }
                Err(e) => {
                    anyhow::bail!("Failed to collect data to bytes: {}", e)
                }
            }
        }

        let models = load_models(output_dir).await?;

        Ok(models)
    }
}
