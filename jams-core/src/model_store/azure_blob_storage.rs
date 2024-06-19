use std::env::VarError;
use std::fs::{File, remove_dir_all};
use std::num::NonZeroU32;
use std::sync::Arc;
use async_trait::async_trait;
use azure_core::Error;
use azure_storage::StorageCredentials;
use azure_storage_blobs::container::operations::ListBlobsResponse;
use azure_storage_blobs::prelude::{BlobClient, BlobServiceClient, ContainerClient};
use dashmap::DashMap;
use dashmap::mapref::one::Ref;
use futures::StreamExt;
use std::io::Write;
use azure_storage_blobs::blob::operations::GetBlobResponse;
use uuid::Uuid;
use crate::model_store::common::{cleanup, unpack_tarball};
use crate::model_store::s3::S3ModelStore;
use crate::model_store::storage::{load_models, Metadata, Model, ModelName, Storage};

const DOWNLOADED_MODELS_DIRECTORY_NAME_PREFIX: &str = "model_store";

/// A struct representing a model store that interfaces with azure blob storage.
pub struct AzureBlobStorageModelStore {
    /// A thread-safe map of model names to their corresponding models.
    pub models: Arc<DashMap<ModelName, Arc<Model>>>,
    /// Azure container client for interacting with storage container and listing blobs
    /// A new blob client will be created for each blob to download it to the local file system
    container_client: ContainerClient,
    /// Directory, which stores the model artifacts downloaded from Azure blob
    model_store_dir: String,
}

impl AzureBlobStorageModelStore {
   pub async fn new(storage_container_name: String) -> anyhow::Result<Self> {
       // Ensure model_dir_uri is not empty, return error if empty
       if storage_container_name.is_empty() {
           anyhow::bail!("Azure storage container name must be specified ❌")
       }
       // Create Azure Blob Service client
       let container_client = match build_azure_storage_client() {
           Ok(blob_service_client) => {
               blob_service_client.container_client(storage_container_name)
           }
           Err(e) => {
               anyhow::bail!("Failed to create Azure Blob Service client: {}", e)
           }
       };
       // Specify temporary directory for storing models downloaded from azure blob
       let model_store_dir = format!(
           "{}/{}_{}",
           std::env::var("HOME").unwrap_or("/usr/local".to_string()),
           DOWNLOADED_MODELS_DIRECTORY_NAME_PREFIX,
           Uuid::new_v4(),
       );

       // Fetch the models from Azure Blob Storage
       let models = match fetch_models(&container_client, model_store_dir.clone()).await
       {
           Ok(models) => {
               log::info!("Successfully fetched valid models from Azure Blob Storage ✅");
               models
           }
           Err(e) => {
               anyhow::bail!("Failed to fetch models ❌ - {}", e.to_string());
           }
       };


       Ok(Self {
           models: Arc::new(models),
           container_client,
           model_store_dir,
       })
   }
}

/// Implements the `Drop` trait for `AzureBlobStorageModelStore`.
///
/// This implementation ensures that the temporary directory used for the model store is cleaned up
/// when the `AzureBlobStorageModelStore` instance is dropped. This helps to avoid leaving temporary files on disk
/// and ensures proper resource cleanup.
///
/// # Fields
///
/// * `model_store_dir` - The directory path for the temporary local model store which contains models downloaded from azure blob.
///
impl Drop for AzureBlobStorageModelStore {
    fn drop(&mut self) {
        cleanup(self.model_store_dir.clone())
    }
}

fn build_azure_storage_client() -> anyhow::Result<BlobServiceClient> {
    let account = match std::env::var("STORAGE_ACCOUNT") {
        Ok(account) => { account }
        Err(_) => {
            anyhow::bail!("Azure STORAGE_ACCOUNT env variable not set ❌")
        }
    };
    let access_key = match std::env::var("STORAGE_ACCESS_KEY") {
        Ok(ak) => { ak }
        Err(_) => {
            anyhow::bail!("Azure STORAGE_ACCESS_KEY env variable not set ❌")
        }

    };
    let storage_credentials = StorageCredentials::access_key(account.clone(), access_key);

    Ok(BlobServiceClient::new(account, storage_credentials))
}


async fn fetch_models(client: &ContainerClient, model_store_dir: String) -> anyhow::Result<DashMap<ModelName, Arc<Model>>> {
    let temp_path = match tempfile::Builder::new().prefix("models").tempdir() {
        Ok(dir) => {
            match dir.path().to_str() {
                None => {
                    anyhow::bail!("Failed to convert TempDir to String ❌")
                }
                Some(p) => { p.to_string() }
            }
        }
        Err(e) => {
            anyhow::bail!("Failed to create temp directory: {}", e)
        }
    };

    let max_results = NonZeroU32::new(10).unwrap();

    // List the blobs in the container
    let mut stream = client
        .list_blobs()
        .max_results(max_results)
        .into_stream();
    // For each blob, create a blob client and download the blob
    while let Some(result) = stream.next().await {
        match result {
            Ok(result) => {
                for blob in result.blobs.blobs() {
                    let blob_name = blob.clone().name;
                    let blob_client = client.blob_client(blob_name.clone());
                    let mut blob_stream = blob_client.get().chunk_size(0x2000u64).into_stream();

                    // Download blob
                    let mut complete_response: Vec<u8> = vec![];
                    while let Some(value) = blob_stream.next().await  {
                        let data = match value {
                            Ok(response) => {
                                match response.data.collect().await {
                                    Ok(data) => {
                                        data.to_vec()
                                    }
                                    Err(e) => {
                                        anyhow::bail!("Failed to convert bytes: {}", e)
                                    }
                                }
                            }
                            Err(e) => {
                                anyhow::bail!("Failed to collect data to bytes: {}", e)
                            }
                        };
                        complete_response.extend(&data);
                    }

                    // Convert bytes to File
                    let temp_save_path = format!("{}/{}", temp_path, blob_name.clone());
                    println!("{:?}", temp_save_path.clone());
                    match File::create(temp_save_path.clone()) {
                        Ok(mut file) => {
                            match file.write_all(&complete_response[..]) {
                                Ok(_) => {
                                    log::info!("File {} downloaded successfully at {} ✅", blob_name.clone(), temp_save_path.clone())
                                }
                                Err(e) => {
                                    anyhow::bail!("Failed to write to file: {}", e)
                                }
                            }
                        }
                        Err(e) => {
                            anyhow::bail!("Failed to create file: {}", e)
                        }
                    }
                    // Untar the file
                    unpack_tarball(temp_save_path.clone().as_str(), model_store_dir.as_str()).unwrap();
                }
            }
            Err(e) => {
                anyhow::bail!("Failed to collect data to bytes: {}", e)
            }
        }
    }

    let models = match load_models(model_store_dir).await {
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

#[async_trait]
impl Storage for AzureBlobStorageModelStore {
    async fn add_model(&self, model_name: ModelName, model_path: &str) -> anyhow::Result<()> { todo!() }

    async fn update_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        todo!()
    }

    fn get_model(&self, model_name: ModelName) -> Option<Ref<ModelName, Arc<Model>>> {
        todo!()
    }

    fn get_models(&self) -> anyhow::Result<Vec<Metadata>> {
        todo!()
    }

    fn delete_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn successfully_load_models_from_azure_blob_storage_model_store() {
        let storage_container_name = "modelstore".to_string();
        let model_store = AzureBlobStorageModelStore::new(storage_container_name).await;

        match model_store {
            Ok(_) => {}
            Err(e) => { println!("ERROR: {}", e)}
        }

        // Assert
        // assert!(model_store.is_ok());
        // assert_ne!(model_store.unwrap().models.len(), 0);
    }

}