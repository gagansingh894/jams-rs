use crate::model_store::common::{
    cleanup, save_and_upack_tarball, DOWNLOADED_MODELS_DIRECTORY_NAME_PREFIX,
};
use crate::model_store::storage::{
    append_model_format, extract_framework_from_path, load_models, load_predictor, Metadata, Model,
    ModelName, Storage,
};
use async_trait::async_trait;
use azure_storage::StorageCredentials;
use azure_storage_blobs::prelude::{BlobServiceClient, ContainerClient};
use bytes::Bytes;
use chrono::Utc;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use futures::StreamExt;
use std::num::NonZeroU32;
use std::sync::Arc;
use uuid::Uuid;

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
    /// Creates a new instance of `AzureBlobStorageModelStore`.
    ///
    /// This function initializes the `AzureBlobStorageModelStore` by:
    /// 1. Ensuring the provided `storage_container_name` is not empty.
    /// 2. Building the Azure Blob Service client using environment variables.
    /// 3. Specifying a temporary directory for storing models downloaded from Azure Blob Storage.
    /// 4. Fetching the models from Azure Blob Storage and loading them into a `DashMap`.
    ///
    /// # Arguments
    ///
    /// * `storage_container_name` - A `String` specifying the name of the Azure Blob Storage container.
    ///
    /// # Returns
    ///
    /// A `Result` containing a new instance of `AzureBlobStorageModelStore` if successful, or an `anyhow::Error` if the operation fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * The `storage_container_name` is empty.
    /// * The Azure Blob Service client cannot be created.
    /// * The models cannot be fetched from Azure Blob Storage.
    pub async fn new(storage_container_name: String) -> anyhow::Result<Self> {
        // Ensure model_dir_uri is not empty, return error if empty
        if storage_container_name.is_empty() {
            anyhow::bail!("Azure storage container name must be specified ❌")
        }
        // Create Azure Blob Service client
        let container_client = match build_azure_storage_client() {
            Ok(blob_service_client) => blob_service_client.container_client(storage_container_name),
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
        let models = match fetch_models(&container_client, model_store_dir.clone()).await {
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

/// Builds an Azure Blob Storage client using environment variables for credentials.
///
/// This function reads the `STORAGE_ACCOUNT` and `STORAGE_ACCESS_KEY` environment variables
/// to construct the credentials required to authenticate with Azure Blob Storage. If these
/// environment variables are not set, the function will return an error.
///
/// # Returns
///
/// A `Result` containing a `BlobServiceClient` if successful, or an `anyhow::Error` if the
/// required environment variables are not set.
///
/// # Errors
///
/// This function will return an error if:
/// * The `STORAGE_ACCOUNT` environment variable is not set.
/// * The `STORAGE_ACCESS_KEY` environment variable is not set.
fn build_azure_storage_client() -> anyhow::Result<BlobServiceClient> {
    let account = match std::env::var("STORAGE_ACCOUNT") {
        Ok(account) => account,
        Err(_) => {
            anyhow::bail!("Azure STORAGE_ACCOUNT env variable not set ❌")
        }
    };
    let access_key = match std::env::var("STORAGE_ACCESS_KEY") {
        Ok(ak) => ak,
        Err(_) => {
            anyhow::bail!("Azure STORAGE_ACCESS_KEY env variable not set ❌")
        }
    };
    let storage_credentials = StorageCredentials::access_key(account.clone(), access_key);

    Ok(BlobServiceClient::new(account, storage_credentials))
}

#[async_trait]
impl Storage for AzureBlobStorageModelStore {
    async fn add_model(&self, model_name: ModelName, _model_path: &str) -> anyhow::Result<()> {
        // Prepare the blob key from model_name
        // It is assumed that model will always be present as a .tar.gz file in S3
        // Panic otherwise
        let blob_name = format!("{}.tar.gz", model_name);

        match download_blob(
            &self.container_client,
            blob_name,
            self.model_store_dir.clone(),
        )
        .await
        {
            Ok(_) => {
                log::info!("Downloaded blob from azure storage ✅");
            }
            Err(e) => {
                anyhow::bail!(
                    "Failed to download blob from azure storage ❌️: {}",
                    e.to_string()
                );
            }
        }

        // todo: Figure out a better approach or provide utils function in python which pack the artefacts in the required format
        // At this point we have extracted the tar ball from S3
        // It is assumed that the name of tar ball and the actual mode name is the same
        // Based on the framework, the path is modified by appending the format
        // If pytorch -> append '.pt'
        // If lightgbm -> append '.txt'

        // We will not be using the model_path from the request
        // as we have all the information to load the model
        let model_path = format!("{}/{}", self.model_store_dir, model_name);

        // Load the model into memory
        match extract_framework_from_path(model_path.clone()) {
            None => {
                anyhow::bail!("Failed to extract framework from path");
            }
            Some(framework) => match load_predictor(
                framework,
                append_model_format(framework, model_path.clone()).as_str(),
            )
            .await
            {
                Ok(predictor) => {
                    let now = Utc::now();
                    let model = Model::new(
                        predictor,
                        model_name.clone(),
                        framework,
                        model_path.to_string(),
                        now.to_rfc2822(),
                    );
                    self.models.insert(model_name, Arc::new(model));
                    Ok(())
                }
                Err(e) => {
                    anyhow::bail!("Failed to add new model: {e}")
                }
            },
        }
    }

    /// Asynchronously updates a specified model by downloading the latest version from Azure Blob Storage.
    ///
    /// This function performs the following steps:
    /// 1. Removes the existing model from the internal model store.
    /// 2. Constructs the blob name from the model name and framework.
    /// 3. Downloads the blob from Azure Blob Storage and unpacks it.
    /// 4. Loads the model predictor from the unpacked files.
    /// 5. Creates a new `Model` instance and updates the internal model store.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be updated.
    ///
    /// # Returns
    ///
    /// A `Result` which is `Ok(())` if the operation is successful, or an `anyhow::Error` if the operation fails.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// * The specified model does not exist in the internal model store.
    /// * Downloading the blob from Azure Blob Storage fails.
    /// * Loading the predictor from the unpacked files fails.
    async fn update_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        match self.models.remove(model_name.as_str()) {
            None => {
                anyhow::bail!(
                    "Failed to update as the specified model {} does not exist",
                    model_name
                )
            }
            Some(model) => {
                let (model_framework, model_path) =
                    (model.1.info.framework, model.1.info.path.as_str());

                // Prepare the blob name from model_name
                let blob_name = format!("{}-{}.tar.gz", model_framework, model_name);

                match download_blob(
                    &self.container_client,
                    blob_name,
                    self.model_store_dir.clone(),
                )
                .await
                {
                    Ok(_) => {
                        log::info!("Downloaded blob from azure storage ✅");

                        match load_predictor(model_framework, model_path).await {
                            Ok(predictor) => {
                                let now = Utc::now();
                                let model = Model::new(
                                    predictor,
                                    model_name.clone(),
                                    model_framework,
                                    model_path.to_string(), // todo: use Azure path here and not the local model dir path
                                    now.to_rfc2822(),
                                );
                                self.models.insert(model_name.clone(), Arc::new(model));
                                Ok(())
                            }
                            Err(e) => {
                                anyhow::bail!(
                                    "Failed to update the specified model {}: {}",
                                    model_name,
                                    e
                                )
                            }
                        }
                    }
                    Err(e) => {
                        anyhow::bail!("Failed to download blob ❌: {}", e)
                    }
                }
            }
        }
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

/// Asynchronously fetches models from an Azure Blob Storage container, unpacks them, and loads them into a `DashMap`.
///
/// # Arguments
///
/// * `client` - A reference to an Azure Blob Storage `ContainerClient`.
/// * `model_store_dir` - A `String` specifying the directory where the models will be stored.
///
/// # Returns
///
/// A `Result` containing a `DashMap` where the keys are `ModelName`s and the values are `Arc<Model>`s, or an `anyhow::Error` if the operation fails.
///
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
async fn fetch_models(
    client: &ContainerClient,
    model_store_dir: String,
) -> anyhow::Result<DashMap<ModelName, Arc<Model>>> {
    let max_results = NonZeroU32::new(10).unwrap();

    // List the blobs in the container
    let mut stream = client.list_blobs().max_results(max_results).into_stream();
    // For each blob, create a blob client and download the blob
    while let Some(result) = stream.next().await {
        match result {
            Ok(result) => {
                for blob in result.blobs.blobs() {
                    // Download blob to model_store_dir
                    match download_blob(client, blob.clone().name, model_store_dir.clone()).await {
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

/// Asynchronously downloads a blob from Azure Blob Storage, saves it to a temporary path, and unpacks it.
///
/// This function performs the following steps:
/// 1. Creates a `BlobClient` for the specified blob.
/// 2. Streams the blob data in chunks and collects it into a complete byte vector.
/// 3. Saves the collected data to a temporary path and unpacks it into the specified model storage directory.
///
/// # Arguments
///
/// * `client` - A reference to an Azure Blob Storage `ContainerClient`.
/// * `blob_name` - A `String` specifying the name of the blob to be downloaded.
/// * `temp_path` - A `String` specifying the temporary directory path where the blob will be saved.
/// * `model_store_dir` - A `String` specifying the directory where the unpacked model will be stored.
///
/// # Returns
///
/// A `Result` which is `Ok(())` if the operation is successful, or an `anyhow::Error` if the operation fails.
///
/// # Errors
///
/// This function will return an error if:
/// * Streaming or collecting the blob data fails.
/// * Saving and unpacking the blob data fails.
async fn download_blob(
    client: &ContainerClient,
    blob_name: String,
    model_store_dir: String,
) -> anyhow::Result<()> {
    let temp_path = match tempfile::Builder::new().prefix("models").tempdir() {
        Ok(dir) => match dir.path().to_str() {
            None => {
                anyhow::bail!("Failed to convert TempDir to String ❌")
            }
            Some(p) => p.to_string(),
        },
        Err(e) => {
            anyhow::bail!("Failed to create temp directory: {}", e)
        }
    };
    let temp_save_path = format!("{}/{}", temp_path, blob_name.clone());

    let blob_client = client.blob_client(blob_name.clone());
    let mut blob_stream = blob_client.get().chunk_size(0x2000u64).into_stream();
    let mut complete_response: Vec<u8> = vec![];
    while let Some(value) = blob_stream.next().await {
        let data = match value {
            Ok(response) => match response.data.collect().await {
                Ok(data) => data.to_vec(),
                Err(e) => {
                    anyhow::bail!("Failed to convert bytes: {}", e)
                }
            },
            Err(e) => {
                anyhow::bail!("Failed to collect data to bytes: {}", e)
            }
        };
        complete_response.extend(&data);
    }

    match save_and_upack_tarball(
        temp_save_path.as_str(),
        blob_name.clone(),
        Bytes::copy_from_slice(&complete_response[..]),
        model_store_dir.as_str(),
    ) {
        Ok(_) => {
            // Do nothing
        }
        Err(e) => {
            log::warn!("Failed to save artefact {} ⚠️: {}", blob_name, e.to_string())
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn successfully_load_models_from_azure_blob_storage_model_store() {
        let storage_container_name = "modelstore".to_string();

        let model_store = AzureBlobStorageModelStore::new(storage_container_name).await;

        // Assert
        assert!(model_store.is_ok());
        assert_ne!(model_store.unwrap().models.len(), 0);
    }
}
