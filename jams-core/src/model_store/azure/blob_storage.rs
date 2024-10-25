use crate::model_store::azure::common::download_blob;
use crate::model_store::common::{cleanup, DOWNLOADED_MODELS_DIRECTORY_NAME_PREFIX};
use crate::model_store::fetcher::Fetcher;
use crate::model_store::storage::{
    append_model_format, extract_framework_from_path, load_predictor, Metadata, Model, ModelName,
    Storage,
};
use async_trait::async_trait;
use azure_storage::{CloudLocation, StorageCredentials};
use azure_storage_blobs::prelude::{BlobServiceClient, ContainerClient};
use chrono::Utc;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use std::env;
use std::sync::Arc;
use std::time::Duration;
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
        let container_client = match build_azure_storage_client(use_azurite()) {
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
        std::fs::create_dir(model_store_dir.clone())?;

        // Check if Azure blob storage is empty, if yes then return models dashmap as empty
        if container_client.is_empty(None).await? {
            log::warn!(
                        "No models found in the Azure model storage container hence no models will be loaded ⚠️"
                    );
            let models: DashMap<ModelName, Arc<Model>> = DashMap::new();
            Ok(Self {
                models: Arc::new(models),
                container_client,
                model_store_dir,
            })
        } else {
            // Fetch the models from Azure Blob Storage
            let models = match container_client
                .fetch_models(None, model_store_dir.clone())
                .await
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
fn build_azure_storage_client(use_azurite: bool) -> anyhow::Result<BlobServiceClient> {
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
    let mut service_client_builder = BlobServiceClient::builder(account, storage_credentials);
    if use_azurite {
        let hostname = env::var("AZURITE_HOSTNAME").unwrap_or("0.0.0.0".to_string());
        service_client_builder = service_client_builder.cloud_location(CloudLocation::Emulator {
            address: hostname,
            port: 10000,
        })
    }

    Ok(service_client_builder.blob_service_client())
}

#[async_trait]
impl Storage for AzureBlobStorageModelStore {
    /// Adds a new model to the Azure Blob Storage model store.
    ///
    /// This method downloads the model from Azure Blob Storage, extracts the framework, loads the model into memory,
    /// and stores it in the `models` hashmap.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model.
    /// * `_model_path` - The path to the model file (unused in this implementation as models are fetched from Azure Blob Storage).
    ///
    /// # Returns
    ///
    /// * `Result<()>` - An empty result indicating success or an error.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// * The model cannot be downloaded from Azure Blob Storage.
    /// * The framework cannot be extracted from the model path.
    /// * The model cannot be loaded into memory.
    async fn add_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        // Prepare the blob key from model_name
        // It is assumed that model will always be present as a .tar.gz file in Azure Blob Storage
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
        // At this point we have extracted the tar ball from Azure Blob Storage
        // It is assumed that the name of tar ball and the actual mode name is the same
        // Based on the framework, the path is modified by appending the format
        // If pytorch -> append '.pt'
        // If lightgbm -> append '.txt'

        // Extract model framework
        let model_framework = match extract_framework_from_path(model_name.clone()) {
            None => {
                anyhow::bail!("Failed to extract framework from path");
            }
            Some(model_framework) => model_framework,
        };

        // We will not be using the model_path from the request
        // as we have all the information to load the model
        let model_path = append_model_format(
            model_framework,
            format!("{}/{}", self.model_store_dir, model_name.clone()),
        );

        // Load the model into memory
        match load_predictor(model_framework, model_path.as_str()).await {
            Ok(predictor) => {
                let sanitized_model_name =
                    match model_name.strip_prefix(format!("{}-", model_framework).as_str()) {
                        None => {
                            anyhow::bail!("Failed to sanitize model name");
                        }
                        Some(name) => name.to_string(),
                    };

                let now = Utc::now();
                let model = Model::new(
                    predictor,
                    sanitized_model_name.clone(),
                    model_framework,
                    model_path.to_string(),
                    now.to_rfc2822(),
                );
                self.models.insert(sanitized_model_name, Arc::new(model));
                Ok(())
            }
            Err(e) => {
                anyhow::bail!("Failed to add new model: {e}")
            }
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

    /// Periodically polls the model store in Azure Blob Storage to fetch and update models.
    ///
    /// This asynchronous function waits for a specified time interval, then attempts to fetch models
    /// from the Azure Blob Storage container, and updates the internal model cache (`self.models`).
    /// The polling interval ensures that the model store is regularly updated with new models, if available.
    ///
    /// # Arguments
    ///
    /// * `interval` - A `Duration` representing the time interval between each polling operation.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the models were successfully fetched and updated in the model store.
    /// * `Err(anyhow::Error)` - If an error occurs during the fetch or update process, such as when
    ///   the models fail to be retrieved from Azure Blob Storage.
    ///
    async fn poll(&self, interval: Duration) -> anyhow::Result<()> {
        // poll every n time interval
        tokio::time::sleep(interval).await;

        log::info!("Polling model store ⌛");
        // let models = match self.fetch_models(None, self.model_store_dir.clone()).await
        // {
        //     Ok(models) => {
        //         log::info!("Successfully fetched valid models from Azure Blob Storage ✅");
        //         models
        //     }
        //     Err(e) => {
        //         anyhow::bail!("Failed to fetch models ❌ - {}", e.to_string());
        //     }
        // };
        //
        // for (model_name, model) in models {
        //     self.models.insert(model_name, model);
        // }

        Ok(())
    }
}

fn use_azurite() -> bool {
    env::var("USE_AZURITE").unwrap_or_default() == "true"
}

#[cfg(test)]
mod tests {
    use super::*;
    use azure_core::tokio::fs::FileStreamBuilder;
    use azure_storage_blobs::prelude::PublicAccess;
    use rand::Rng;
    use std::time::Duration;
    use tokio::fs::File;

    fn setup_client() -> BlobServiceClient {
        if !use_azurite() {
            panic!("USE_AZURITE env not set")
        }
        build_azure_storage_client(true).unwrap()
    }

    async fn setup_test_dependencies(
        client: BlobServiceClient,
        azure_storage_container_name: String,
    ) {
        create_test_azure_storage_container(client.clone(), azure_storage_container_name.clone())
            .await;
        upload_blobs_to_azure_storage_containers(
            client
                .clone()
                .container_client(azure_storage_container_name.clone()),
        )
        .await
    }
    fn generate_container_name() -> String {
        let mut rng = rand::thread_rng();
        let random_number = rng.gen_range(0..999999);
        format!("jams-model-store-test-{}", random_number)
    }

    async fn create_test_azure_storage_container(
        client: BlobServiceClient,
        azure_storage_container_name: String,
    ) {
        let container_client = client.container_client(azure_storage_container_name);
        container_client
            .create()
            .public_access(PublicAccess::Container)
            .await
            .unwrap()
    }

    async fn upload_blobs_to_azure_storage_containers(client: ContainerClient) {
        let mut dir = tokio::fs::read_dir("tests/model_storage/model_store")
            .await
            .unwrap();

        while let Some(entry) = dir.next_entry().await.unwrap() {
            let file = File::open(entry.path()).await.unwrap();
            let file_stream = FileStreamBuilder::new(file).build().await.unwrap();
            client
                .blob_client(entry.file_name().into_string().unwrap())
                .put_block_blob(file_stream)
                .await
                .unwrap();
        }
    }

    async fn delete_models_for_test(container_client: ContainerClient) {
        container_client.delete().await.unwrap()
    }

    #[tokio::test]
    async fn successfully_create_azure_blob_storage_model_store_without_models_and_then_add_one() {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        create_test_azure_storage_container(client.clone(), container_name.clone()).await;

        // create azure model store without models
        let model_store = AzureBlobStorageModelStore::new(container_name.clone()).await;

        // assert
        assert!(model_store.is_ok());
        let model_store = model_store.unwrap();
        assert_eq!(model_store.models.len(), 0);

        // add model - upload model to s3 then call add model
        upload_blobs_to_azure_storage_containers(
            client.clone().container_client(container_name.clone()),
        )
        .await;

        let resp = model_store
            .add_model("catboost-titanic_model".to_string())
            .await;
        assert!(resp.is_ok());
        assert_eq!(model_store.models.len(), 1);
    }

    #[tokio::test]
    async fn successfully_load_models_from_azure_blob_storage_model_store() {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        setup_test_dependencies(client.clone(), container_name.clone()).await;

        let model_store = AzureBlobStorageModelStore::new(container_name.clone()).await;

        // Assert
        assert!(model_store.is_ok());
        assert_ne!(model_store.unwrap().models.len(), 0);

        // Cleanup
        delete_models_for_test(client.container_client(container_name)).await
    }

    #[tokio::test]
    async fn successfully_get_model_from_azure_blob_storage_model_store() {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        setup_test_dependencies(client.clone(), container_name.clone()).await;

        // load models
        let model_store = AzureBlobStorageModelStore::new(container_name.clone())
            .await
            .unwrap();
        let model = model_store.get_model("my_awesome_reg_model".to_string());

        // assert
        assert!(model.is_some());

        // cleanup
        delete_models_for_test(client.container_client(container_name)).await
    }

    #[tokio::test]
    async fn fails_to_get_model_from_azure_blob_storage_model_store_when_model_name_is_wrong() {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        setup_test_dependencies(client.clone(), container_name.clone()).await;

        // load models
        let model_store = AzureBlobStorageModelStore::new(container_name.clone())
            .await
            .unwrap();
        let model = model_store.get_model("model_which_does_not_exist".to_string());

        // assert
        assert!(model.is_none());

        // cleanup
        delete_models_for_test(client.container_client(container_name)).await
    }

    #[tokio::test]
    async fn successfully_get_models_from_azure_blob_storage_model_store() {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        setup_test_dependencies(client.clone(), container_name.clone()).await;

        // load models
        let model_store = AzureBlobStorageModelStore::new(container_name.clone())
            .await
            .unwrap();
        let models = model_store.get_models();

        // assert
        assert!(models.is_ok());
        assert_ne!(models.unwrap().len(), 0);

        // cleanup
        delete_models_for_test(client.container_client(container_name)).await
    }

    #[tokio::test]
    async fn successfully_deletes_model_in_the_azure_blob_storage_model_store() {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        setup_test_dependencies(client.clone(), container_name.clone()).await;

        // load models
        let model_store = AzureBlobStorageModelStore::new(container_name.clone())
            .await
            .unwrap();
        let deletion = model_store.delete_model("my_awesome_penguin_model".to_string());

        // assert
        assert!(deletion.is_ok());

        // cleanup
        delete_models_for_test(client.container_client(container_name)).await
    }

    #[tokio::test]
    async fn fails_to_deletes_model_in_the_azure_blob_storage_model_store() {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        setup_test_dependencies(client.clone(), container_name.clone()).await;

        // load models
        let model_store = AzureBlobStorageModelStore::new(container_name.clone())
            .await
            .unwrap();
        let deletion = model_store.delete_model("model_which_does_not_exist".to_string());

        // assert
        assert!(deletion.is_err());

        // cleanup
        delete_models_for_test(client.container_client(container_name)).await
    }

    #[tokio::test]
    async fn successfully_update_model_in_the_azure_blob_storage_model_store() {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        setup_test_dependencies(client.clone(), container_name.clone()).await;
        let model_name = "my_awesome_reg_model".to_string();

        // load models
        let model_store = AzureBlobStorageModelStore::new(container_name.clone())
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_secs_f32(1.25)).await;

        // retrieve timestamp from existing to model for assertion
        let model = model_store
            .get_model(model_name.clone())
            .unwrap()
            .to_owned();

        // update model
        let update = model_store.update_model(model_name.clone()).await;
        assert!(update.is_ok());
        let updated_model = model_store
            .get_model(model_name.clone())
            .unwrap()
            .to_owned();

        // assert
        assert_eq!(model.info.name, updated_model.info.name);
        assert_eq!(model.info.path, updated_model.info.path);
        assert_ne!(model.info.last_updated, updated_model.info.last_updated); // as model will be updated

        // cleanup
        delete_models_for_test(client.container_client(container_name)).await
    }

    #[tokio::test]
    async fn fails_to_update_model_in_the_azure_blob_storage_model_store_when_model_name_is_incorrect(
    ) {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        setup_test_dependencies(client.clone(), container_name.clone()).await;
        let incorrect_model_name = "my_awesome_reg_model_incorrect".to_string();

        // load models
        let model_store = AzureBlobStorageModelStore::new(container_name.clone())
            .await
            .unwrap();

        // update model with incorrect model name
        let update = model_store.update_model(incorrect_model_name).await;

        // assert
        assert!(update.is_err());

        // cleanup
        delete_models_for_test(client.container_client(container_name)).await
    }

    #[tokio::test]
    async fn successfully_add_model_in_the_azure_blob_storage_model_store() {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        setup_test_dependencies(client.clone(), container_name.clone()).await;

        // load models
        let model_store = AzureBlobStorageModelStore::new(container_name.clone())
            .await
            .unwrap();

        // delete model to set up test
        model_store
            .delete_model("titanic_model".to_string())
            .unwrap();
        // assert that model is not present
        let model = model_store.get_model("titanic_model".to_string());
        assert!(model.is_none());
        let num_models = model_store.get_models().unwrap().len();

        // add model - unlike local model store we will pass the blob name without .tar.gz in the model name
        // model_path is not required when adding models via Azure Blob Storage model store
        let add = model_store
            .add_model("catboost-titanic_model".to_string())
            .await;
        let num_models_after_add = model_store.get_models().unwrap().len();

        // assert
        assert!(add.is_ok());
        let model = model_store.get_model("titanic_model".to_string());
        assert!(model.is_some());
        assert_eq!(num_models_after_add - num_models, 1);

        // cleanup
        delete_models_for_test(client.container_client(container_name)).await
    }

    #[tokio::test]
    async fn fails_to_add_model_in_the_azure_blob_storage_model_store_when_the_model_name_is_wrong()
    {
        // setup
        let client = setup_client();
        let container_name = generate_container_name();
        setup_test_dependencies(client.clone(), container_name.clone()).await;

        // load models
        let model_store = AzureBlobStorageModelStore::new(container_name.clone())
            .await
            .unwrap();

        // add model
        let add = model_store
            .add_model("my_awesome_penguin_model_wrong_azure_blob_storage_key".to_string())
            .await;

        // assert
        assert!(add.is_err());

        // cleanup
        delete_models_for_test(client.container_client(container_name)).await
    }
}
