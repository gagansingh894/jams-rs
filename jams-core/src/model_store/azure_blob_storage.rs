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
use crate::model_store::storage::{Metadata, Model, ModelName, Storage};

/// A struct representing a model store that interfaces with azure blob storage.
pub struct AzureBlobStorageModelStore {
    /// A thread-safe map of model names to their corresponding models.
    pub models: Arc<DashMap<ModelName, Arc<Model>>>,
    /// Azure container client for interacting with storage container and listing blobs
    /// A new blob client will be created for each blob to download it to the local file system
    container_client: ContainerClient,
}

impl AzureBlobStorageModelStore {
   pub async fn new(storage_container_name: String) -> anyhow::Result<Self> {
       let account = std::env::var("STORAGE_ACCOUNT").unwrap();
       let access_key = std::env::var("STORAGE_ACCESS_KEY").unwrap();
       let storage_credentials = StorageCredentials::access_key(account.clone(), access_key);

       let container_client = BlobServiceClient::new(account, storage_credentials)
           .container_client(storage_container_name);
       let models: Arc<DashMap<ModelName, Arc<Model>>> = Arc::new(DashMap::new());

       Ok(AzureBlobStorageModelStore{models, container_client})
   }
}

async fn fetch_models(client: ContainerClient) {
    let max_results = NonZeroU32::new(10).unwrap();
    // list the blobs in the container
    let mut stream = client
        .list_blobs()
        .max_results(max_results)
        .into_stream();
    // for each blob, create a blob client and download the blob
    while let Some(result) = stream.next().await {
        match result {
            Ok(result) => {
                println!("{:?}", result.max_results);
            }
            Err(_) => {}
        }

    }
    // once the dir is readu, repeat the same process as in S3
}

#[async_trait]
impl Storage for AzureBlobStorageModelStore {
    async fn add_model(&self, model_name: ModelName, model_path: &str) -> anyhow::Result<()> {
        todo!()
    }

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
    use std::num::NonZeroU32;
    use azure_storage::prelude::*;
    use azure_storage_blobs::prelude::*;
    use futures::stream::StreamExt;
    #[tokio::test]
    async fn explore_azure_blob() {
        // First we retrieve the account name and access key from environment variables.
        let account = std::env::var("STORAGE_ACCOUNT").unwrap_or_else(|_| "jamsmodelstore".to_string());
        let access_key = std::env::var("STORAGE_ACCESS_KEY").unwrap_or_else(|_| "7NlJB6EI2s7K2RCAI/TzR6h22lH3IR931NB1rx6y242hBEuDaNIoDt6BWXk99NLxvWxD4b1RWHi/+AStja7d4Q==".to_string());
        let container = std::env::var("STORAGE_CONTAINER").unwrap_or_else(|_| "modelstore".to_string());

        // ideally we will list all the blobs inside the container
        // a blob = s3 key

        let storage_credentials = StorageCredentials::access_key(account.clone(), access_key);
        let service_client = BlobServiceClient::new(account, storage_credentials);
        let container_client = service_client.container_client(container);
        // let container_client = BlobServiceClient::new(account, storage_credentials).container_client(container);


        let max_results = NonZeroU32::new(3).unwrap();
        let mut stream = container_client
            .list_blobs()
            .max_results(max_results)
            .into_stream();

        let mut count = 0;
        while let Some(result) = stream.next().await {
            let result = result.unwrap();
            for blob in result.blobs.blobs() {
                count += 1;
                println!(
                    "\t{}\t{} MB",
                    blob.clone().name,
                    blob.clone().properties.content_length / (1024 * 1024)
                );
                let blob_client = container_client.blob_client(blob.clone().name);
            }
        }
        println!("List blob returned {count} blobs.");
    }

}