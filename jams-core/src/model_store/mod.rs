use crate::model_store::aws::s3::S3ModelStore;
use crate::model_store::azure::blob_storage::AzureBlobStorageModelStore;
use crate::model_store::local::filesystem::LocalModelStore;
use crate::model_store::storage::{Metadata, Model, ModelName, Storage};
use dashmap::mapref::one::Ref;
use std::sync::Arc;
use std::time;

pub mod aws;
pub mod azure;
pub mod common;
mod fetcher;
pub mod local;
pub mod storage;

/// Enum representing different types of model stores.
///
/// This enum can represent models stored in Azure, AWS, or locally. Each variant
/// contains a type representing the specific model store implementation.
///
/// # Variants
/// - `Azure`: Represents a model store on Azure Blob Storage.
/// - `AWS`: Represents a model store on AWS S3.
/// - `Local`: Represents a local file-based model store.
pub enum ModelStore {
    /// Azure Blob Storage model store.
    Azure(AzureBlobStorageModelStore),

    /// AWS S3 model store.
    AWS(S3ModelStore),

    /// Local model store.
    Local(LocalModelStore),
}

impl ModelStore {
    /// Adds a new model to the store.
    ///
    /// This method calls the `add_model` function on the underlying model store
    /// implementation (Azure, AWS, or Local). It requires the model name to add.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be added to the store.
    ///
    /// # Returns
    ///
    /// This method returns an `anyhow::Result<()>` indicating success or failure.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying model store fails to add the model.
    pub async fn add_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        match self {
            ModelStore::Azure(azure) => azure.add_model(model_name).await,
            ModelStore::AWS(aws) => aws.add_model(model_name).await,
            ModelStore::Local(local) => local.add_model(model_name).await,
        }
    }

    /// Updates an existing model in the store.
    ///
    /// This method calls the `update_model` function on the underlying model store
    /// implementation (Azure, AWS, or Local). It requires the model name to update.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be updated in the store.
    ///
    /// # Returns
    ///
    /// This method returns an `anyhow::Result<()>` indicating success or failure.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying model store fails to update the model.
    pub async fn update_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        match self {
            ModelStore::Azure(azure) => azure.update_model(model_name).await,
            ModelStore::AWS(aws) => aws.update_model(model_name).await,
            ModelStore::Local(local) => local.update_model(model_name).await,
        }
    }

    /// Retrieves a model from the store by its name.
    ///
    /// This method calls the `get_model` function on the underlying model store
    /// implementation (Azure, AWS, or Local). It requires the model name to retrieve.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to retrieve from the store.
    ///
    /// # Returns
    ///
    /// This method returns an `Option<Ref<ModelName, Arc<Model>>>` where `Some(model)`
    /// is returned if the model is found, or `None` if the model does not exist in the store.
    pub fn get_model(&self, model_name: ModelName) -> Option<Ref<ModelName, Arc<Model>>> {
        match self {
            ModelStore::Azure(azure) => azure.get_model(model_name),
            ModelStore::AWS(aws) => aws.get_model(model_name),
            ModelStore::Local(local) => local.get_model(model_name),
        }
    }

    /// Retrieves a list of all models in the store.
    ///
    /// This method calls the `get_models` function on the underlying model store
    /// implementation (Azure, AWS, or Local) to retrieve a list of all models.
    ///
    /// # Returns
    ///
    /// This method returns a `Result<Vec<Metadata>>` containing the metadata of all models
    /// in the store. If the store fails to fetch the models, an error is returned.
    pub fn get_models(&self) -> anyhow::Result<Vec<Metadata>> {
        match self {
            ModelStore::Azure(azure) => azure.get_models(),
            ModelStore::AWS(aws) => aws.get_models(),
            ModelStore::Local(local) => local.get_models(),
        }
    }

    /// Deletes a model from the store.
    ///
    /// This method calls the `delete_model` function on the underlying model store
    /// implementation (Azure, AWS, or Local). It requires the model name to delete.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to delete from the store.
    ///
    /// # Returns
    ///
    /// This method returns an `anyhow::Result<()>` indicating success or failure.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying model store fails to delete the model.
    pub fn delete_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        match self {
            ModelStore::Azure(azure) => azure.delete_model(model_name),
            ModelStore::AWS(aws) => aws.delete_model(model_name),
            ModelStore::Local(local) => local.delete_model(model_name),
        }
    }

    /// Polls the model store for updates at a specified interval.
    ///
    /// This method calls the `poll` function on the underlying model store
    /// implementation (Azure, AWS, or Local). It periodically checks the store for
    /// any updates.
    ///
    /// # Arguments
    ///
    /// * `interval` - The duration between each poll check.
    ///
    /// # Returns
    ///
    /// This method returns an `anyhow::Result<()>` indicating success or failure.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying model store fails during polling.
    pub async fn poll(&self, interval: time::Duration) -> anyhow::Result<()> {
        match self {
            ModelStore::Azure(azure) => azure.poll(interval).await,
            ModelStore::AWS(aws) => aws.poll(interval).await,
            ModelStore::Local(local) => local.poll(interval).await,
        }
    }
}
