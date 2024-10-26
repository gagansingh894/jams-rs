use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;

use crate::model_store::storage::{Model, ModelName};

/// Defines a trait for fetching models from a storage location, which could be
/// an S3 bucket, a storage container, or a filesystem directory.
#[async_trait]
pub trait Fetcher {
    /// Checks if the model store is empty.
    ///
    /// # Parameters
    /// - `artefacts_dir_name`: A `String` representing the name of the directory, S3 bucket,
    ///   storage container, or absolute path where model artefacts are stored.
    ///
    /// # Returns
    /// - `Ok(true)` if the model store has no artefacts.
    /// - `Ok(false)` if there are artefacts present in the model store.
    /// - `Err`: Returns an error wrapped in `anyhow::Result` if the check fails.
    async fn is_empty(&self, artefacts_dir_name: Option<String>) -> anyhow::Result<bool>;

    /// Fetches models from the specified artefacts directory and loads them into a `DashMap`.
    ///
    /// # Parameters
    /// - `artefacts_dir_name`: A `String` that represents the storage location (e.g., an S3 bucket name,
    ///   a storage container, or an absolute path) where models are stored.
    /// - `output_dir`: A `String` specifying the target directory where fetched models will be
    ///   saved. If the source is local, models will be copied here; otherwise, they will be downloaded
    ///   to this location.
    ///
    /// # Returns
    /// - `Ok(DashMap<ModelName, Arc<Model>>)` on success, where each `ModelName` key maps to an
    ///   `Arc<Model>` containing the fetched model.
    /// - `Err`: Returns an error wrapped in `anyhow::Result` if the fetching process fails.
    async fn fetch_models(
        &self,
        artefacts_dir_name: Option<String>,
        output_dir: String,
    ) -> anyhow::Result<DashMap<ModelName, Arc<Model>>>;
}
