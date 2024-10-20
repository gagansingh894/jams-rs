use crate::common::server;
use jams_core::manager::Manager;
use jams_core::model_store::azure_blob_storage::AzureBlobStorageModelStore;
use jams_core::model_store::local::LocalModelStore;
use jams_core::model_store::s3::S3ModelStore;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::env;
use std::sync::Arc;

/// AppState struct holds application state.
pub struct AppState {
    /// The manager component wrapped in an Arc (atomic reference count) for shared ownership.
    pub manager: Arc<Manager>,
    /// A thread pool for executing CPU-bound tasks asynchronously.
    pub cpu_pool: ThreadPool,
}

/// Builds the application state from the provided configuration.
///
/// This function initializes the necessary components based on the given configuration,
/// including the model store (local or S3-based) and a thread pool for CPU-intensive tasks.
///
/// # Arguments
///
/// * `config` - The server configuration.
///
/// # Returns
///
/// * `Result<Arc<AppState>>` - The application state wrapped in a `Result` and an `Arc`.
///
/// # Errors
///
/// This function returns an error if:
/// * The number of worker threads is less than 1.
/// * The S3 bucket name is not specified when `with_s3_model_store` is true.
/// * Any failure occurs during the initialization of the thread pool, model store, or manager.
///
/// # Environment Variables
///
/// * `MODEL_STORE_DIR` - The directory to store models locally (optional).
/// * `S3_BUCKET_NAME` - The name of the S3 bucket to store models (required if `with_s3_model_store` is true).
///
pub async fn build_app_state_from_config(config: server::Config) -> anyhow::Result<Arc<AppState>> {
    let model_dir = config.model_dir.unwrap_or_else(|| {
        // search for environment variable
        env::var("MODEL_STORE_DIR").unwrap_or_else(|_| "".to_string())
    });

    let worker_pool_threads = config.num_workers.unwrap_or(2);
    let with_s3_model_store = config.with_s3_model_store.unwrap_or(false);
    let with_azure_model_store = config.with_azure_model_store.unwrap_or(false);

    // initialize threadpool for cpu intensive tasks
    if worker_pool_threads < 1 {
        anyhow::bail!("At least 1 worker is required for rayon threadpool")
    }
    let cpu_pool = ThreadPoolBuilder::new()
        .num_threads(worker_pool_threads)
        .build()
        .expect("Failed to build rayon threadpool ❌");

    log::info!(
        "Rayon threadpool started with {} workers ⚙️",
        worker_pool_threads
    );

    // initialize manager with madel store
    if with_s3_model_store && with_azure_model_store {
        anyhow::bail!("Use of more than 1 model store backend is not permitted. Please use one of the following - Local File System/AWS S3/ Azure Blob Storage")
    }

    let manager = if with_s3_model_store {
        let s3_bucket_name = config.s3_bucket_name.unwrap_or_else(|| {
            // search for environment variable
            env::var("S3_BUCKET_NAME").expect("S3 bucket name not specified ❌. Either set the S3_BUCKET_NAME env variable or provide the value using --s3-bucket-name flag ")
        });
        let model_store = S3ModelStore::new(s3_bucket_name)
            .await
            .expect("Failed to create S3 model store ❌");
        Arc::new(
            Manager::new(Arc::new(model_store))
                .await
                .expect("Failed to initialize manager ❌"),
        )
    } else if with_azure_model_store {
        let azure_storage_container_name = config.azure_storage_container_name.unwrap_or_else(|| {
            // search for environment variable
            env::var("AZURE_STORAGE_CONTAINER_NAME").expect("Azure Storage container name not specified ❌. Either set the AZURE_STORAGE_CONTAINER_NAME env variable or provide the value using --azure-container-name flag ")
        });
        let model_store = AzureBlobStorageModelStore::new(azure_storage_container_name)
            .await
            .expect("Failed to create Azure model store ❌");
        Arc::new(
            Manager::new(Arc::new(model_store))
                .await
                .expect("Failed to initialize manager ❌"),
        )
    } else {
        let model_store = LocalModelStore::new(model_dir)
            .await
            .expect("Failed to create local model store ❌");
        Arc::new(
            Manager::new(Arc::new(model_store))
                .await
                .expect("Failed to initialize manager ❌"),
        )
    };

    // setup shared state
    Ok(Arc::new(AppState { manager, cpu_pool }))
}
