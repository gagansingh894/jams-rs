use crate::common::server;
use jams_core::manager::Manager;
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

pub async fn build_app_state_from_config(config: server::Config) -> anyhow::Result<Arc<AppState>> {
    let model_dir = config.model_dir.unwrap_or_else(|| {
        // search for environment variable
        env::var("MODEL_STORE_DIR").unwrap_or_else(|_| "".to_string())
    });

    let worker_pool_threads = config.num_workers.unwrap_or(2);
    let with_s3_model_store = config.with_s3_model_store.unwrap_or(false);

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
    let manager = match with_s3_model_store {
        true => {
            let s3_bucket_name = config.s3_bucket_name.unwrap_or_else(|| {
                // search for environment variable
                env::var("S3_BUCKET_NAME").expect("S3 bucket name not specified ❌. Either set the S3_BUCKET_NAME env variable or provide the value using --s3-bucket-name flag ")
            });
            let model_store = S3ModelStore::new(s3_bucket_name)
                .await
                .expect("Failed to create model store ❌");
            Arc::new(Manager::new(Arc::new(model_store)).expect("Failed to initialize manager ❌"))
        }
        false => {
            let model_store =
                LocalModelStore::new(model_dir).expect("Failed to create model store ❌");
            Arc::new(Manager::new(Arc::new(model_store)).expect("Failed to initialize manager ❌"))
        }
    };

    // setup shared state
    Ok(Arc::new(AppState { manager, cpu_pool }))
}
