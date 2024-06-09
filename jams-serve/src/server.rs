use crate::service::{
    add_model, delete_model, get_models, healthcheck, predict, root, update_model,
};
use axum::routing::{delete, get, post, put};
use axum::Router;
use jams_core::manager::Manager;
use jams_core::model_store::local::LocalModelStore;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::Arc;
use tokio::signal;
use tower_http::trace::TraceLayer;

/// AppState struct holds application state.
pub struct AppState {
    /// The manager component wrapped in an Arc (atomic reference count) for shared ownership.
    pub manager: Arc<Manager>,
    /// A thread pool for executing CPU-bound tasks asynchronously.
    pub cpu_pool: ThreadPool,
}

pub fn build_router(
    model_dir: String,
    use_debug_log: bool,
    worker_pool_threads: usize,
) -> anyhow::Result<Router> {
    // initialize tracing
    let mut log_level = tracing::Level::INFO;
    if use_debug_log {
        log_level = tracing::Level::TRACE
    }

    tracing_subscriber::fmt().with_max_level(log_level).init();

    // setup rayon thread pool for cpu intensive task
    let cpu_pool = ThreadPoolBuilder::new()
        .num_threads(worker_pool_threads)
        .build()
        .expect("Failed to build rayon threadpool ❌");

    // setup model store
    let model_store = LocalModelStore::new(model_dir).expect("Failed to create model store ❌");
    let manager = Manager::new(Arc::new(model_store)).expect("Failed to initialize manager ❌");

    // shared state
    let app_state = AppState {
        manager: Arc::new(manager),
        cpu_pool,
    };
    let shared_state = Arc::new(app_state);

    // API routes
    let api_routes = Router::new()
        .route("/models", get(get_models))
        .route("/models", post(add_model))
        .route("/models", put(update_model))
        .route("/models", delete(delete_model))
        .route("/predict", post(predict));

    tracing::info!(
        "Rayon threadpool started with {} workers ⚙️",
        worker_pool_threads
    );

    // build router
    Ok(Router::new()
        .route("/", get(root))
        .route("/healthcheck", get(healthcheck))
        .nest("/api", api_routes)
        .with_state(shared_state)
        .layer(TraceLayer::new_for_http()))
}

pub async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}