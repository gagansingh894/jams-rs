use crate::common::state::AppState;
use crate::http::service::{
    add_model, delete_model, get_models, healthcheck, predict, update_model,
};
use axum::routing::{delete, get, post, put};
use axum::Router;
use jams_core::manager::Manager;
use jams_core::model_store::local::LocalModelStore;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use tower_http::trace::TraceLayer;

pub fn build_router(model_dir: String, worker_pool_threads: usize) -> anyhow::Result<Router> {
    if worker_pool_threads < 1 {
        anyhow::bail!("At least 1 worker is required for rayon threadpool")
    }

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

    log::info!(
        "Rayon threadpool started with {} workers ⚙️",
        worker_pool_threads
    );

    // build router
    Ok(Router::new()
        .route("/healthcheck", get(healthcheck))
        .nest("/api", api_routes)
        .with_state(shared_state)
        .layer(TraceLayer::new_for_http()))
}

#[cfg(test)]
mod tests {
    use crate::http::router::build_router;

    #[test]
    fn successfully_build_router() {
        // Arrange
        let model_dir = "".to_string();
        let worker_pool_threads = 1;

        // Act
        let router = build_router(model_dir, worker_pool_threads);

        // Assert
        assert!(router.is_ok())
    }

    #[test]
    fn failed_to_build_router_due_to_zero_worker_in_rayon_threadpool() {
        // Arrange
        let model_dir = "".to_string();
        let worker_pool_threads = 0;

        // Act
        let router = build_router(model_dir, worker_pool_threads);

        // Assert
        assert!(router.is_err())
    }

    #[test]
    #[should_panic]
    fn failed_to_build_router_because_manager_is_unable_to_initialize() {
        // Arrange
        let model_dir = "incorrect/or/invalid/path/".to_string();
        let worker_pool_threads = 1;

        // Act
        let router = build_router(model_dir, worker_pool_threads);

        // Assert
        assert!(router.is_err())
    }
}
