use crate::common::state::AppState;
use crate::http::service::{
    add_model, delete_model, get_models, healthcheck, predict, update_model,
};
use axum::routing::{delete, get, post, put};
use axum::Router;
use std::sync::Arc;
use tower_http::trace::TraceLayer;

pub fn build_router(shared_state: Arc<AppState>) -> anyhow::Result<Router> {
    // API routes
    let api_routes = Router::new()
        .route("/models", get(get_models))
        .route("/models", post(add_model))
        .route("/models", put(update_model))
        .route("/models", delete(delete_model))
        .route("/predict", post(predict));

    // build router
    Ok(Router::new()
        .route("/healthcheck", get(healthcheck))
        .nest("/api", api_routes)
        .with_state(shared_state)
        .layer(TraceLayer::new_for_http()))
}

#[cfg(test)]
mod tests {
    use crate::common::state::AppState;
    use crate::http::router::build_router;
    use jams_core::manager::Manager;
    use jams_core::model_store::local::LocalModelStore;
    use rayon::ThreadPoolBuilder;
    use std::sync::Arc;

    fn setup_shared_state() -> Arc<AppState> {
        let cpu_pool = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("Failed to build rayon threadpool ❌");

        let model_store =
            LocalModelStore::new("".to_string()).expect("Failed to create model store ❌");

        let manager =
            Arc::new(Manager::new(Arc::new(model_store)).expect("Failed to initialize manager ❌"));

        Arc::new(AppState { manager, cpu_pool })
    }

    #[test]
    fn successfully_build_router() {
        // Arrange
        let shared_state = setup_shared_state();

        // Act
        let router = build_router(shared_state);

        // Assert
        assert!(router.is_ok())
    }
}
