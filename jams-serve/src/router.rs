use crate::service::{healthcheck, predict, root};
use axum::routing::{get, post};
use axum::Router;
use jams_core::manager::Manager;
use jams_core::model_store::local::LocalModelStore;
use std::sync::Arc;

pub fn build_router(model_dir: String) -> anyhow::Result<Router> {
    // initialize tracing
    tracing_subscriber::fmt::init();

    // setup model store
    let model_store = LocalModelStore::new(model_dir).expect("failed to create model store");
    let manager = Manager::new(Arc::new(model_store)).expect("failed to initialize manage");

    // shared state
    let shared_state = Arc::new(manager);

    // API routes
    let api_routes = Router::new().route("/predict", post(predict));

    // build router
    Ok(Router::new()
        .route("/", get(root))
        .route("/healthcheck", get(healthcheck))
        .nest("/api", api_routes)
        .with_state(shared_state))
}
