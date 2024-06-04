mod service;
use crate::service::{healthcheck, predict, root};

use axum::{
    routing::{get, post},
    Router,
};
use jams_core::{manager::Manager, model_store::local::LocalModelStore};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // initialize tracing
    tracing_subscriber::fmt::init();

    // setup model store
    let model_dir = "assets/model_storage/local_model_store".to_string();
    let model_store = LocalModelStore::new(model_dir).unwrap();
    let manager = Manager::new(Arc::new(model_store)).unwrap();

    // shared state
    let shared_state = Arc::new(manager);

    // API routes
    let api_routes = Router::new().route("/predict", post(predict));

    // build server
    let app = Router::new()
        .route("/", get(root))
        .route("/healthcheck", get(healthcheck))
        .nest("/api", api_routes)
        .with_state(shared_state);

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000")
        .await
        .expect("failed to create TCP listener ❌");

    // log that the server is running
    tracing::info!("server is running on http://0.0.0.0:3000 ✅ \n");

    // run on hyper
    axum::serve(listener, app)
        .await
        .expect("failed to start server ❌ \n")
}
