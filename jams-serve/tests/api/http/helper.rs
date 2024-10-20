use axum::Router;
use jams_core::manager::Manager;
use jams_core::model_store::local::LocalModelStore;
use jams_serve::common::state::AppState;
use jams_serve::http::router::build_router;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;

async fn setup_shared_state() -> Arc<AppState> {
    let cpu_pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("Failed to build rayon threadpool ❌");

    let model_store = LocalModelStore::new("tests/model_store".to_string())
        .await
        .expect("Failed to create model store ❌");

    let manager = Arc::new(
        Manager::new(Arc::new(model_store))
            .await
            .expect("Failed to initialize manager ❌"),
    );

    Arc::new(AppState { manager, cpu_pool })
}
pub async fn test_router() -> Router {
    // we will not set a model for testing purpose
    // this will start the model server without any models loaded
    let shared_state = setup_shared_state().await;

    build_router(shared_state).unwrap()
}
