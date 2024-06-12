use axum::Router;
use jams_serve::server::build_router;

pub fn test_router() -> Router {
    // we will not set a model for testing purpose
    // this will start the model server without any models loaded
    let model_dir = "".to_string();
    let worker_pool_threads = 1;

    build_router(model_dir, worker_pool_threads).unwrap()
}
