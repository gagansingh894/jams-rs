// use axum::routing::{delete, get, post, put};
// use axum::Router;
// use jams_core::manager::Manager;
// use jams_core::model_store::local::LocalModelStore;
// use jams_serve::service::{
//     add_model, delete_model, get_models, healthcheck, predict, root, update_model,
// };
// use rayon::ThreadPoolBuilder;
// use std::sync::Arc;
// use tower_http::trace::TraceLayer;
//
// pub fn test_router() -> Router {
//     // setup rayon thread pool for cpu intensive task
//     let cpu_pool = ThreadPoolBuilder::new()
//         .num_threads(1)
//         .build()
//         .expect("Failed to build rayon threadpool ❌");
//
//     // setup model store
//     let model_store =
//         LocalModelStore::new("".to_string()).expect("Failed to create model store ❌");
//     let manager = Manager::new(Arc::new(model_store)).expect("Failed to initialize manager ❌");
//
//     // shared state
//     let app_state = jams_serve::server::AppState {
//         manager: Arc::new(manager),
//         cpu_pool,
//     };
//     let shared_state = Arc::new(app_state);
//
//     // API routes
//     let api_routes = Router::new()
//         .route("/models", get(get_models))
//         .route("/models", post(add_model))
//         .route("/models", put(update_model))
//         .route("/models", delete(delete_model))
//         .route("/predict", post(predict));
//
//     // build router
//     Router::new()
//         .route("/", get(root))
//         .route("/healthcheck", get(healthcheck))
//         .nest("/api", api_routes)
//         .with_state(shared_state)
//         .layer(TraceLayer::new_for_http())
// }
