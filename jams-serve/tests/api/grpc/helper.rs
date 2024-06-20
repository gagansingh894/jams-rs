use jams_core::manager::Manager;
use jams_core::model_store::local::LocalModelStore;
use jams_serve::common::state::AppState;
use jams_serve::grpc::service::jams_v1::model_server_client::ModelServerClient;
use jams_serve::grpc::service::jams_v1::model_server_server::ModelServerServer;
use jams_serve::grpc::service::JamsService;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use std::time::Duration;
use tonic::transport::server::Router;
use tonic::transport::{Channel, Server};

async fn setup_shared_state() -> Arc<AppState> {
    let cpu_pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("Failed to build rayon threadpool ❌");

    let model_store = LocalModelStore::new("tests/model_store".to_string())
        .await
        .expect("Failed to create model store ❌");

    let manager =
        Arc::new(Manager::new(Arc::new(model_store)).expect("Failed to initialize manager ❌"));

    Arc::new(AppState { manager, cpu_pool })
}

pub async fn jams_grpc_test_router() -> Router {
    // we will not set a model for testing purpose
    // this will start the model server without any models loaded
    let shared_state = setup_shared_state().await;

    let jams_service = JamsService::new(shared_state).unwrap();

    Server::builder().add_service(ModelServerServer::new(jams_service))
}

pub async fn grpc_client_stub(addr: String) -> ModelServerClient<Channel> {
    let channel = Channel::builder(format!("http://{}", addr).parse().unwrap())
        .timeout(Duration::from_secs(2))
        .connect()
        .await
        .unwrap();

    ModelServerClient::new(channel)
}
