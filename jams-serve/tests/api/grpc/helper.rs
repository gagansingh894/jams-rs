use jams_serve::grpc::service::jams_v1::model_server_client::ModelServerClient;
use jams_serve::grpc::service::jams_v1::model_server_server::ModelServerServer;
use jams_serve::grpc::service::JamsService;
use std::time::Duration;
use tonic::transport::server::Router;
use tonic::transport::{Channel, Server};

pub fn jams_grpc_test_router() -> Router {
    // we will not set a model for testing purpose
    // this will start the model server without any models loaded
    let model_dir = "".to_string();
    let worker_pool_threads = 1;

    let jams_service = JamsService::new(model_dir, worker_pool_threads).unwrap();

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
