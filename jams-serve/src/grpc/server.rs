use crate::common::shutdown::shutdown_signal;
use crate::common::state::AppState;
use crate::grpc::service::JamsService;
use jams_proto::jams_v1::model_server_server::ModelServerServer;
use jams_proto::jams_v1::FILE_DESCRIPTOR_SET;
use std::sync::Arc;
use tonic::codegen::tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

/// Starts the gRPC server with the provided configuration.
///
/// This function initializes the necessary components and starts the server,
/// including setting up logging, building the shared application state, and
/// configuring the gRPC services.
///
/// # Arguments
///
/// * `config` - The server configuration.
///
/// # Returns
///
/// * `Result<()>` - An empty result indicating success or an error.
///
/// # Errors
///
/// This function will return an error if:
/// * The shared state cannot be built.
/// * The TCP listener cannot be created.
/// * Any failure occurs during the initialization of the services or the server.
///
pub async fn start(shared_state: Arc<AppState>, port: u16) -> anyhow::Result<()> {
    // create service
    let jams_service = JamsService::new(shared_state).expect("Failed to create J.A.M.S service ❌");

    // add reflection
    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .build()
        .unwrap();

    // run our app with hyper, listening globally on specified port
    let address = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .expect("Failed to create TCP listener ❌");

    // log that the server is running
    tracing::info!(
        "{}",
        format!("Server is running on http://0.0.0.0:{} 🚀 \n", port)
    );

    Server::builder()
        .add_service(reflection_service)
        .add_service(ModelServerServer::new(jams_service))
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), shutdown_signal())
        .await?;

    Ok(())
}
