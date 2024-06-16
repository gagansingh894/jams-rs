use crate::common::server;
use crate::common::shutdown::shutdown_signal;
use crate::common::state::build_app_state_from_config;
use crate::grpc::service::jams_v1::model_server_server::ModelServerServer;
use crate::grpc::service::{jams_v1, JamsService};
use tonic::codegen::tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;

/// Starts the gRPC server with the provided configuration.
///
/// This function initializes and starts the gRPC server using the provided configuration. It handles
/// the following:
/// - Determines the model directory from the configuration or environment variable.
/// - Sets the port number, log level, and number of CPU worker threads based on the configuration.
/// - Builds the router and starts the server, listening on the specified port.
/// - Logs server status and handles graceful shutdown signals.
///
/// # Arguments
/// - `config`: The configuration for the gRPC server, including the model directory, port number, log level,
///   and the number of CPU worker threads.
///
/// # Panics
/// This function will panic if:
/// - The router fails to build.
/// - The TCP listener fails to bind to the specified address.
/// - The server fails to start.
///
/// # Environment Variables
/// - `MODEL_STORE_DIR`: If the `model_dir` is not provided in the configuration, this environment variable
///   is used to determine the model directory.
///
/// # Logging
/// - Logs errors if the `model_dir` is not provided and the `MODEL_STORE_DIR` environment variable is not set.
/// - Log the server status, including the address it's running on and any shutdown signals received.
pub async fn start(config: server::Config) -> anyhow::Result<()> {
    // init port number
    let port = config.port.unwrap_or(3000);

    // set log level
    let use_debug_level = config.use_debug_level.unwrap_or(false);
    let mut log_level = tracing::Level::INFO;
    if use_debug_level {
        log_level = tracing::Level::TRACE
    }

    // initialize tracing
    tracing_subscriber::fmt().with_max_level(log_level).init();

    // setup shared state
    let shared_state = match build_app_state_from_config(config).await {
        Ok(shared_state) => shared_state,
        Err(e) => {
            anyhow::bail!(
                "Failed to build shared state for application ❌: {}",
                e.to_string()
            )
        }
    };

    // create service
    let jams_service = JamsService::new(shared_state).expect("Failed to create J.A.M.S service ❌");

    // add reflection
    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(jams_v1::FILE_DESCRIPTOR_SET)
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

#[cfg(test)]
mod tests {
    use crate::common;
    use crate::grpc::server;

    #[tokio::test]
    async fn successfully_starts_the_server() {
        let config = common::server::Config {
            model_dir: Some("".to_string()),
            port: Some(15000),
            use_debug_level: Some(false),
            num_workers: Some(1),
            with_s3_model_store: Some(false),
            s3_bucket_name: Some("".to_string()),
        };

        // Act
        tokio::spawn(async move { server::start(config).await.unwrap() });

        // The test will fail if the server fails to start
    }

    #[tokio::test]
    async fn server_fails_to_start_due_to_zero_workers_in_worker_pool() {
        let config = common::server::Config {
            model_dir: Some("".to_string()),
            port: Some(15000),
            use_debug_level: Some(false),
            num_workers: Some(0),
            with_s3_model_store: Some(false),
            s3_bucket_name: Some("".to_string()),
        };

        // Act
        let server = server::start(config).await;

        // Assert
        assert!(server.is_err())
    }
}
