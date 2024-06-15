use crate::common::shutdown::shutdown_signal;
use crate::grpc::service::jams_v1::model_server_server::ModelServerServer;
use crate::grpc::service::{jams_v1, JamsService};
use std::env;
use std::sync::Arc;
use rayon::ThreadPoolBuilder;
use tonic::codegen::tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use jams_core::manager::Manager;
use jams_core::model_store::local::LocalModelStore;
use jams_core::model_store::s3::S3ModelStore;
use crate::common::state::AppState;

/// Configuration for the gRPC server.
///
/// This struct holds various configuration options for the gRPC server, including the model directory,
/// port number, log level, and the number of worker threads for CPU-intensive tasks.
pub struct GRPCConfig {
    /// Path to the directory containing models.
    ///
    /// This is an optional field. If not provided, the server may use a default path or handle the absence
    /// of this directory in some other way.
    pub model_dir: Option<String>,

    /// Port number for the gRPC server.
    ///
    /// This is an optional field. If not provided, the default port number is 4000.
    pub port: Option<u16>,

    /// Toggle DEBUG logs on or off.
    ///
    /// This is an optional field. If set to `Some(true)`, DEBUG logs will be enabled. If `None` or `Some(false)`,
    /// DEBUG logs will be disabled.
    pub use_debug_level: Option<bool>,

    /// Number of threads to be used in the CPU thread pool.
    ///
    /// This thread pool is different from the I/O thread pool and is used for computing CPU-intensive tasks.
    /// This is an optional field. If not provided, the default number of worker threads is 2.
    pub num_workers: Option<usize>,

    /// An optional boolean flag indicating whether to use S3 as the model store.
    ///
    /// - `Some(true)`: Use S3 for model storage.
    /// - `Some(false)`: Do not use S3 for model storage.
    /// - `None`: The configuration for using S3 is not specified.
    pub with_s3_model_store: Option<bool>,

    /// An optional string specifying the name of the S3 bucket to be used for model storage.
    ///
    /// - `Some(String)`: The name of the S3 bucket.
    /// - `None`: No S3 bucket name is specified.
    pub s3_bucket_name: Option<String>,
}

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
pub async fn start(config: GRPCConfig) -> anyhow::Result<()> {
    let model_dir = config.model_dir.unwrap_or_else(|| {
        // search for environment variable
        env::var("MODEL_STORE_DIR").unwrap_or_else(|_| "".to_string())
    });

    let port = config.port.unwrap_or(4000);
    let use_debug_level = config.use_debug_level.unwrap_or(false);
    let worker_pool_threads = config.num_workers.unwrap_or(2);
    let with_s3_model_store = config.with_s3_model_store.unwrap_or(false);

    // set log level
    let mut log_level = tracing::Level::INFO;
    if use_debug_level {
        log_level = tracing::Level::TRACE
    }

    // initialize tracing
    tracing_subscriber::fmt().with_max_level(log_level).init();

    // initialize threadpool for cpu intensive tasks
    if worker_pool_threads < 1 {
        anyhow::bail!("At least 1 worker is required for rayon threadpool")
    }
    let cpu_pool = ThreadPoolBuilder::new()
        .num_threads(worker_pool_threads)
        .build()
        .expect("Failed to build rayon threadpool ❌");

    // initialize manager with madel store
    let manager = match with_s3_model_store {
        true => {
            let model_store = S3ModelStore::new(config.s3_bucket_name.unwrap())
                .await
                .expect("Failed to create model store ❌");
            Arc::new(Manager::new(Arc::new(model_store)).expect("Failed to initialize manager ❌"))
        }
        false => {
            let model_store = LocalModelStore::new(model_dir)
                .expect("Failed to create model store ❌");
            Arc::new(Manager::new(Arc::new(model_store)).expect("Failed to initialize manager ❌"))
        }
    };

    // setup app state
    let app_state = Arc::new(AppState {manager, cpu_pool});


    // create service
    let jams_service =
        JamsService::new(app_state).expect("Failed to create J.A.M.S service ❌");

    tracing::info!("Rayon threadpool started with {} workers ⚙️", worker_pool_threads);

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
    use crate::grpc::server;

    #[tokio::test]
    async fn successfully_starts_the_server() {
        let config = server::GRPCConfig {
            model_dir: Some("".to_string()),
            port: Some(15000),
            use_debug_level: Some(false),
            num_workers: Some(1),
            with_s3_model_store: Some(false),
            s3_bucket_name: Some("".to_string())
        };

        // Act
        tokio::spawn(async move { server::start(config).await.unwrap() });

        // The test will fail if the server fails to start
    }

    #[tokio::test]
    async fn server_fails_to_start_due_to_zero_workers_in_worker_pool() {
        let config = server::GRPCConfig {
            model_dir: Some("".to_string()),
            port: Some(15000),
            use_debug_level: Some(false),
            num_workers: Some(0),
            with_s3_model_store: Some(false),
            s3_bucket_name: Some("".to_string())
        };

        // Act
        let server = server::start(config).await;

        // Assert
        assert!(server.is_err())
    }
}
