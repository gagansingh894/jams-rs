mod server;
mod service;

use crate::server::{build_router, shutdown_signal};
use std::env;

/// Configuration for the HTTP server.
///
/// This struct holds various configuration options for the HTTP server, including the model directory,
/// port number, log level, and the number of worker threads for CPU-intensive tasks.
pub struct HTTPConfig {
    /// Path to the directory containing models.
    ///
    /// This is an optional field. If not provided, the server may use a default path or handle the absence
    /// of this directory in some other way.
    pub model_dir: Option<String>,

    /// Port number for the HTTP server.
    ///
    /// This is an optional field. If not provided, the default port number is 3000.
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
}

/// Starts the HTTP server with the provided configuration.
///
/// This function initializes and starts the HTTP server using the provided configuration. It handles
/// the following:
/// - Determines the model directory from the configuration or environment variable.
/// - Sets the port number, log level, and number of CPU worker threads based on the configuration.
/// - Builds the router and starts the server, listening on the specified port.
/// - Logs server status and handles graceful shutdown signals.
///
/// # Arguments
/// - `config`: The configuration for the HTTP server, including the model directory, port number, log level,
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
pub async fn start_server(config: HTTPConfig) -> anyhow::Result<()> {
    let model_dir = match config.model_dir {
        Some(dir) => dir,
        None => {
            // search for environment variable
            match env::var("MODEL_STORE_DIR") {
                Ok(model_dir) => model_dir,
                Err(_) => {
                    anyhow::bail!("Either set MODEL_STORE_DIR or use 'model-dir' argument ❌")
                }
            }
        }
    };

    let port = config.port.unwrap_or(3000);
    let use_debug_level = config.use_debug_level.unwrap_or(false);
    let num_workers = config.num_workers.unwrap_or(2);

    let app = match build_router(model_dir, use_debug_level, num_workers) {
        Ok(app) => app,
        Err(_) => {
            anyhow::bail!("Failed to build the router ❌");
        }
    };

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

    // run on hyper
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}
