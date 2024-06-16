use crate::common::server;
use crate::common::shutdown::shutdown_signal;
use crate::common::state::build_app_state_from_config;
use crate::http::router::build_router;

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

    let app = match build_router(shared_state) {
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

#[cfg(test)]
mod tests {
    use crate::common;
    use crate::http::server;

    #[tokio::test]
    async fn successfully_starts_the_server() {
        // Arrange
        let config = common::server::Config {
            model_dir: Some("".to_string()),
            port: Some(5000),
            use_debug_level: Some(false),
            num_workers: Some(1),
            with_s3_model_store: Some(false),
            s3_bucket_name: Some("".to_string()),
        };

        // Act
        tokio::spawn(async move {
            server::start(config).await.unwrap();
        });

        // The test will fail if the server fails to start
    }

    #[tokio::test]
    async fn server_fails_to_start_due_to_zero_workers_in_worker_pool() {
        // Arrange
        let config = common::server::Config {
            model_dir: Some("".to_string()),
            port: Some(5000),
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
