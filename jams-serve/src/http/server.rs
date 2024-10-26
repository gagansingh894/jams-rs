use crate::common::shutdown::shutdown_signal;
use crate::common::state::build_app_state_from_config;
use crate::common::{instrument, server};
use crate::http::router::build_router;

/// Starts the HTTP server with the provided configuration.
///
/// This function initializes the necessary components and starts the server,
/// including setting up logging, building the shared application state, configuring the router,
/// and running the server with graceful shutdown.
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
/// * The router cannot be built.
/// * The TCP listener cannot be created.
/// * Any failure occurs during the initialization of the services or the server.
pub async fn start(config: server::Config) -> anyhow::Result<()> {
    // print terminal art
    println!("{}", server::ART);

    // init port number
    let port = config.port.unwrap_or(3000);

    // set log level
    let use_debug_level = config.use_debug_level.unwrap_or(false);
    let mut log_level = tracing::Level::INFO;
    if use_debug_level {
        log_level = tracing::Level::TRACE
    }

    // logs to stdout, if service is running it will send traces to jaegar
    instrument::jaeger::init(log_level)?;

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
            model_store: "local".to_string(),
            model_dir: Some("tests/model_store".to_string()),
            port: Some(5000),
            use_debug_level: Some(false),
            num_workers: Some(1),
            s3_bucket_name: Some("".to_string()),
            azure_storage_container_name: Some("".to_string()),
            poll_interval: Some(0),
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
            model_store: "local".to_string(),
            model_dir: Some("".to_string()),
            port: Some(5000),
            use_debug_level: Some(false),
            num_workers: Some(0),
            s3_bucket_name: Some("".to_string()),
            azure_storage_container_name: Some("".to_string()),
            poll_interval: Some(0),
        };

        // Act
        let server = server::start(config).await;

        // Assert
        assert!(server.is_err())
    }
}
