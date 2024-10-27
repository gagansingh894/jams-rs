use crate::common::shutdown::shutdown_signal;
use crate::common::state::AppState;
use crate::http::router::build_router;
use std::sync::Arc;

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
pub async fn start(shared_state: Arc<AppState>, port: u16) -> anyhow::Result<()> {
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
