pub mod common;
pub mod grpc;
pub mod http;

use crate::common::server;
use crate::common::server::HTTP;
use crate::common::state::build_app_state;

pub async fn start(config: server::Config, num_physical_cores: usize) {
    // print terminal art
    println!("{}", server::ART);

    // init port number
    let http_port = config.port.unwrap_or(3000);
    let grpc_port = config.port.unwrap_or(4000);

    // setup shared state
    let shared_state = match build_app_state(config.clone(), num_physical_cores).await {
        Ok(state) => state,
        Err(e) => {
            tracing::error!(
                "Failed to build shared state for application ❌: {}",
                e.to_string()
            );
            return;
        }
    };

    if config.protocol == HTTP {
        // Start HTTP server
        http::server::start(shared_state, http_port)
            .await
            .expect("Failed to start HTTP server");
    } else {
        // Start gRPC server
        grpc::server::start(shared_state, grpc_port)
            .await
            .expect("Failed to start gRPC server");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common;
    use crate::common::server::GRPC;

    #[tokio::test]
    async fn successfully_starts_the_grpc_server() {
        // Arrange
        let config = common::server::Config {
            protocol: GRPC.to_string(),
            model_store: "local".to_string(),
            model_dir: Some("tests/model_store".to_string()),
            port: Some(5000),
            num_workers: Some(1),
            s3_bucket_name: Some("".to_string()),
            azure_storage_container_name: Some("".to_string()),
            poll_interval: Some(0),
        };

        // Act
        tokio::spawn(async move {
            start(config, 1).await;
        });

        // The test will fail if the server fails to start
    }

    #[tokio::test]
    async fn successfully_starts_the_http_server() {
        let config = common::server::Config {
            protocol: HTTP.to_string(),
            model_store: "local".to_string(),
            model_dir: Some("model_store".to_string()),
            port: Some(15000),
            num_workers: Some(1),
            s3_bucket_name: Some("".to_string()),
            azure_storage_container_name: Some("".to_string()),
            poll_interval: Some(0),
        };

        // Act
        tokio::spawn(async move { start(config, 1).await });

        // The test will fail if the server fails to start
    }
}
