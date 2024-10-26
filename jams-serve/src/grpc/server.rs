use crate::common::server;
use crate::common::shutdown::shutdown_signal;
use crate::common::state::build_app_state_from_config;
use crate::common::opentelemetry::jaeger::init_otlp_trace;
use crate::grpc::service::JamsService;
use jams_proto::jams_v1::model_server_server::ModelServerServer;
use jams_proto::jams_v1::FILE_DESCRIPTOR_SET;
use tonic::codegen::tokio_stream::wrappers::TcpListenerStream;
use tonic::transport::Server;
use opentelemetry::global;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use tower_http::trace::TraceLayer;
use tracing_subscriber::layer::SubscriberExt;

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
pub async fn start(config: server::Config) -> anyhow::Result<()> {
    // print terminal art
    println!("{}", server::ART);

    // init port number
    let port = config.port.unwrap_or(4000);

    // set log level
    let use_debug_level = config.use_debug_level.unwrap_or(false);
    let mut _log_level = tracing::Level::INFO;
    if use_debug_level {
        _log_level = tracing::Level::TRACE
    }

    global::set_text_map_propagator(TraceContextPropagator::new());
    let tracer = init_otlp_trace().expect("Failed to create OTLP tracer provider ❌");
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
    let subscriber = tracing_subscriber::Registry::default().with(telemetry);
    tracing::subscriber::set_global_default(subscriber).unwrap();
    

    // // initialize tracing - by default log to fmt
    // tracing_subscriber::fmt()
    //     .with_line_number(true)
    //     .with_max_level(log_level)
    //     .pretty()
    //     .init();


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
        .layer(TraceLayer::new_for_grpc())
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
            model_store: "local".to_string(),
            model_dir: Some("model_store".to_string()),
            port: Some(15000),
            use_debug_level: Some(false),
            num_workers: Some(1),
            s3_bucket_name: Some("".to_string()),
            azure_storage_container_name: Some("".to_string()),
            poll_interval: Some(0),
        };

        // Act
        tokio::spawn(async move { server::start(config).await.unwrap() });

        // The test will fail if the server fails to start
    }

    #[tokio::test]
    async fn server_fails_to_start_due_to_zero_workers_in_worker_pool() {
        let config = common::server::Config {
            model_store: "local".to_string(),
            model_dir: Some("".to_string()),
            port: Some(15000),
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
