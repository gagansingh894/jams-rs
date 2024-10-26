use std::env;
use opentelemetry::{global, KeyValue};
use opentelemetry::trace::TraceError;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::{Resource, runtime, trace};
use tracing_subscriber::layer::SubscriberExt;

pub fn init() {
    global::set_text_map_propagator(TraceContextPropagator::new());
    // Create a formatted subscriber and combine it with the OpenTelemetry layer
    let subscriber = tracing_subscriber::fmt()
        .with_line_number(true)
        .with_max_level(tracing::Level::INFO)
        .pretty()
        .finish();

    let tracer = init_otlp_trace().expect("Failed to create OTLP tracer provider ❌");
    // Create the OpenTelemetry tracing layer
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
    
    // Set this combined subscriber as the global default
    tracing::subscriber::set_global_default(subscriber.with(telemetry)).unwrap();
}


fn init_otlp_trace() -> Result<trace::Tracer, TraceError> {
    let endpoint = env::var("OTLP_EXPORTER_URL").unwrap_or("http://localhost:4317".to_string());
    
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint(endpoint),
        )
        .with_trace_config(
            trace::Config::default().with_resource(Resource::new(vec![KeyValue::new(
                opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                "jams",
            )])),
        )
        .install_batch(runtime::Tokio)
        .expect("Failed to install OpenTelemetry tracer.");

    Ok(tracer)
}