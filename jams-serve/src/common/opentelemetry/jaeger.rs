use opentelemetry::KeyValue;
use opentelemetry_sdk::{Resource, runtime, trace};
use opentelemetry::trace::TraceError;
use opentelemetry_otlp::WithExportConfig;

pub fn init_otlp_trace() -> Result<trace::Tracer, TraceError> {
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint("http://localhost:4317"),
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