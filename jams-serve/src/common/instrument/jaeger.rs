use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Tracer;
use opentelemetry_sdk::{runtime, trace, Resource};
use std::env;
use tracing_subscriber::layer::SubscriberExt;

pub fn init(service_name: String, log_level: tracing::Level) -> anyhow::Result<()> {
    global::set_text_map_propagator(TraceContextPropagator::new());
    let tracer = init_otlp_trace(service_name)?;
    // Create the OpenTelemetry tracing layer
    let telemetry = tracing_opentelemetry::layer().with_tracer(tracer);
    // Create a formatted subscriber and combine it with the OpenTelemetry layer
    let subscriber = tracing_subscriber::fmt()
        .with_line_number(true)
        .with_max_level(log_level)
        .pretty()
        .finish()
        .with(telemetry);

    // Set this combined subscriber as the global default
    match tracing::subscriber::set_global_default(subscriber) {
        Ok(_) => Ok(()),
        Err(_) => {
            anyhow::bail!("Failed to set_global_default tracing subscriber ")
        }
    }
}

fn init_otlp_trace(service_name: String) -> anyhow::Result<Tracer> {
    let endpoint = env::var("OTLP_EXPORTER_URL").unwrap_or("http://localhost:4317".to_string());

    match opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint(endpoint),
        )
        .with_trace_config(trace::Config::default().with_resource(Resource::new(vec![
            KeyValue::new(
                opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                service_name,
            ),
        ])))
        .install_batch(runtime::Tokio)
    {
        Ok(tracer) => Ok(tracer),
        Err(_) => {
            anyhow::bail!("Failed to create otlp tracer")
        }
    }
}
