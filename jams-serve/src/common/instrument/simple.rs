pub fn init(log_level: tracing::Level) {
    tracing_subscriber::fmt()
        .with_line_number(true)
        .with_max_level(log_level)
        .pretty()
        .init();
}
