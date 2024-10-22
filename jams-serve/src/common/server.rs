// Terminal art
pub const ART:  &str = r#"
    ___           ________           _____ ______            ________
   |\  \         |\   __  \         |\   _ \  _   \         |\   ____\
   \ \  \        \ \  \|\  \        \ \  \\\__\ \  \        \ \  \___|_
 __ \ \  \        \ \   __  \        \ \  \\|__| \  \        \ \_____  \
|\  \\_\  \  ___   \ \  \ \  \  ___   \ \  \    \ \  \  ___   \|____|\  \
\ \________\|\__\   \ \__\ \__\|\__\   \ \__\    \ \__\|\__\    ____\_\  \
 \|________|\|__|    \|__|\|__|\|__|    \|__|     \|__|\|__|   |\_________\
                                                               \|_________|

J.A.M.S - Just Another Model Server
    "#;

/// Configuration for the J.A.M.S.
///
/// This common struct holds various configuration options for the HTTP/gRPC server, including the model directory,
/// port number, log level, and the number of worker threads for CPU-intensive tasks.
pub struct Config {
    /// Path to the directory containing models.
    ///
    /// This is an optional field. If not provided, the server may use a default path or handle the absence
    /// of this directory in some other way.
    pub model_dir: Option<String>,

    /// Port number for the HTTP/gRPC server.
    ///
    /// This is an optional field.
    /// If not provided, the default port number is 3000 for HTTP and 4000 for gRPC
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

    /// An optional boolean flag indicating whether to use S3 as the model store.
    ///
    /// - `Some(true)`: Use S3 for model storage.
    /// - `Some(false)`: Do not use S3 for model storage.
    /// - `None`: The configuration for using S3 is not specified.
    pub with_s3_model_store: Option<bool>,

    /// An optional string specifying the name of the S3 bucket to be used for model storage.
    ///
    /// - `Some(String)`: The name of the S3 bucket.
    /// - `None`: No S3 bucket name is specified.
    pub s3_bucket_name: Option<String>,

    /// An optional boolean flag indicating whether to use Azure as the model store.
    ///
    /// - `Some(true)`: Use Azure for model storage.
    /// - `Some(false)`: Do not use Azure for model storage.
    /// - `None`: The configuration for using Azure is not specified.
    pub with_azure_model_store: Option<bool>,

    /// An optional string specifying the name of the azure storage container to be used for model storage.
    ///
    /// - `Some(String)`: The name of the S3 bucket.
    /// - `None`: No S3 bucket name is specified.
    pub azure_storage_container_name: Option<String>,

    /// An optional value representing the interval (in seconds) for polling the model store.
    ///
    /// - `Some(u64)`: The polling interval in seconds.
    /// - `None`: No polling interval is specified, which may disable periodic checks for model updates.
    pub poll_interval: Option<u64>,
}
