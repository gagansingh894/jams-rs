use serde::Deserialize;
use std::fs;

// Terminal art
pub const ART: &str = r#"
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

pub type Protocol = &'static str;
type ModelStore = &'static str;

pub const GRPC: Protocol = "grpc";
pub const HTTP: Protocol = "http";
pub const LOCAL: ModelStore = "local";
pub const AZURE: ModelStore = "azure";
pub const AWS: ModelStore = "aws";
pub const MINIO: ModelStore = "minio";

/// Configuration for the J.A.M.S.
///
/// This common struct holds various configuration options for the HTTP/gRPC server, including the model directory,
/// port number, log level, and the number of worker threads for CPU-intensive tasks.
/// Check https://github.com/gagansingh894/jams-rs/tree/main/build/run_config for examples
#[derive(Deserialize, Clone)]
pub struct Config {
    /// Protocol to use for serving = `http` ot `grpc`
    pub protocol: String,

    /// Port number for the HTTP/gRPC server.
    ///
    /// This is an optional field.
    /// If not provided, the default port number is 3000 for HTTP and 4000 for gRPC
    pub port: Option<u16>,

    /// Model store to use.
    /// The valid options are
    /// - `local` - filesystem.Must pass model dir
    /// - `aws` - aws s3
    /// - `azure` - azure blob storage
    /// - `minio` - minio
    pub model_store: String,

    /// Path to the directory containing models.
    ///
    /// This is an optional field. If not provided, the server may use a default path or handle the absence
    /// of this directory in some other way.
    pub model_dir: Option<String>,

    /// An optional string specifying the name of the S3 bucket to be used for model storage.
    /// Applicable only for aws or minio
    ///
    /// - `Some(String)`: The name of the S3 bucket.
    /// - `None`: No S3 bucket name is specified.
    pub s3_bucket_name: Option<String>,
    /// An optional string specifying the name of the azure storage container to be used for model storage.
    ///
    /// - `Some(String)`: The name of the S3 bucket.
    /// - `None`: No S3 bucket name is specified.
    pub azure_storage_container_name: Option<String>,

    /// Number of threads to be used in the CPU thread pool.
    ///
    /// This thread pool is different from the I/O thread pool and is used for computing CPU-intensive tasks.
    /// This is an optional field. If not provided, the default number of worker threads is 2.
    pub num_workers: Option<usize>,

    /// An optional value representing the interval (in seconds) for polling the model store.
    ///
    /// - `Some(u64)`: The polling interval in seconds.
    /// - `None`: No polling interval is specified, which will disable periodic checks for model updates.
    pub poll_interval: Option<u64>,
}

impl Config {
    pub fn parse(file_path: String) -> anyhow::Result<Config> {
        let contents = match fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(e) => {
                anyhow::bail!("Failed to read config file ❌: {}", e.to_string())
            }
        };

        let config: Config = match toml::from_str(contents.as_str()) {
            Ok(config) => config,
            Err(e) => {
                anyhow::bail!("Failed to parse config file ❌: {}", e.to_string())
            }
        };

        let protocol = config.clone().protocol;
        if (protocol != HTTP) && (protocol != GRPC) {
            anyhow::bail!("Only following protocols are supported: {}, {}", HTTP, GRPC)
        }

        let model_store = config.clone().model_store;
        if (model_store != LOCAL)
            && (model_store != AZURE)
            && (model_store != AWS)
            && (model_store != MINIO)
        {
            anyhow::bail!(
                "Only following model stores are supported: {}, {}, {}, {}",
                LOCAL,
                AZURE,
                AWS,
                MINIO
            )
        }

        Ok(config)
    }
}
