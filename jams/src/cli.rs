use clap::{Args, Parser, Subcommand};
use jams_core::model::predict::Predict;
use jams_serve::common::server::{Config, Protocol};
use std::fs;

/// CLI for starting an J.A.M.S
#[derive(Parser, Debug)]
#[clap(
    name = "J.A.M.S - Just Another Model Server",
    version = "0.1.23",
    author = "Gagandeep Singh",
    about = "J.A.M.S aims to provide a fast, comprehensive and modular serving solution for tree based and deep learning models written in Rust 🦀"
)]
pub struct Cli {
    #[clap(subcommand)]
    pub cmd: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Start the model server with a separate rayon threadpool for computing predictions
    #[clap(name = "start")]
    Start(StartCommands),

    /// Make prediction directly from CLI
    #[clap(name = "predict")]
    Predict(PredictCommands),
}

#[derive(Parser, Debug)]
pub struct StartCommands {
    #[clap(subcommand)]
    pub cmd: Option<StartSubCommands>,

    /// Path to config file
    #[arg(short = 'f', long)]
    pub file: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum StartSubCommands {
    /// Start the HTTP server
    Http(StartCommandArgs),
    /// Start the gRPC server
    Grpc(StartCommandArgs),
}

#[derive(Parser, Debug)]
pub struct PredictCommands {
    #[clap(subcommand)]
    pub cmd: PredictSubCommands,
}

#[derive(Subcommand, Debug)]
pub enum PredictSubCommands {
    /// Make predictions using a Tensorflow model
    Tensorflow(PredictCommandArgs),
    /// Make predictions using a PyTorch model
    Torch(PredictCommandArgs),
    /// Make predictions using a Catboost model
    Catboost(PredictCommandArgs),
    /// Make predictions using a LightGBM model
    Lightgbm(PredictCommandArgs),
}

#[derive(Args, Debug, Clone)]
pub struct StartCommandArgs {
    /// Model Store to use for hosting models - aws, azure, minio, Local
    #[clap(long)]
    pub model_store: String,

    /// Path to the directory containing models. Must contain models, To be specified when model_store = local
    #[clap(long)]
    pub model_dir: Option<String>,

    /// Port number (default: 3000)
    #[clap(long)]
    pub port: Option<u16>,

    /// Number of threads to be used in CPU threadpool. This threadpool is different from the
    /// I/O threadpool and used for computing CPU intensive tasks (default: 2)
    #[clap(long)]
    pub num_workers: Option<usize>,

    /// Name of S3 bucket hosting models. To be specified when model_store = aws|minio
    #[clap(long)]
    pub s3_bucket_name: Option<String>,

    /// Name of Azure Storage container hosting models. To be specified when model_store = azure
    #[clap(long)]
    pub azure_storage_container_name: Option<String>,

    /// Polling interval for model store
    #[clap(long)]
    pub poll_interval: Option<u64>,
}

#[derive(Args, Debug, Clone)]
pub struct PredictCommandArgs {
    /// Path to the model to use for making predictions
    #[clap(long)]
    pub model_path: Option<String>,

    /// Input data in json like string for prediction
    #[clap(long)]
    pub input: Option<String>,

    /// Path to JSON file containing input data
    #[clap(long)]
    pub input_path: Option<String>,
}

pub fn parse_server_config_from_args(args: StartCommandArgs, protocol: Protocol) -> Config {
    Config {
        protocol: protocol.to_string(),
        model_store: args.model_store,
        model_dir: args.model_dir,
        port: args.port,
        num_workers: args.num_workers,
        s3_bucket_name: args.s3_bucket_name,
        azure_storage_container_name: args.azure_storage_container_name,
        poll_interval: args.poll_interval,
    }
}

pub fn predict(
    model: impl Predict,
    input: Option<String>,
    input_path: Option<String>,
) -> anyhow::Result<jams_core::model::output::ModelOutput> {
    Ok(match input {
        None => match input_path {
            None => {
                anyhow::bail!("Either input or path to input should be specified ❌")
            }
            Some(path) => {
                let data = fs::read_to_string(path)?;
                let model_inputs = jams_core::model::input::ModelInput::from_str(data.as_str())?;
                model.predict(model_inputs)?
            }
        },
        Some(input) => {
            let model_inputs = jams_core::model::input::ModelInput::from_str(input.as_str())?;
            model.predict(model_inputs)?
        }
    })
}
