use clap::{Args, Parser, Subcommand};
use jams_core::model::predictor::Predictor;
use serde::Deserialize;
use std::fs;

/// CLI for starting an J.A.M.S
#[derive(Parser, Debug)]
#[clap(
    name = "J.A.M.S - Just Another Model Server",
    version = "0.1.20",
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
    /// Path to the directory containing models
    #[clap(long)]
    pub model_dir: Option<String>,

    /// Port number (default: 3000)
    #[clap(long)]
    pub port: Option<u16>,

    /// Toggle DEBUG logs on/off
    #[clap(long)]
    pub use_debug_level: Option<bool>,

    /// Number of threads to be used in CPU threadpool. This threadpool is different from the
    /// I/O threadpool and used for computing CPU intensive tasks (default: 2)
    #[clap(long)]
    pub num_workers: Option<usize>,

    /// Start with S3 model store rather than local model store.
    #[clap(long)]
    pub with_s3_model_store: Option<bool>,

    /// Name of S3 bucket hosting models
    #[clap(long)]
    pub s3_bucket_name: Option<String>,

    /// Start with Azure model store rather than local model store.
    #[clap(long)]
    pub with_azure_model_store: Option<bool>,

    /// Name of Azure Storage container hosting models
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

// Top level struct to hold the TOML data.
#[derive(Deserialize)]
pub struct Data {
    pub config: Config,
}

// Config struct holds to data from the `[config]` section.
#[derive(Deserialize)]
pub struct Config {
    pub protocol: String,
    pub port: Option<u16>,
    pub model_store: String,
    pub num_workers: Option<usize>,
    pub poll_interval: Option<u64>,
    pub model_dir: Option<String>,
    pub azure_storage_container_name: Option<String>,
    pub s3_bucket_name: Option<String>,
}

pub fn predict(
    model: impl Predictor,
    input: Option<String>,
    input_path: Option<String>,
) -> anyhow::Result<jams_core::model::predictor::Output> {
    Ok(match input {
        None => match input_path {
            None => {
                anyhow::bail!("Either input or path to input should be specified ❌")
            }
            Some(path) => {
                let data = fs::read_to_string(path)?;
                let model_inputs =
                    jams_core::model::predictor::ModelInput::from_str(data.as_str())?;
                model.predict(model_inputs)?
            }
        },
        Some(input) => {
            let model_inputs = jams_core::model::predictor::ModelInput::from_str(input.as_str())?;
            model.predict(model_inputs)?
        }
    })
}
