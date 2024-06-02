use clap::{Args, Parser, Subcommand};
use jams_core::model::predictor::Predictor;
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Args, Debug, Clone)]
struct CommandArgs {
    #[arg(long)]
    model_path: Option<String>,
    #[arg(long)]
    input: Option<String>,
    #[arg(long)]
    input_path: Option<String>,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    Tensorflow(CommandArgs),
    Torch(CommandArgs),
    Catboost(CommandArgs),
    LightGBM(CommandArgs),
}

#[cfg(not(tarpaulin_include))]
fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    match args.cmd {
        Commands::Tensorflow(cmd_args) => {
            // load the tensorflow model
            let model = match cmd_args.model_path {
                None => {
                    anyhow::bail!("model path not specified")
                }
                Some(path) => core::model::tensorflow::Tensorflow::load(path.as_str())
                    .expect("failed to load model"),
            };
            let predictions = predict(model, cmd_args.input, cmd_args.input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
            Ok(())
        }
        Commands::Torch(cmd_args) => {
            // load the torch model
            let model = match cmd_args.model_path {
                None => {
                    anyhow::bail!("model path not specified")
                }
                Some(path) => {
                    core::model::torch::Torch::load(path.as_str()).expect("failed to load model")
                }
            };
            let predictions = predict(model, cmd_args.input, cmd_args.input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
            Ok(())
        }
        Commands::Catboost(cmd_args) => {
            // load the catboost model
            let model = match cmd_args.model_path {
                None => {
                    anyhow::bail!("model path not specified")
                }
                Some(path) => core::model::catboost::Catboost::load(path.as_str())
                    .expect("failed to load model"),
            };

            let predictions = predict(model, cmd_args.input, cmd_args.input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
            Ok(())
        }
        Commands::LightGBM(cmd_args) => {
            // load the lightGBM model
            let model = match cmd_args.model_path {
                None => {
                    anyhow::bail!("model path not specified")
                }
                Some(path) => core::model::lightgbm::LightGBM::load(path.as_str())
                    .expect("failed to load model"),
            };

            let predictions = predict(model, cmd_args.input, cmd_args.input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
            Ok(())
        }
    }
}

#[cfg(not(tarpaulin_include))]
fn predict(
    model: impl Predictor,
    input: Option<String>,
    input_path: Option<String>,
) -> anyhow::Result<core::model::predictor::Output> {
    Ok(match input {
        None => match input_path {
            None => {
                anyhow::bail!("either input or path to input should be specified")
            }
            Some(path) => {
                let data = fs::read_to_string(path).expect("unable to read file");
                let model_inputs = core::model::predictor::ModelInput::from_str(data.as_str())?;
                model.predict(model_inputs)?
            }
        },
        Some(input) => {
            let model_inputs = core::model::predictor::ModelInput::from_str(input.as_str())?;
            model.predict(model_inputs)?
        }
    })
}
