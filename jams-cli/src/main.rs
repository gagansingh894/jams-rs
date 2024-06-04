use clap::{Args, Parser, Subcommand};
use jams_core::model::predictor::Predictor;
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Commands>,
}

#[derive(Args, Debug, Clone)]
struct CommandArgs {
    #[arg(long)]
    #[doc = "Path to the model file"]
    model_path: Option<String>,
    #[arg(long)]
    #[doc = "Input data for prediction"]
    input: Option<String>,
    #[arg(long)]
    #[doc = "Path to a file containing input data"]
    input_path: Option<String>,
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    Tensorflow(CommandArgs),
    Torch(CommandArgs),
    Catboost(CommandArgs),
    Lightgbm(CommandArgs),
}

#[cfg(not(tarpaulin_include))]
fn main() -> anyhow::Result<()> {
    let args = Cli::parse();
    match args.cmd {
        Some(Commands::Tensorflow(cmd_args)) => {
            explain_flags(&cmd_args);
            // load the tensorflow model
            let model = match cmd_args.model_path {
                None => {
                    anyhow::bail!("model path not specified")
                }
                Some(path) => jams_core::model::tensorflow::Tensorflow::load(path.as_str())
                    .expect("failed to load model"),
            };
            let predictions = predict(model, cmd_args.input, cmd_args.input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
            Ok(())
        }
        Some(Commands::Torch(cmd_args)) => {
            explain_flags(&cmd_args);
            // load the torch model
            let model = match cmd_args.model_path {
                None => {
                    anyhow::bail!("model path not specified")
                }
                Some(path) => jams_core::model::torch::Torch::load(path.as_str())
                    .expect("failed to load model"),
            };
            let predictions = predict(model, cmd_args.input, cmd_args.input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
            Ok(())
        }
        Some(Commands::Catboost(cmd_args)) => {
            explain_flags(&cmd_args);
            // load the catboost model
            let model = match cmd_args.model_path {
                None => {
                    anyhow::bail!("model path not specified")
                }
                Some(path) => jams_core::model::catboost::Catboost::load(path.as_str())
                    .expect("failed to load model"),
            };

            let predictions = predict(model, cmd_args.input, cmd_args.input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
            Ok(())
        }
        Some(Commands::Lightgbm(cmd_args)) => {
            // load the lightGBM model
            let model = match cmd_args.model_path {
                None => {
                    anyhow::bail!("model path not specified")
                }
                Some(path) => jams_core::model::lightgbm::LightGBM::load(path.as_str())
                    .expect("failed to load model"),
            };

            let predictions = predict(model, cmd_args.input, cmd_args.input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
            Ok(())
        }
        None => {
            explain_commands();
            Ok(())
        }
    }
}

#[cfg(not(tarpaulin_include))]
fn predict(
    model: impl Predictor,
    input: Option<String>,
    input_path: Option<String>,
) -> anyhow::Result<jams_core::model::predictor::Output> {
    Ok(match input {
        None => match input_path {
            None => {
                anyhow::bail!("either input or path to input should be specified")
            }
            Some(path) => {
                let data = fs::read_to_string(path).expect("unable to read file");
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

fn explain_flags(args: &CommandArgs) {
    println!("Flags Explanation:");
    if let Some(model_path) = &args.model_path {
        println!("  --model_path: Path to the model file - {}", model_path);
    }
    if let Some(input) = &args.input {
        println!(
            "  --input: Input data in json format for prediction - {}",
            input
        );
    }
    if let Some(input_path) = &args.input_path {
        println!(
            "  --input_path: Path to a json file containing input data - {}",
            input_path
        );
    }
}

fn explain_commands() {
    println!("Available Commands:");
    println!("  tensorflow: Run Tensorflow model predictions");
    println!("  torch: Run Torch model predictions");
    println!("  catboost: Run Catboost model predictions");
    println!("  lightgbm: Run LightGBM model predictions");
    println!("  <command> -h: Display explanations of commands and flags");
}
