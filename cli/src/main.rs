use std::fs;
use clap::{Parser, Subcommand};
use core::model::predictor::Predictor;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    cmd: Commands
}

#[derive(Subcommand, Debug, Clone)]
enum Commands {
    Tensorflow {
        model_path: String,
        input: Option<String>,
        input_path: Option<String>
    },
    Torch {
        model_path: String,
        input: Option<String>,
        input_path: Option<String>
    },
    Catboost {
        model_path: String,
        input: Option<String>,
        input_path: Option<String>
    },
    LightGBM {
        model_path: String,
        input: Option<String>,
        input_path: Option<String>
    }
}

fn main()  {
    let args = Args::parse();
    match args.cmd {
        Commands::Tensorflow {model_path, input, input_path } => {
            // load the tensorflow model
            let model = core::model::tensorflow::Tensorflow::load(model_path.as_str())
                .expect("failed to load model");
            let predictions =predict(model, input, input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
        }
        Commands::Torch {model_path, input, input_path } => {
            let model = core::model::torch::Torch::load(model_path.as_str())
                .expect("failed to load model");
            let predictions =predict(model, input, input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
        }
        Commands::Catboost {model_path, input, input_path } => {
            let model = core::model::catboost::Catboost::load(model_path.as_str())
                .expect("failed to load model");
            let predictions =predict(model, input, input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
        }
        Commands::LightGBM {model_path, input, input_path } => {
            let model = core::model::lightgbm::LightGBM::load(model_path.as_str())
                .expect("failed to load model");
            let predictions =predict(model, input, input_path)
                .expect("failed to make predictions");
            println!("{:?}", predictions);
        }
    }
}

fn predict(model: impl Predictor, input: Option<String>, input_path: Option<String>) -> anyhow::Result<core::model::predictor::Output> {
    Ok(match input {
        None => {
            match input_path {
                None => {
                    anyhow::bail!("either input or path to input should be specified")
                }
                Some(path) => {
                    let data = fs::read_to_string(path)
                        .expect("unable to read file");
                    let model_inputs = core::model::predictor::ModelInput::from_str(data.as_str())?;
                    model.predict(model_inputs)?
                }
            }
        }
        Some(input) => {
            let model_inputs = core::model::predictor::ModelInput::from_str(input.as_str())?;
            model.predict(model_inputs)?
        }
    })
}

fn show_commands() {
    println!(r#"COMMANDS:
tensorflow <KEY> <> <> - Gets the value of a given key and displays it. If no key given, retrieves all values and displays them.
set <KEY> <VALUE> - Sets the value of a given key.
    Flags: --is-true
"#);
}