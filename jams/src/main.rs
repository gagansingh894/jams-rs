use crate::cli::{
    parse_server_config_from_args, predict, Commands, PredictSubCommands, StartSubCommands,
};
use clap::Parser;
use jams_serve;
use jams_serve::common::server::{GRPC, HTTP};
use std::fs;

mod cli;

#[cfg(not(tarpaulin_include))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();

    match cli.cmd {
        Commands::Start(subcommands) => {
            match subcommands.file {
                Some(file_path) => {
                    let config = jams_serve::common::server::Config::parse(file_path)?;
                    jams_serve::start(config).await;
                    // shutdown signal received
                    tracing::error!("Shutdown signal received ⚠️");
                    Ok(())
                }
                None => {
                    match subcommands.cmd {
                        None => {
                            anyhow::bail!(
                        "Either pass path to config file using -f or use http/grpc subcommands "
                    );
                        }
                        Some(StartSubCommands::Http(args)) => {
                            let config = parse_server_config_from_args(args, HTTP);

                            jams_serve::start(config).await;

                            // shutdown signal received
                            tracing::error!("Shutdown signal received ⚠️");
                            Ok(())
                        }
                        Some(StartSubCommands::Grpc(args)) => {
                            let config = parse_server_config_from_args(args, GRPC);

                            jams_serve::start(config).await;

                            // shutdown signal received
                            tracing::error!("Shutdown signal received ⚠️");
                            Ok(())
                        }
                    }
                }
            }
        }
        Commands::Predict(subcommands) => match subcommands.cmd {
            PredictSubCommands::Tensorflow(args) => {
                match args.model_path {
                    None => {
                        anyhow::bail!("Model path not specified ❌")
                    }
                    Some(path) => {
                        match jams_core::model::tensorflow::Tensorflow::load(path.as_str()) {
                            Ok(model) => match predict(model, args.input, args.input_path) {
                                Ok(predictions) => {
                                    log::info!("✅ {:?} \n", predictions);
                                }
                                Err(e) => {
                                    anyhow::bail!("Failed to make predictions ❌.\n {}", e)
                                }
                            },
                            Err(e) => {
                                anyhow::bail!("Failed to load the model ❌.\n {}", e)
                            }
                        }
                    }
                };
                Ok(())
            }
            PredictSubCommands::Torch(args) => {
                match args.model_path {
                    None => {
                        anyhow::bail!("Model path not specified ❌")
                    }
                    Some(path) => match jams_core::model::torch::Torch::load(path.as_str()) {
                        Ok(model) => match predict(model, args.input, args.input_path) {
                            Ok(predictions) => {
                                log::info!("✅ {:?} \n", predictions);
                            }
                            Err(e) => {
                                anyhow::bail!("Failed to make predictions ❌.\n {}", e)
                            }
                        },
                        Err(e) => {
                            anyhow::bail!("Failed to load the model ❌.\n {}", e)
                        }
                    },
                };
                Ok(())
            }
            PredictSubCommands::Catboost(args) => {
                match args.model_path {
                    None => {
                        anyhow::bail!("Model path not specified ❌")
                    }
                    Some(path) => match jams_core::model::catboost::Catboost::load(path.as_str()) {
                        Ok(model) => match predict(model, args.input, args.input_path) {
                            Ok(predictions) => {
                                log::info!("✅ {:?} \n", predictions);
                            }
                            Err(e) => {
                                anyhow::bail!("Failed to make predictions ❌.\n {}", e)
                            }
                        },
                        Err(e) => {
                            anyhow::bail!("Failed to load the model ❌.\n {}", e)
                        }
                    },
                };
                Ok(())
            }
            PredictSubCommands::Lightgbm(args) => {
                match args.model_path {
                    None => {
                        anyhow::bail!("Model path not specified ❌")
                    }
                    Some(path) => match jams_core::model::lightgbm::LightGBM::load(path.as_str()) {
                        Ok(model) => match predict(model, args.input, args.input_path) {
                            Ok(predictions) => {
                                log::info!("✅ {:?} \n", predictions);
                            }
                            Err(e) => {
                                anyhow::bail!("Failed to make predictions ❌.\n {}", e)
                            }
                        },
                        Err(e) => {
                            anyhow::bail!("Failed to load the model ❌.\n {}", e)
                        }
                    },
                };
                Ok(())
            }
        },
    }
}
