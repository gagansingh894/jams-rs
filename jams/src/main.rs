use crate::cli::{
    parse_server_config_from_args, predict, Commands, PredictSubCommands, StartSubCommands,
};
use clap::Parser;
use jams_serve::common::server::{GRPC, HTTP};
use tokio::runtime::Builder;

mod cli;

#[cfg(not(tarpaulin_include))]
fn main() -> anyhow::Result<()> {
    // get physical cpu count and divide by 2.
    // one half is given to tokio runtime and the another to rayon thread pool.
    // the rayon threadpool worker count can also be set via config/flag
    let physical_cores = (num_cpus::get_physical() / 2).max(1);

    let tokio_runtime = Builder::new_multi_thread()
        .worker_threads(physical_cores)
        .max_blocking_threads(50)
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime");

    let cli = cli::Cli::parse();

    match cli.cmd {
        Commands::Start(subcommands) => {
            match subcommands.file {
                Some(file_path) => {
                    let config = jams_serve::common::server::Config::parse(file_path)?;

                    tokio_runtime.block_on(async {
                        jams_serve::start(config, physical_cores).await;
                        // shutdown signal received
                        tracing::error!("Shutdown signal received ⚠️");
                    });

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

                            tokio_runtime.block_on(async {
                                jams_serve::start(config, physical_cores).await;

                                // shutdown signal received
                                tracing::error!("Shutdown signal received ⚠️");
                            });

                            Ok(())
                        }
                        Some(StartSubCommands::Grpc(args)) => {
                            let config = parse_server_config_from_args(args, GRPC);

                            tokio_runtime.block_on(async {
                                jams_serve::start(config, physical_cores).await;

                                // shutdown signal received
                                tracing::error!("Shutdown signal received ⚠️");
                            });

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
