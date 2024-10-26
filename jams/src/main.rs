use crate::cli::{predict, Commands, Data, PredictSubCommands, StartSubCommands};
use clap::Parser;
use std::fs;

mod cli;

const GRPC: &'static str = "grpc";
const HTTP: &'static str = "http";
const LOCAL: &'static str = "local";
const AZURE: &'static str = "azure";
const AWS: &'static str = "aws";
const MINIO: &'static str = "minio";

#[cfg(not(tarpaulin_include))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = cli::Cli::parse();

    match cli.cmd {
        Commands::Start(subcommands) => {
            match subcommands.file {
                Some(file_path) => {
                    let contents = match fs::read_to_string(file_path) {
                        Ok(c) => c,
                        Err(e) => {
                            anyhow::bail!("Failed to read config file ❌: {}", e.to_string())
                        }
                    };

                    let config_data: Data = match toml::from_str(contents.as_str()) {
                        Ok(config_data) => config_data,
                        Err(e) => {
                            anyhow::bail!("Failed to parse config file ❌: {}", e.to_string())
                        }
                    };

                    let model_store = config_data.config.model_store;
                    if (model_store != LOCAL)
                        || (model_store != AZURE)
                        || (model_store != AWS)
                        || (model_store != MINIO)
                    {
                        anyhow::bail!(
                            "Only following model stores are supported: {}, {}, {}, {}",
                            LOCAL,
                            AZURE,
                            AWS,
                            MINIO
                        )
                    }

                    let config = jams_serve::common::server::Config {
                        port: config_data.config.port,
                        use_debug_level: Some(false),
                        num_workers: config_data.config.num_workers,
                        model_store: model_store.clone(),
                        model_dir: config_data.config.model_dir,
                        s3_bucket_name: None
                            .filter(|_| model_store == LOCAL)
                            .or(config_data.config.s3_bucket_name),
                        azure_storage_container_name: None
                            .filter(|_| model_store == LOCAL)
                            .or(config_data.config.azure_storage_container_name),
                        poll_interval: config_data.config.poll_interval,
                    };

                    if config_data.config.protocol == GRPC {
                        jams_serve::grpc::server::start(config)
                            .await
                            .expect("Failed to start server ❌ \n");

                        // shutdown signal received
                        tracing::error!("Shutdown signal received ⚠️");
                        Ok(())
                    } else if config_data.config.protocol == HTTP {
                        jams_serve::http::server::start(config)
                            .await
                            .expect("Failed to start server ❌ \n");

                        // shutdown signal received
                        tracing::error!("Shutdown signal received ⚠️");
                        Ok(())
                    } else {
                        anyhow::bail!("Unrecognised protocol ❌")
                    }
                }
                None => {
                    match subcommands.cmd {
                        None => {
                            anyhow::bail!(
                        "Either pass path to config file using -f or use http/grpc subcommands "
                    );
                        }
                        Some(StartSubCommands::Http(args)) => {
                            let config = jams_serve::common::server::Config {
                                model_store: args.model_store,
                                model_dir: args.model_dir,
                                port: args.port,
                                use_debug_level: args.use_debug_level,
                                num_workers: args.num_workers,
                                s3_bucket_name: args.s3_bucket_name,
                                azure_storage_container_name: args.azure_storage_container_name,
                                poll_interval: args.poll_interval,
                            };

                            jams_serve::http::server::start(config)
                                .await
                                .expect("Failed to start server ❌ \n");

                            // shutdown signal received
                            tracing::error!("Shutdown signal received ⚠️");
                            Ok(())
                        }
                        Some(StartSubCommands::Grpc(args)) => {
                            let config = jams_serve::common::server::Config {
                                model_store: args.model_store,
                                model_dir: args.model_dir,
                                port: args.port,
                                use_debug_level: args.use_debug_level,
                                num_workers: args.num_workers,
                                s3_bucket_name: args.s3_bucket_name,
                                azure_storage_container_name: args.azure_storage_container_name,
                                poll_interval: args.poll_interval,
                            };

                            jams_serve::grpc::server::start(config)
                                .await
                                .expect("Failed to start server ❌ \n");

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
