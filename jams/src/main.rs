use crate::cli::{predict, Commands, PredictSubCommands, StartSubCommands};
use clap::Parser;

mod cli;

#[cfg(not(tarpaulin_include))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Terminal art
    let art = r#"
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

    println!("{}", art);

    // Separate logs
    println!("\n-----------------------------------------------------\n");

    let cli = cli::Cli::parse();

    match cli.cmd {
        Commands::Start(subcommands) => {
            match subcommands.cmd {
                StartSubCommands::Http(args) => {
                    let config = jams_serve::http::server::HTTPConfig {
                        model_dir: args.model_dir,
                        port: args.port,
                        use_debug_level: args.use_debug_level,
                        num_workers: args.num_workers,
                        with_s3_model_store: args.with_s3_model_store,
                        s3_bucket_name: args.s3_bucket_name,
                    };

                    jams_serve::http::server::start(config)
                        .await
                        .expect("Failed to start server ❌ \n");

                    // shutdown signal received
                    tracing::error!("Shutdown signal received ⚠️");
                    Ok(())
                }
                StartSubCommands::Grpc(args) => {
                    let config = jams_serve::grpc::server::GRPCConfig {
                        model_dir: args.model_dir,
                        port: args.port,
                        use_debug_level: args.use_debug_level,
                        num_workers: args.num_workers,
                        with_s3_model_store: args.with_s3_model_store,
                        s3_bucket_name: args.s3_bucket_name,
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
