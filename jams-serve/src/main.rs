mod router;
mod service;

use crate::router::{build_router, shutdown_signal};
use clap::{Args, Parser, Subcommand};
use std::env;

/// CLI for starting an J.A.M.S
#[derive(Parser, Debug)]
#[clap(
    name = "J.A.M.S - Just Another Model Server",
    version = "0.1.0",
    author = "Gagandeep Singh",
    about = "Starts the server with specified model directory and optional port number. "
)]
struct Cli {
    #[clap(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the Axum server
    #[clap(name = "start")]
    Start(CommandArgs),
}

#[derive(Args, Debug)]
struct CommandArgs {
    /// Path to the directory containing models
    #[clap(long)]
    model_dir: Option<String>,

    /// Port number (default: 3000)
    #[clap(long)]
    port: Option<u16>,

    /// Toggle DEBUG logs on/off
    #[clap(long)]
    use_debug_level: Option<bool>,

    /// Number of threads to be used in CPU threadpool. This threadpool is different from the
    /// I/O threadpool and used for computing CPU intensive tasks (default: 2)
    #[clap(long)]
    num_workers: Option<usize>,
}

#[cfg(not(tarpaulin_include))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Commands::Start(args) => {
            start_server(args).await;
            Ok(())
        }
    }
}

async fn start_server(args: CommandArgs) {
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

    let model_dir = match args.model_dir {
        Some(dir) => dir,
        None => {
            // search for environment variable
            match env::var("MODEL_STORE_DIR") {
                Ok(model_dir) => model_dir,
                Err(_) => {
                    eprintln!("Error: either set MODEL_STORE_DIR  or use 'model-dir' argument.");
                    return;
                }
            }
        }
    };

    let port = args.port.unwrap_or(3000);
    let use_debug_level = args.use_debug_level.unwrap_or(false);
    let num_workers = args.num_workers.unwrap_or(2);

    let app =
        build_router(model_dir, use_debug_level, num_workers).expect("Failed to build router");
    // run our app with hyper, listening globally on specified port
    let address = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(address)
        .await
        .expect("Failed to create TCP listener ❌");

    // log that the server is running
    tracing::info!(
        "{}",
        format!("Server is running on http://0.0.0.0:{} ✅ \n", port)
    );

    // run on hyper
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("Failed to start server ❌ \n");

    // shutdown signal received
    tracing::error!("Shutdown signal received ⚠️")
}
