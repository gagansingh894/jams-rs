use std::fs;
use jams_client::http::{ApiClient, Client};

const URL: &str = "https://jams-http.onrender.com";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let http_client = ApiClient::new(URL.to_string())?;

    // health check
    http_client.health_check().await?;

    // read file
    let payload = fs::read_to_string("clients/rust/jams-client/examples/pytorch/request.json")?;

    // this is a regression model so output would be continous for each input record
    println!("PYTORCH PREDICTIONS");
    let preds = http_client.predict("my_awesome_californiahousing_model".to_string(), payload).await?;
    println!("values: {:?}", preds.to_vec());

    Ok(())
}

