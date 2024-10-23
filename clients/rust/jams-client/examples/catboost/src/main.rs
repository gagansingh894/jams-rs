use jams_client::http::{ApiClient, Client};

const URL: &str = "https://jams-http.onrender.com";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let http_client = ApiClient::new(URL.to_string())?;
    
    // health check
    http_client.health_check().await?;
    
    Ok(())
}
