use jams_client::http::{ApiClientBuilder, Client};
use std::fs;

const URL: &str = "https://jams-http.onrender.com";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let http_client = ApiClientBuilder::new(URL.to_string()).build()?;

    // health check
    http_client.health_check().await?;

    // read file
    let payload = fs::read_to_string("clients/rust/jams-client/examples/tensorflow/request.json")?;

    // this will return a multiclass response for each input record. we can use argmax to get the index of the class
    println!("TENSORFLOW PREDICTIONS");
    let preds = http_client
        .predict("my_awesome_penguin_model".to_string(), payload)
        .await?;
    println!("penguin species labels: {:?}", apply_argmax(preds.to_vec()));

    Ok(())
}

// Argmax function returns the index of the maximum value in a slice
fn argmax(arr: Vec<f64>) -> Option<usize> {
    if arr.is_empty() {
        return None; // Return None for an empty array
    }

    let mut max_index = 0;
    let mut max_value = arr[0];

    for (i, &value) in arr.iter().enumerate() {
        if value > max_value {
            max_value = value;
            max_index = i;
        }
    }

    Some(max_index)
}

// Apply argmax function to each row in the 2D vector
fn apply_argmax(inputs: Vec<Vec<f64>>) -> Vec<Option<usize>> {
    let mut outputs: Vec<Option<usize>> = Vec::new();

    for row in inputs.into_iter() {
        outputs.push(argmax(row));
    }

    outputs
}
