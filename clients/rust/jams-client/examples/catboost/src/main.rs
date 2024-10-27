use jams_client::http::{ApiClientBuilder, Client};
use std::fs;

const URL: &str = "https://jams-http.onrender.com";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let http_client = ApiClientBuilder::new(URL.to_string()).build()?;

    // health check
    http_client.health_check().await?;

    // read file
    let payload = fs::read_to_string("clients/rust/jams-client/examples/catboost/request.json")?;

    // this is a binary classifier model and will return logits of each input record
    println!("CATBOOST PREDICTIONS");
    let preds = http_client
        .predict("titanic_model".to_string(), payload)
        .await?;
    let logits = preds.to_vec();
    println!("logits: {:?}", logits.clone());
    let probabilities = apply_sigmoid(logits.to_vec());
    println!("probabilities: {:?}", probabilities.clone().to_vec());
    let class_predictions = apply_class_label(probabilities);
    println!("class predictions: {:?}", class_predictions);

    Ok(())
}

// Sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Apply sigmoid to each element in a 2D vector
fn apply_sigmoid(inputs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut outputs: Vec<Vec<f64>> = vec![];

    for row in inputs.iter() {
        let mut sigmoid_row: Vec<f64> = vec![];
        for &value in row.iter() {
            sigmoid_row.push(sigmoid(value));
        }
        outputs.push(sigmoid_row);
    }

    outputs
}

// Get class labels from probabilities
fn get_class_label(input: f64) -> i32 {
    if input >= 0.5 {
        1
    } else {
        0
    }
}

// Apply class label threshold to each element in the 2D vector
fn apply_class_label(inputs: Vec<Vec<f64>>) -> Vec<i32> {
    let mut outputs: Vec<i32> = vec![];

    for row in inputs.iter() {
        for &value in row.iter() {
            outputs.push(get_class_label(value));
        }
    }

    outputs
}
