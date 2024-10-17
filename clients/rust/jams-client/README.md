# JAMS-CLIENT

A HTTP and gRPC client library for interacting with  [**J.A.M.S - Just Another Model Server**](https://github.com/gagansingh894/jams-rs) written in Rust 🦀

## Installation
Add the following to cargo.toml

```
jams-client = "0.1"
```

## Usage

Start `J.A.M.S` by following the instructions [here](https://github.com/gagansingh894/jams-rs?tab=readme-ov-file#docker-setup)

Both HTTP and gRPC have the same API with the only difference in client creation

```
use jams_client::*;

// Create client
let client = http::ApiClient::new(get_url()).unwrap();

// For GRPC client
// let client = grpc::ApiClient::new(get_url()).unwrap();

// Predict
let model_name = "titanic_model".to_string();
let model_input = serde_json::json!(
        {
            "pclass": ["1", "3"],
            "sex": ["male", "female"],
            "age": [22.0, 23.79929292929293],
            "sibsp": ["0", "1", ],
            "parch": ["0", "0"],
            "fare": [151.55, 14.4542],
            "embarked": ["S", "C"],
            "class": ["First", "Third"],
            "who": ["man", "woman"],
            "adult_male": ["True", "False"],
            "deck": ["Unknown", "Unknown"],
            "embark_town": ["Southampton", "Cherbourg"],
            "alone": ["True", "False"]
        }
)
.to_string();
let resp = client.predict(model_name, model_input).await;
let predictions = resp.unwrap().to_vec() // use values


// Health Check
let resp = client.health_check().await;

// Get Models
let result = client.get_models().await;

// Add model - <MODEL_FRAMEWORK>-<MODEL_NAME>
let resp = client.add_model("pytorch-my_awesome_californiahousing_model".to_string()).await

// Update Model
let resp = client.update_model("titanic_model".to_string()).await;

// Delete Model
client.delete_model("my_awesome_penguin_model".to_string()).await.unwrap();
```