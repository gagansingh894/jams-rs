use crate::http::helper::test_router;
use reqwest::Client;
use tokio::net::TcpListener;

#[tokio::test]
async fn successfully_calls_the_get_models_endpoint_and_return_200() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router();
    let url = format!("http://{}/api/models", addr).to_string();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Act
    let response = client
        .get(url)
        .send()
        .await
        .expect("Failed to make request");

    // Assert
    assert!(response.status().is_success())
}

#[tokio::test]
async fn successfully_calls_the_add_model_endpoint_and_return_200() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router();
    let url = format!("http://{}/api/models", addr).to_string();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Act
    let response = client
        .post(url)
        .json(&serde_json::json!(
            {
                "model_name": "my_awesome_penguin_model",
                "model_path": "tests/local_model_store/pytorch-my_awesome_californiahousing_model.pt"
            }
        ))
        .send()
        .await
        .expect("Failed to make request");

    // Assert
    assert!(response.status().is_success())
}

#[tokio::test]
async fn successfully_calls_the_update_model_endpoint_and_return_200() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router();
    let url = format!("http://{}/api/models", addr).to_string();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Act - 1: Add model first
    let response = client
        .post(url.clone())
        .json(&serde_json::json!(
            {
                "model_name": "my_awesome_penguin_model",
                "model_path": "tests/local_model_store/pytorch-my_awesome_californiahousing_model.pt"
            }
        ))
        .send()
        .await
        .expect("Failed to make request");
    assert!(response.status().is_success());

    // Act - 2: Make update call
    let response = client
        .put(url)
        .json(&serde_json::json!(
            {
                "model_name": "my_awesome_penguin_model",
            }
        ))
        .send()
        .await
        .expect("Failed to make request");

    // Assert
    assert!(response.status().is_success())
}

#[tokio::test]
async fn fails_to_call_the_update_model_endpoint_and_return_500_when_model_name_is_incorrect() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router();
    let url = format!("http://{}/api/models", addr).to_string();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Act: Make update call
    let response = client
        .put(url)
        .json(&serde_json::json!(
            {
                "model_name": "incorrect_model_name",
            }
        ))
        .send()
        .await
        .expect("Failed to make request");

    // Assert
    assert!(response.status().is_server_error())
}

#[tokio::test]
async fn successfully_calls_the_delete_model_endpoint_and_return_200() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router();
    let url = format!(
        "http://{}/api/models?model_name=my_awesome_californiahousing_model",
        addr
    )
    .to_string();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Act - 1: Add model first
    let response = client
        .post(url.clone())
        .json(&serde_json::json!(
            {
                "model_name": "my_awesome_californiahousing_model",
                "model_path": "tests/local_model_store/pytorch-my_awesome_californiahousing_model.pt"
            }
        ))
        .send()
        .await
        .expect("Failed to make request");
    assert!(response.status().is_success());

    // Act: Make delete call
    let response = client
        .delete(url)
        .send()
        .await
        .expect("Failed to make request");

    // Assert
    assert!(response.status().is_success())
}
