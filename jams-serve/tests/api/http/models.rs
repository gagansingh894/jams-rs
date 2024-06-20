use crate::http::helper::test_router;
use reqwest::Client;
use tokio::net::TcpListener;

#[tokio::test]
async fn successfully_calls_the_get_models_endpoint_and_return_200() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router().await;
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
    let router = test_router().await;
    let url = format!("http://{}/api/models", addr).to_string();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Act
    let response = client
        .post(url)
        .json(&serde_json::json!(
            {
                "model_name": "tensorflow-my_awesome_penguin_model",
                "model_path": ""
            }
        ))
        .send()
        .await
        .expect("Failed to make request");

    // Assert
    assert!(response.status().is_success())
}

#[tokio::test]
async fn fails_to_call_the_add_model_endpoint_and_return_500_when_model_path_is_wrong() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router().await;
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
                "model_path": "incorrect/path"
            }
        ))
        .send()
        .await
        .expect("Failed to make request");

    // Assert
    assert!(response.status().is_server_error())
}

#[tokio::test]
async fn successfully_calls_the_update_model_endpoint_and_return_200() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router().await;
    let url = format!("http://{}/api/models", addr).to_string();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Act: Make update call
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
    let router = test_router().await;
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
    let router = test_router().await;
    let delete_url =
        format!("http://{}/api/models?model_name=my_awesome_reg_model", addr).to_string();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Act: Make delete call
    let response = client
        .delete(delete_url)
        .send()
        .await
        .expect("Failed to make request");

    // Assert
    assert!(response.status().is_success())
}

#[tokio::test]
async fn fails_to_call_the_delete_model_endpoint_and_return_500_when_model_does_not_exist() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router().await;
    let delete_url =
        format!("http://{}/api/models?model_name=model_does_not_exist", addr).to_string();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Act: Make delete call
    let response = client
        .delete(delete_url)
        .send()
        .await
        .expect("Failed to make request");

    // Assert
    assert!(response.status().is_server_error())
}
