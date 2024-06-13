use crate::http::helper::test_router;
use reqwest::Client;
use tokio::net::TcpListener;

#[tokio::test]
async fn successfully_calls_the_root_endpoint_and_return_200() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router();
    let url = format!("http://{}", addr).to_string();

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
async fn successfully_calls_the_healthcheck_endpoint_and_return_200() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router();
    let url = format!("http://{}/healthcheck", addr).to_string();

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
