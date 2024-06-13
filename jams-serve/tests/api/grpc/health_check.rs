use crate::grpc::helper::{grpc_client_stub, jams_grpc_test_router};

#[tokio::test]
async fn successfully_calls_the_health_check_rpc() {
    // Arrange
    let addr = "0.0.0.0:0".parse().unwrap();
    let test_server = jams_grpc_test_router();

    tokio::spawn(async move {
        test_server.serve(addr).await.unwrap();
    });

    let mut client = grpc_client_stub(addr.to_string()).await;

    // Act
    let response = client.health_check(()).await;

    // Assert
    assert!(response.is_ok());
}
