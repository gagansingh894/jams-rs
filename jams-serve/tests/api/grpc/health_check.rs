use tokio::net::TcpListener;
use tonic::codegen::tokio_stream::wrappers::TcpListenerStream;
use crate::grpc::helper::{grpc_client_stub, jams_grpc_test_router};

#[tokio::test]
async fn successfully_calls_the_health_check_rpc() {
    // Arrange
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let test_server = jams_grpc_test_router();

    tokio::spawn(async move {
        test_server
            .serve_with_incoming(TcpListenerStream::new(listener)).await.unwrap();
    });
    let mut client = grpc_client_stub(addr.to_string()).await;

    // Act
    let response = client.health_check(()).await;

    // Assert
    assert!(response.is_ok());
}
