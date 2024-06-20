use crate::grpc::helper::{grpc_client_stub, jams_grpc_test_router};
use jams_serve::grpc::service::jams_v1::{AddModelRequest, DeleteModelRequest, UpdateModelRequest};
use tokio::net::TcpListener;
use tonic::codegen::tokio_stream::wrappers::TcpListenerStream;

#[tokio::test]
async fn successfully_calls_the_get_models_rpc() {
    // Arrange
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let test_server = jams_grpc_test_router().await;

    tokio::spawn(async move {
        test_server
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let mut client = grpc_client_stub(addr.to_string()).await;

    // Act
    let result = client.get_models(()).await;

    // Assert
    assert!(result.is_ok());
}

#[tokio::test]
async fn successfully_calls_the_add_model_rpc() {
    // Arrange
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let test_server = jams_grpc_test_router().await;

    tokio::spawn(async move {
        test_server
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let mut client = grpc_client_stub(addr.to_string()).await;

    // Act
    let response = client
        .add_model(AddModelRequest {
            model_name: "tensorflow-my_awesome_penguin_model".to_string(),
            model_path: "".to_string(),
        })
        .await;

    // Assert
    assert!(response.is_ok());
}

#[tokio::test]
async fn fails_to_call_the_add_model_rpc_when_model_path_is_wrong() {
    // Arrange
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let test_server = jams_grpc_test_router().await;

    tokio::spawn(async move {
        test_server
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let mut client = grpc_client_stub(addr.to_string()).await;

    // Act
    let response = client
        .add_model(AddModelRequest {
            model_name: "my_awesome_penguin_model".to_string(),
            model_path: "incorrect/model/path".to_string(),
        })
        .await;

    // Assert
    assert!(response.is_err());
}

#[tokio::test]
async fn successfully_calls_the_update_model_rpc() {
    // Arrange
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let test_server = jams_grpc_test_router().await;

    tokio::spawn(async move {
        test_server
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let mut client = grpc_client_stub(addr.to_string()).await;

    // Act: Make update call
    let response = client
        .update_model(UpdateModelRequest {
            model_name: "my_awesome_penguin_model".to_string(),
        })
        .await;

    // Assert
    assert!(response.is_ok());
}

#[tokio::test]
async fn fails_to_call_the_update_model_rpc_when_model_name_is_wrong() {
    // Arrange
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let test_server = jams_grpc_test_router().await;

    tokio::spawn(async move {
        test_server
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let mut client = grpc_client_stub(addr.to_string()).await;

    // Act: Make update call
    let response = client
        .update_model(UpdateModelRequest {
            model_name: "incorrect_model_name".to_string(),
        })
        .await;

    // Assert
    assert!(response.is_err());
}

#[tokio::test]
async fn successfully_calls_the_delete_model_rpc() {
    // Arrange
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let test_server = jams_grpc_test_router().await;

    tokio::spawn(async move {
        test_server
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let mut client = grpc_client_stub(addr.to_string()).await;

    // Act: Make delete call
    let response = client
        .delete_model(DeleteModelRequest {
            model_name: "my_awesome_reg_model".to_string(),
        })
        .await;

    // Assert
    assert!(response.is_ok());
}

#[tokio::test]
async fn fails_to_call_the_delete_model_rpc_when_model_name_is_wrong() {
    // Arrange
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let test_server = jams_grpc_test_router().await;

    tokio::spawn(async move {
        test_server
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let mut client = grpc_client_stub(addr.to_string()).await;

    // Act: Make delete call
    let response = client
        .delete_model(DeleteModelRequest {
            model_name: "incorrect_model".to_string(),
        })
        .await;

    // Assert
    assert!(response.is_err());
}
