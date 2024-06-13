use crate::grpc::helper::{grpc_client_stub, jams_grpc_test_router};
use jams_serve::grpc::service::jams_v1::{AddModelRequest, PredictRequest};
use tokio::net::TcpListener;
use tonic::codegen::tokio_stream::wrappers::TcpListenerStream;

#[tokio::test]
async fn successfully_calls_the_predict_rpc() {
    // Arrange
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let test_server = jams_grpc_test_router();

    tokio::spawn(async move {
        test_server
            .serve_with_incoming(TcpListenerStream::new(listener))
            .await
            .unwrap();
    });
    let mut client = grpc_client_stub(addr.to_string()).await;

    // Act: Add model for prediction
    let response = client
        .add_model(AddModelRequest {
            model_name: "titanic_model".to_string(),
            model_path: "tests/local_model_store/catboost-titanic_model".to_string(),
        })
        .await;
    assert!(response.is_ok());

    // Act: Make Predictions
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
    let response = client
        .predict(PredictRequest {
            model_name: "titanic_model".to_string(),
            input: model_input,
        })
        .await;

    // Assert
    assert!(response.is_ok());
}
