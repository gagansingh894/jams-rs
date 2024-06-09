use crate::helper::test_router;
use reqwest::Client;
use tokio::net::TcpListener;

#[tokio::test]
async fn successfully_calls_the_predict_endpoint_and_return_200() {
    // Arrange
    let client = Client::new();
    let listener = TcpListener::bind("0.0.0.0:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = test_router();
    let url = format!("http://{}/api/predict", addr).to_string();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Act: Add model for prediction
    let response = client
        .post(url.clone())
        .json(&serde_json::json!(
            {
                "model_name": "catboost-titanic_model",
                "model_path": "assets/model_storage/local_model_store/catboost-titanic_model"
            }
        ))
        .send()
        .await
        .expect("Failed to make request");
    assert!(response.status().is_success());

    // Act: Make Predictions
    let response = client
        .post(url)
        .json(&serde_json::json!(
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
        ))
        .send()
        .await
        .expect("Failed to make request");

    // Assert
    assert!(response.status().is_success())
}
