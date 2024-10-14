use serde::Serialize;

#[derive(Serialize)]
pub struct PredictRequest {
    model_name: String,
    input: String,
}

#[derive(Serialize)]
pub struct AddModelRequest {
    model_name: String,
}

#[derive(Serialize)]
pub struct UpdateModelRequest {
    model_name: String,
}
