use crate::common::state::AppState;
use crate::common::worker;
use jams_core::model_store::storage::Metadata;
use jams_proto::jams_v1::get_models_response::Model;
use jams_proto::jams_v1::model_server_server::ModelServer;
use jams_proto::jams_v1::{
    AddModelRequest, DeleteModelRequest, GetModelsResponse, PredictRequest, PredictResponse,
    UpdateModelRequest,
};
use std::sync::Arc;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

pub struct JamsService {
    app_state: Arc<AppState>,
}

impl JamsService {
    pub fn new(app_state: Arc<AppState>) -> anyhow::Result<Self> {
        Ok(JamsService { app_state })
    }
}

#[tonic::async_trait]
impl ModelServer for JamsService {
    #[tracing::instrument(skip(self))]
    async fn health_check(&self, _request: Request<()>) -> Result<Response<()>, Status> {
        Ok(Response::new(()))
    }

    #[tracing::instrument(skip(self, request))]
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let (tx, rx) = oneshot::channel();

        let cpu_pool = &self.app_state.cpu_pool;
        let manager = Arc::clone(&self.app_state.manager);
        let prediction_request = request.into_inner();
        let model_name = prediction_request.model_name;
        let model_input = prediction_request.input;

        cpu_pool.spawn(move || worker::predict_and_send(manager, model_name, model_input, tx));

        match rx.await {
            Ok(predictions) => match predictions {
                Ok(output) => Ok(Response::new(PredictResponse { output })),
                Err(e) => Err(Status::new(
                    tonic::Code::Internal,
                    format!("Failed to predict ❌: {}", e),
                )),
            },
            Err(e) => Err(Status::new(
                tonic::Code::Internal,
                format!("Failed to predict ❌: {}", e),
            )),
        }
    }

    #[tracing::instrument(skip(self, _request))]
    async fn get_models(
        &self,
        _request: Request<()>,
    ) -> Result<Response<GetModelsResponse>, Status> {
        match self.app_state.manager.get_models() {
            Ok(models) => Ok(Response::new(GetModelsResponse {
                total: models.len() as i32,
                models: parse_to_proto_models(models),
            })),
            Err(e) => Err(Status::new(
                tonic::Code::Internal,
                format!("Failed to get models ❌: {}", e),
            )),
        }
    }

    #[tracing::instrument(skip(self, request))]
    async fn add_model(&self, request: Request<AddModelRequest>) -> Result<Response<()>, Status> {
        let add_model_request = request.into_inner();
        match self
            .app_state
            .manager
            .add_model(add_model_request.model_name)
            .await
        {
            Ok(_) => Ok(Response::new(())),
            Err(_) => Err(Status::new(
                tonic::Code::Internal,
                "Failed to add model ❌. Please check server logs",
            )),
        }
    }

    #[tracing::instrument(skip(self, request))]
    async fn update_model(
        &self,
        request: Request<UpdateModelRequest>,
    ) -> Result<Response<()>, Status> {
        match self
            .app_state
            .manager
            .update_model(request.into_inner().model_name.to_string())
            .await
        {
            Ok(_) => Ok(Response::new(())),
            Err(_) => Err(Status::new(
                tonic::Code::Internal,
                "Failed to update model ❌. Please check server logs",
            )),
        }
    }

    #[tracing::instrument(skip(self, request))]
    async fn delete_model(
        &self,
        request: Request<DeleteModelRequest>,
    ) -> Result<Response<()>, Status> {
        match self
            .app_state
            .manager
            .delete_model(request.into_inner().model_name.to_string())
        {
            Ok(_) => Ok(Response::new(())),
            Err(e) => Err(Status::new(
                tonic::Code::Internal,
                format!("Failed to delete model ❌: {}", e),
            )),
        }
    }
}

fn parse_to_proto_models(models_metadata: Vec<Metadata>) -> Vec<Model> {
    let mut out: Vec<Model> = Vec::new();

    for data in models_metadata {
        out.push(Model {
            name: data.name,
            framework: data.framework.to_string(),
            path: data.path,
            last_updated: data.last_updated,
        })
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use jams_core::manager::ManagerBuilder;
    use jams_core::model::frameworks::TENSORFLOW;
    use jams_core::model_store::local::filesystem::LocalModelStore;
    use jams_core::model_store::storage::Metadata;
    use jams_core::model_store::ModelStore;
    use rayon::ThreadPoolBuilder;

    async fn setup_shared_state() -> Arc<AppState> {
        let cpu_pool = ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("Failed to build rayon threadpool ❌");

        let model_store = LocalModelStore::new("tests/model_store".to_string())
            .await
            .expect("Failed to create model store ❌");

        let manager = Arc::new(
            ManagerBuilder::new(Arc::new(ModelStore::Local(model_store)))
                .build()
                .expect("Failed to initialize manager ❌"),
        );

        Arc::new(AppState { manager, cpu_pool })
    }

    #[tokio::test]
    async fn successfully_create_jams_service() {
        // Arrange
        let shared_state = setup_shared_state().await;

        // Act
        let service = JamsService::new(shared_state);

        // Assert
        assert!(service.is_ok())
    }

    #[test]
    fn successfully_parse_to_proto_models() {
        // Arrange
        let now = Utc::now();
        let models_metadata: Vec<Metadata> = vec![
            Metadata {
                name: "my_model_1".to_string(),
                framework: TENSORFLOW,
                path: "some_path_1".to_string(),
                last_updated: now.to_rfc3339(),
            },
            Metadata {
                name: "my_model_2".to_string(),
                framework: TENSORFLOW,
                path: "some_path_2".to_string(),
                last_updated: now.to_rfc3339(),
            },
        ];

        // Act
        let proto_models = parse_to_proto_models(models_metadata.clone());

        // Assert
        assert_eq!(proto_models.len(), models_metadata.len());
        for i in 0..proto_models.len() {
            assert_eq!(proto_models[i].name, models_metadata[i].name);
            assert_eq!(proto_models[i].framework, models_metadata[i].framework);
            assert_eq!(proto_models[i].path, models_metadata[i].path);
            assert_eq!(
                proto_models[i].last_updated,
                models_metadata[i].last_updated
            );
        }
    }
}
