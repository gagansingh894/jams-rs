use crate::common::state::AppState;
use crate::common::worker;
use crate::grpc::service::jams_v1::get_models_response::Model;
use crate::grpc::service::jams_v1::{
    AddModelRequest, DeleteModelRequest, GetModelsResponse, PredictRequest, PredictResponse,
    UpdateModelRequest,
};
use jams_core::manager::Manager;
use jams_core::model_store::local::LocalModelStore;
use jams_core::model_store::storage::Metadata;
use jams_v1::model_server_server::ModelServer;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use tokio::sync::oneshot;
use tonic::{Request, Response, Status};

// load proto gen code as module
pub mod jams_v1 {
    tonic::include_proto!("jams_v1");
}

pub struct JamsService {
    app_state: Arc<AppState>,
}

impl JamsService {
    pub fn new(model_dir: String, worker_pool_threads: usize) -> anyhow::Result<Self> {
        if worker_pool_threads < 1 {
            anyhow::bail!("At least 1 worker is required for rayon threadpool")
        }

        // setup rayon thread pool for cpu intensive task
        let cpu_pool = ThreadPoolBuilder::new()
            .num_threads(worker_pool_threads)
            .build()
            .expect("Failed to build rayon threadpool ❌");

        // setup model store
        let model_store = LocalModelStore::new(model_dir).expect("Failed to create model store ❌");
        let manager = Manager::new(Arc::new(model_store)).expect("Failed to initialize manager ❌");

        // app state
        let app_state = Arc::new(AppState {
            manager: Arc::new(manager),
            cpu_pool,
        });

        Ok(JamsService { app_state })
    }
}

#[tonic::async_trait]
impl ModelServer for JamsService {
    async fn health_check(&self, _request: Request<()>) -> Result<Response<()>, Status> {
        Ok(Response::new(()))
    }

    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>, Status> {
        let (tx, rx) = oneshot::channel();

        let cpu_pool = &self.app_state.cpu_pool;
        let manager = Arc::clone(&self.app_state.manager);
        let model_name = request.get_ref().model_name.to_string();
        let model_input = request.get_ref().input.to_string();

        cpu_pool.spawn(move || worker::predict_and_send(manager, model_name, model_input, tx));

        match rx.await {
            Ok(predictions) => match predictions {
                Ok(output) => Ok(Response::new(PredictResponse { output })),
                Err(e) => Err(Status::new(
                    tonic::Code::Internal,
                    format!("Failed to make predictions: {}", e),
                )),
            },
            Err(e) => Err(Status::new(
                tonic::Code::Internal,
                format!("Failed to make predictions: {}", e),
            )),
        }
    }

    async fn get_models(
        &self,
        _request: Request<()>,
    ) -> Result<Response<GetModelsResponse>, Status> {
        match self.app_state.manager.get_models() {
            Ok(models) => Ok(Response::new(GetModelsResponse {
                total: models.len() as i32,
                models: parse_to_proto_models(models),
            })),
            Err(_) => Err(Status::new(
                tonic::Code::Internal,
                "Failed to add new model",
            )),
        }
    }

    async fn add_model(&self, request: Request<AddModelRequest>) -> Result<Response<()>, Status> {
        match self.app_state.manager.add_model(
            request.get_ref().model_name.as_str().to_string(),
            request.get_ref().model_path.as_str(),
        ) {
            Ok(_) => Ok(Response::new(())),
            Err(_) => Err(Status::new(
                tonic::Code::Internal,
                "Failed to add new model",
            )),
        }
    }

    async fn update_model(
        &self,
        request: Request<UpdateModelRequest>,
    ) -> Result<Response<()>, Status> {
        match self
            .app_state
            .manager
            .update_model(request.get_ref().model_name.to_string())
        {
            Ok(_) => Ok(Response::new(())),
            Err(_) => Err(Status::new(
                tonic::Code::Internal,
                "Failed to update existing model",
            )),
        }
    }

    async fn delete_model(
        &self,
        request: Request<DeleteModelRequest>,
    ) -> Result<Response<()>, Status> {
        match self
            .app_state
            .manager
            .delete_model(request.get_ref().model_name.to_string())
        {
            Ok(_) => Ok(Response::new(())),
            Err(_) => Err(Status::new(
                tonic::Code::Internal,
                "Failed to delete existing model",
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
    use jams_core::model::frameworks::TENSORFLOW;
    use jams_core::model_store::storage::Metadata;

    #[test]
    fn successfully_create_jams_service() {
        // Arrange
        let model_dir = "".to_string();
        let worker_pool_threads = 2;

        // Act
        let service = JamsService::new(model_dir, worker_pool_threads);

        // Assert
        assert!(service.is_ok())
    }

    #[test]
    fn failed_to_create_jams_service_due_to_zero_worker_in_rayon_threadpool() {
        // Arrange
        let model_dir = "".to_string();
        let worker_pool_threads = 0;

        // Act
        let service = JamsService::new(model_dir, worker_pool_threads);

        // Assert
        assert!(service.is_err())
    }

    #[test]
    #[should_panic]
    fn failed_to_build_jams_service_because_manager_is_unable_to_initialize() {
        // Arrange
        let model_dir = "incorrect/or/invalid/path/".to_string();
        let worker_pool_threads = 1;

        // Act
        let service = JamsService::new(model_dir, worker_pool_threads);

        // Assert
        assert!(service.is_err())
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
