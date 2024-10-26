use crate::model_store::aws::common::download_objects;
use crate::model_store::common::{cleanup, DOWNLOADED_MODELS_DIRECTORY_NAME_PREFIX};
use crate::model_store::fetcher::Fetcher;
use crate::model_store::storage::{
    append_model_format, extract_framework_from_path, load_predictor, Metadata, Model, ModelName,
    Storage,
};
use async_trait::async_trait;
use aws_config::meta::region::ProvideRegion;
use aws_config::BehaviorVersion;
use aws_sdk_s3 as s3;
use aws_sdk_s3::config::{Credentials, Region};
use chrono::Utc;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use std::env;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

/// A struct representing a model store that interfaces with S3.
pub struct S3ModelStore {
    /// A thread-safe map of model names to their corresponding models.
    pub models: Arc<DashMap<ModelName, Arc<Model>>>,
    /// An S3 client for interacting with the S3 service.
    client: s3::Client,
    /// The name S3 bucket where models are stored.
    bucket_name: String,
    /// Directory, which stores the model artifacts downloaded from S3
    model_store_dir: String,
}

impl S3ModelStore {
    /// Creates a new instance of `S3ModelStore`.
    ///
    /// This function initializes an `S3ModelStore` by loading the AWS configuration, creating an S3 client,
    /// and fetching models from the specified S3 bucket. It also sets up a local directory to store the
    /// downloaded models.
    /// # Arguments
    ///
    /// * `bucket_name` - Name of the S3 bucket where models are stored.
    ///
    /// # Returns
    ///
    /// A result containing the newly created `S3ModelStore` or an error if the initialization fails.
    pub async fn new(bucket_name: String, use_minio: Option<bool>) -> anyhow::Result<Self> {
        // Ensure model_dir_uri is not empty, return error if empty
        if bucket_name.is_empty() {
            anyhow::bail!("S3 bucket name must be specified ❌")
        }

        // Check if minio is to be used instead of the standard AWS client.
        let client = if use_minio.is_some() {
            match build_minio_client().await {
                Ok(client) => {
                    log::info!("Using MinIO as model store ℹ️");
                    client
                }
                Err(e) => {
                    anyhow::bail!("Failed to build MinIO client ❌: {}", e.to_string());
                }
            }
        } else {
            match build_s3_client(use_localstack()).await {
                Ok(client) => {
                    log::info!("Using AWS S3 as model store ℹ️");
                    client
                }
                Err(e) => {
                    anyhow::bail!("Failed to build S3 client ❌: {}", e.to_string());
                }
            }
        };

        // Directory, which stores the models downloaded from S3
        let model_store_dir = format!(
            "{}/{}_{}",
            std::env::var("HOME").unwrap_or("/usr/local".to_string()),
            DOWNLOADED_MODELS_DIRECTORY_NAME_PREFIX,
            Uuid::new_v4(),
        );
        std::fs::create_dir(model_store_dir.clone())?;

        // Check if S3 is empty, if yes then return models dashmap as empty
        if client.is_empty(Some(bucket_name.clone())).await? {
            let models: DashMap<ModelName, Arc<Model>> = DashMap::new();
            Ok(Self {
                models: Arc::new(models),
                client,
                bucket_name,
                model_store_dir,
            })
        } else {
            // Fetch the models from S3
            let models = match client
                .fetch_models(Some(bucket_name.clone()), model_store_dir.clone())
                .await
            {
                Ok(models) => {
                    log::info!("Successfully fetched valid models from S3 ✅");
                    models
                }
                Err(e) => {
                    anyhow::bail!("Failed to fetch models ❌ - {}", e.to_string());
                }
            };

            Ok(Self {
                models: Arc::new(models),
                client,
                bucket_name,
                model_store_dir,
            })
        }
    }
}

/// Asynchronously builds and returns an S3 client.
///
/// # Parameters
///
/// - `use_localstack`: A boolean flag to indicate if LocalStack should be used.
///
/// # Returns
///
/// - `Result<s3::Client, anyhow::Error>`: A result containing the configured S3 client or an error.
///
/// # Errors
///
/// - Returns an error if the AWS region cannot be determined from the configuration.
async fn build_s3_client(use_localstack: bool) -> anyhow::Result<s3::Client> {
    // Setup AWS configs
    let aws_config = aws_config::defaults(BehaviorVersion::latest()).load().await;
    let region = match aws_config.region() {
        None => {
            anyhow::bail!("Failed to get AWS region from config ❌")
        }
        Some(value) => {
            // Convert &Region to Region
            value.region().await
        }
    };

    let credentials = match aws_config.credentials_provider() {
        None => {
            anyhow::bail!("failed to get credentials provider ❌")
        }
        Some(credentials) => credentials,
    };

    let mut s3_config = s3::Config::builder()
        .region(region)
        .credentials_provider(credentials)
        .behavior_version_latest();
    if use_localstack {
        let hostname = env::var("LOCALSTACK_HOSTNAME").unwrap_or("localhost".to_string());
        s3_config = s3_config
            .force_path_style(true)
            .endpoint_url(format!("http://{}:4566/", hostname));
    }
    let s3_config = s3_config.build();
    // Create an S3 client with the loaded configuration
    Ok(s3::Client::from_conf(s3_config))
}

async fn build_minio_client() -> anyhow::Result<s3::Client> {
    let key_id = env::var("MINIO_ACCESS_KEY_ID").unwrap_or("minioadmin".to_string());
    let secret_key = env::var("MINIO_ACCESS_KEY_ID").unwrap_or("minioadmin".to_string());
    let url = env::var("MINIO_URL").unwrap_or("http://0.0.0.0:9000".to_string());
    let region = env::var("AWS_REGION").unwrap_or("eu-west-2".to_string());

    let cred = Credentials::new(
        key_id,
        secret_key,
        None,
        None,
        "minio-loaded-from-custom-env",
    );

    let s3_config = aws_sdk_s3::config::Builder::new()
        .endpoint_url(url)
        .credentials_provider(cred)
        .behavior_version_latest()
        .region(Region::new(region))
        .force_path_style(true) // apply bucketname as path param instead of pre-domain
        .build();

    let client = aws_sdk_s3::Client::from_conf(s3_config);

    // healthcheck
    match client.list_buckets().send().await {
        Ok(_) => Ok(client),
        Err(e) => {
            anyhow::bail!(
                "Failed to connect to MinIO model store: {} ❌",
                e.to_string()
            )
        }
    }
}

/// Implements the `Drop` trait for `S3ModelStore`.
///
/// This implementation ensures that the temporary directory used for the model store is cleaned up
/// when the `S3ModelStore` instance is dropped. This helps to avoid leaving temporary files on disk
/// and ensures proper resource cleanup.
///
/// # Fields
///
/// * `model_store_dir` - The directory path for the temporary local model store which contains models downloaded from S3.
///
impl Drop for S3ModelStore {
    fn drop(&mut self) {
        cleanup(self.model_store_dir.clone())
    }
}

/// Implements the `Storage` trait for `S3ModelStore`, allowing models to be added and updated.
///
/// This implementation interacts with an S3 bucket to add and update machine learning models.
/// It downloads models from S3, loads them into memory, and manages them using a hash map.
///
#[async_trait]
impl Storage for S3ModelStore {
    /// Adds a new model to the S3 model store.
    ///
    /// This method downloads the model from S3, extracts the framework, loads the model into memory,
    /// and stores it in the `models` hashmap.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model.
    /// * `_model_path` - The path to the model file (unused in this implementation as models are fetched from S3).
    ///
    /// # Returns
    ///
    /// * `Result<()>` - An empty result indicating success or an error.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// * The model cannot be downloaded from S3.
    /// * The framework cannot be extracted from the model path.
    /// * The model cannot be loaded into memory.
    #[tracing::instrument(skip(self))]
    async fn add_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        // Prepare the S3 key from model_name
        // It is assumed that model will always be present as a .tar.gz file in S3
        // Panic otherwise
        let object_key = format!("{}.tar.gz", model_name);

        // Download the model
        match download_objects(
            &self.client,
            self.bucket_name.clone(),
            vec![object_key],
            self.model_store_dir.as_str(),
        )
        .await
        {
            Ok(_) => {
                log::info!("Downloaded object from s3 ✅");
            }
            Err(e) => {
                anyhow::bail!("Failed to download object from s3 ❌: {}", e.to_string());
            }
        };

        // todo: Figure out a better approach or provide utils function in python which pack the artefacts in the required format
        // At this point we have extracted the tar ball from S3
        // It is assumed that the name of tar ball and the actual mode name is the same
        // Based on the framework, the path is modified by appending the format
        // If pytorch -> append '.pt'
        // If lightgbm -> append '.txt'

        // Extract model framework
        let model_framework = match extract_framework_from_path(model_name.clone()) {
            None => {
                anyhow::bail!("Failed to extract framework from path");
            }
            Some(model_framework) => model_framework,
        };

        // We will not be using the model_path from the request
        // as we have all the information to load the model
        let model_path = append_model_format(
            model_framework,
            format!("{}/{}", self.model_store_dir, model_name.clone()),
        );

        // Load the model into memory
        match load_predictor(model_framework, model_path.as_str()).await {
            Ok(predictor) => {
                let sanitized_model_name =
                    match model_name.strip_prefix(format!("{}-", model_framework).as_str()) {
                        None => {
                            anyhow::bail!("Failed to sanitize model name");
                        }
                        Some(name) => name.to_string(),
                    };

                let now = Utc::now();
                let model = Model::new(
                    predictor,
                    sanitized_model_name.clone(),
                    model_framework,
                    model_path.to_string(),
                    now.to_rfc2822(),
                );
                self.models.insert(sanitized_model_name, Arc::new(model));
                Ok(())
            }
            Err(e) => {
                anyhow::bail!("Failed to add new model: {e}")
            }
        }
    }

    /// Updates an existing model in the S3 model store.
    ///
    /// This method updates the specified model by fetching the latest version from S3,
    /// extracting the framework, reloading the model into memory, and updating the `models` hashmap.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to update.
    ///
    /// # Returns
    ///
    /// * `Result<()>` - An empty result indicating success or an error.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// * The specified model does not exist in the `models` hashmap.
    /// * The latest model version cannot be downloaded from S3.
    /// * The framework cannot be extracted from the model path.
    /// * The model cannot be loaded into memory.
    #[tracing::instrument(skip(self))]
    async fn update_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        // By calling remove on the hashmap, the object is returned on success/
        // We use the returned object, in this case the model to extract the framework and model path
        match self.models.remove(model_name.as_str()) {
            None => {
                anyhow::bail!(
                    "Failed to update as the specified model {} does not exist",
                    model_name
                )
            }
            Some(model) => {
                let (model_framework, model_path) =
                    (model.1.info.framework, model.1.info.path.as_str());

                // Prepare the S3 key from model_name
                let object_key = format!("{}-{}.tar.gz", model_framework, model_name);

                // Fetch the latest model from S3
                match download_objects(
                    &self.client,
                    self.bucket_name.clone(),
                    vec![object_key],
                    self.model_store_dir.as_str(),
                )
                .await
                {
                    Ok(_) => {
                        log::info!("Downloaded object from s3 ✅");

                        match load_predictor(model_framework, model_path).await {
                            Ok(predictor) => {
                                let now = Utc::now();
                                let model = Model::new(
                                    predictor,
                                    model_name.clone(),
                                    model_framework,
                                    model_path.to_string(), // todo: use S3 path here and not the local model dir path
                                    now.to_rfc2822(),
                                );
                                self.models.insert(model_name.clone(), Arc::new(model));
                                Ok(())
                            }
                            Err(e) => {
                                anyhow::bail!(
                                    "Failed to update the specified model {}: {}",
                                    model_name,
                                    e
                                )
                            }
                        }
                    }
                    Err(_) => {
                        anyhow::bail!("Failed to download object from s3 ❌");
                    }
                }
            }
        }
    }

    /// Retrieves a model from the model store.
    ///
    /// This function returns a reference to the model with the specified name if it exists.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be retrieved.
    ///
    /// # Returns
    ///
    /// This function returns an `Option`:
    /// * `Some(Ref<ModelName, Arc<Model>>)` if the model exists.
    /// * `None` if the model does not exist.
    ///
    #[tracing::instrument(skip(self))]
    fn get_model(&self, model_name: ModelName) -> Option<Ref<ModelName, Arc<Model>>> {
        self.models.get(model_name.as_str())
    }

    /// Retrieves metadata for all models in the model store.
    ///
    /// This function returns a vector of `Metadata` containing information about all the models in the store.
    ///
    /// # Returns
    ///
    /// This function returns an `anyhow::Result` containing a vector of `Metadata`.
    ///
    #[tracing::instrument(skip(self))]
    fn get_models(&self) -> anyhow::Result<Vec<Metadata>> {
        let model: Vec<Metadata> = self
            .models
            .iter()
            .map(|f| f.value().info.to_owned())
            .collect();
        Ok(model)
    }

    /// Deletes a model from the model store.
    ///
    /// This function removes the model with the specified name from the store.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model to be deleted.
    ///
    /// # Errors
    ///
    /// This function returns an error if the specified model does not exist in the store.
    #[tracing::instrument(skip(self))]
    fn delete_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        match self.models.remove(&model_name) {
            None => {
                anyhow::bail!(
                    "Failed to delete model as the specified model {} does not exist",
                    model_name
                )
            }
            Some(_) => Ok(()),
        }
    }

    /// Periodically polls the model store to fetch and update models.
    ///
    /// This asynchronous function waits for the specified time interval, then fetches models from an S3
    /// bucket using the `fetch_models` function. If new or updated models are found, they are inserted into
    /// the in-memory model store (`self.models`). The polling process continues indefinitely with each cycle
    /// waiting for the specified interval.
    ///
    /// # Arguments
    ///
    /// * `interval` - A `Duration` specifying the time interval between each polling operation.
    ///
    /// # Returns
    ///
    /// * `Ok(())` if the models were successfully fetched and updated in the model store.
    /// * `Err(anyhow::Error)` if there was an error during the fetch or update process, including S3 fetch failures.
    ///
    #[tracing::instrument(skip(self))]
    async fn poll(&self, interval: Duration) -> anyhow::Result<()> {
        // poll every n time interval
        tokio::time::sleep(interval).await;

        log::info!("Polling model store ⌛");
        let models = match self
            .client
            .fetch_models(Some(self.bucket_name.clone()), self.model_store_dir.clone())
            .await
        {
            Ok(models) => {
                log::info!("Successfully fetched valid models from S3 ✅");
                models
            }
            Err(e) => {
                anyhow::bail!("Failed to fetch models ❌ - {}", e.to_string());
            }
        };

        for (model_name, model) in models {
            self.models.insert(model_name, model);
        }

        Ok(())
    }
}

fn use_localstack() -> bool {
    std::env::var("USE_LOCALSTACK").unwrap_or_default() == "true"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::aws::s3::S3ModelStore;
    use aws_sdk_s3::primitives::ByteStream;
    use aws_sdk_s3::types::{
        BucketLocationConstraint, CreateBucketConfiguration, Delete, ObjectIdentifier,
    };
    use rand::prelude::*;
    use std::time::Duration;

    fn generate_bucket_name() -> String {
        let mut rng = rand::thread_rng();
        let random_number = rng.gen_range(0..999999);
        format!("jams-model-store-test-{}", random_number)
    }

    async fn setup_client() -> s3::Client {
        if !use_localstack() {
            panic!("USE_LOCALSTACK env not set")
        }
        build_s3_client(true).await.unwrap()
    }

    async fn setup_test_dependencies(client: s3::Client, bucket_name: String) {
        create_test_bucket(client.clone(), bucket_name.clone()).await;
        upload_models_for_test(client.clone(), bucket_name).await;
    }

    async fn create_test_bucket(client: s3::Client, bucket_name: String) {
        let region = client.config().region().unwrap().to_string();
        let constraint = BucketLocationConstraint::from(region.as_str());
        let cfg = CreateBucketConfiguration::builder()
            .location_constraint(constraint)
            .build();
        client
            .create_bucket()
            .create_bucket_configuration(cfg)
            .bucket(bucket_name.clone())
            .send()
            .await
            .unwrap();
    }

    async fn upload_models_for_test(client: s3::Client, bucket_name: String) {
        let mut dir = tokio::fs::read_dir("tests/model_storage/model_store")
            .await
            .unwrap();

        while let Some(entry) = dir.next_entry().await.unwrap() {
            let body = ByteStream::from_path(entry.path()).await.unwrap();
            client
                .put_object()
                .bucket(bucket_name.clone())
                .key(entry.file_name().to_str().unwrap())
                .body(body)
                .send()
                .await
                .unwrap();
        }
    }

    // Code used from AWS documentation
    async fn delete_models_for_test(client: s3::Client, bucket_name: String) {
        let objects = client
            .list_objects_v2()
            .bucket(bucket_name.clone())
            .send()
            .await
            .unwrap();

        let mut delete_objects: Vec<ObjectIdentifier> = vec![];
        for obj in objects.contents() {
            let obj_id = ObjectIdentifier::builder()
                .set_key(Some(obj.key().unwrap().to_string()))
                .build()
                .unwrap();
            delete_objects.push(obj_id);
        }

        if !delete_objects.is_empty() {
            client
                .delete_objects()
                .bucket(bucket_name.clone())
                .delete(
                    Delete::builder()
                        .set_objects(Some(delete_objects))
                        .build()
                        .unwrap(),
                )
                .send()
                .await
                .unwrap();
        }
        client
            .delete_bucket()
            .bucket(bucket_name)
            .send()
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn successfully_create_s3_model_store_without_models_and_then_add_one() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        create_test_bucket(client.clone(), bucket_name.clone()).await;

        // create s3 model store without models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await;

        // assert
        assert!(model_store.is_ok());
        let model_store = model_store.unwrap();
        assert_eq!(model_store.models.len(), 0);

        // add model - upload model to s3 then call add model
        upload_models_for_test(client.clone(), bucket_name).await;

        let resp = model_store
            .add_model("catboost-titanic_model".to_string())
            .await;
        assert!(resp.is_ok());
        assert_eq!(model_store.models.len(), 1);
    }

    #[tokio::test]
    async fn successfully_create_minio_model_store_without_models_and_then_add_one() {
        // setup
        let client = build_minio_client().await.unwrap();
        let bucket_name = generate_bucket_name();
        create_test_bucket(client.clone(), bucket_name.clone()).await;

        // create minio model store without models
        let model_store = S3ModelStore::new(bucket_name.clone(), Some(true)).await;

        // assert
        assert!(model_store.is_ok());
        let model_store = model_store.unwrap();
        assert_eq!(model_store.models.len(), 0);

        // add model - upload model to s3 then call add model
        upload_models_for_test(client.clone(), bucket_name).await;

        let resp = model_store
            .add_model("catboost-titanic_model".to_string())
            .await;
        assert!(resp.is_ok());
        assert_eq!(model_store.models.len(), 1);
    }

    #[tokio::test]
    async fn successfully_load_models_from_s3_model_store() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        setup_test_dependencies(client.clone(), bucket_name.clone()).await;

        // load models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await;

        // assert
        assert!(model_store.is_ok());
        assert_ne!(model_store.unwrap().models.len(), 0);

        // cleanup
        delete_models_for_test(client.clone(), bucket_name.clone()).await
    }

    #[tokio::test]
    async fn successfully_get_model_from_s3_model_store() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        setup_test_dependencies(client.clone(), bucket_name.clone()).await;

        // load models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await.unwrap();
        let model = model_store.get_model("my_awesome_reg_model".to_string());

        // assert
        assert!(model.is_some());

        // cleanup
        delete_models_for_test(client.clone(), bucket_name.clone()).await
    }

    #[tokio::test]
    async fn fails_to_get_model_from_s3_model_store_when_model_name_is_wrong() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        setup_test_dependencies(client.clone(), bucket_name.clone()).await;

        // load models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await.unwrap();
        let model = model_store.get_model("model_which_does_not_exist".to_string());

        // assert
        assert!(model.is_none());

        // cleanup
        delete_models_for_test(client.clone(), bucket_name.clone()).await
    }

    #[tokio::test]
    async fn successfully_get_models_from_s3_model_store() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        setup_test_dependencies(client.clone(), bucket_name.clone()).await;

        // load models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await.unwrap();
        let models = model_store.get_models();

        // assert
        assert!(models.is_ok());
        assert_ne!(models.unwrap().len(), 0);

        // cleanup
        delete_models_for_test(client.clone(), bucket_name.clone()).await
    }

    #[tokio::test]
    async fn successfully_deletes_model_in_the_s3_model_store() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        setup_test_dependencies(client.clone(), bucket_name.clone()).await;

        // load models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await.unwrap();
        let deletion = model_store.delete_model("my_awesome_penguin_model".to_string());

        // assert
        assert!(deletion.is_ok());

        // cleanup
        delete_models_for_test(client.clone(), bucket_name.clone()).await
    }

    #[tokio::test]
    async fn fails_to_deletes_model_in_the_s3_model_store() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        setup_test_dependencies(client.clone(), bucket_name.clone()).await;

        // load models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await.unwrap();
        let deletion = model_store.delete_model("model_which_does_not_exist".to_string());

        // assert
        assert!(deletion.is_err());

        // cleanup
        delete_models_for_test(client.clone(), bucket_name.clone()).await
    }

    #[tokio::test]
    async fn successfully_update_model_in_the_s3_model_store() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        setup_test_dependencies(client.clone(), bucket_name.clone()).await;
        let model_name = "my_awesome_reg_model".to_string();

        // load models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await.unwrap();
        tokio::time::sleep(Duration::from_secs_f32(1.25)).await;

        // retrieve timestamp from existing to model for assertion
        let model = model_store
            .get_model(model_name.clone())
            .unwrap()
            .to_owned();

        // update model
        let update = model_store.update_model(model_name.clone()).await;
        assert!(update.is_ok());
        let updated_model = model_store
            .get_model(model_name.clone())
            .unwrap()
            .to_owned();

        // assert
        assert_eq!(model.info.name, updated_model.info.name);
        assert_eq!(model.info.path, updated_model.info.path);
        assert_ne!(model.info.last_updated, updated_model.info.last_updated); // as model will be updated

        // cleanup
        delete_models_for_test(client.clone(), bucket_name.clone()).await
    }

    #[tokio::test]
    async fn fails_to_update_model_in_the_s3_model_store_when_model_name_is_incorrect() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        setup_test_dependencies(client.clone(), bucket_name.clone()).await;
        let incorrect_model_name = "my_awesome_reg_model_incorrect".to_string();

        // load models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await.unwrap();

        // update model with incorrect model name
        let update = model_store.update_model(incorrect_model_name).await;

        // assert
        assert!(update.is_err());

        // cleanup
        delete_models_for_test(client.clone(), bucket_name.clone()).await
    }

    #[tokio::test]
    async fn successfully_add_model_in_the_s3_model_store() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        setup_test_dependencies(client.clone(), bucket_name.clone()).await;

        // load models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await.unwrap();

        // delete model to set up test
        model_store
            .delete_model("titanic_model".to_string())
            .unwrap();
        // assert that model is not present
        let model = model_store.get_model("titanic_model".to_string());
        assert!(model.is_none());
        let num_models = model_store.get_models().unwrap().len();

        // add model - unlike local model store we will pass the s3 key without .tar.gz in the model name
        // model_path is not required when adding models via S3 model store
        let add = model_store
            .add_model("catboost-titanic_model".to_string())
            .await;
        let num_models_after_add = model_store.get_models().unwrap().len();

        // assert
        assert!(add.is_ok());
        let model = model_store.get_model("titanic_model".to_string());
        assert!(model.is_some());
        assert_eq!(num_models_after_add - num_models, 1);

        // cleanup
        delete_models_for_test(client.clone(), bucket_name.clone()).await
    }

    #[tokio::test]
    async fn fails_to_add_model_in_the_s3_model_store_when_the_model_name_is_wrong() {
        // setup
        let client = setup_client().await;
        let bucket_name = generate_bucket_name();
        setup_test_dependencies(client.clone(), bucket_name.clone()).await;

        // load models
        let model_store = S3ModelStore::new(bucket_name.clone(), None).await.unwrap();

        // add model
        let add = model_store.add_model("wrong_model_name".to_string()).await;

        // assert
        assert!(add.is_err());

        // cleanup
        delete_models_for_test(client.clone(), bucket_name.clone()).await
    }
}
