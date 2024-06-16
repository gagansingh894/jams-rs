use async_trait::async_trait;
use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3 as s3;
use chrono::Utc;
use dashmap::mapref::one::Ref;
use dashmap::DashMap;
use flate2::read::GzDecoder;
use std::fs::remove_dir_all;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use tar::Archive;
use tokio::io::AsyncWriteExt;

use crate::model_store::storage::{
    extract_framework_from_path, load_models, load_predictor, Metadata, Model, ModelName, Storage,
};

/// A struct representing a model store that interfaces with S3.
#[allow(dead_code)]
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

const DOWNLOADED_MODELS_DIRECTORY_NAME: &str = "model_store";

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
    pub async fn new(bucket_name: String) -> anyhow::Result<Self> {
        // Ensure model_dir_uri is not empty, return error if empty
        if bucket_name.is_empty() {
            anyhow::bail!("S3 bucket name must be specified ❌")
        }
        // Use the default region provider chain to load the region from the configuration file
        let region_provider = RegionProviderChain::default_provider();
        // Load the AWS configuration from the environment
        let config = aws_config::from_env().region(region_provider).load().await;
        // Create an S3 client with the loaded configuration
        let client = s3::Client::new(&config);
        // Directory, which stores the models downloaded from S3
        let model_store_dir = format!(
            "{}/{}",
            std::env::var("HOME").unwrap_or("/usr/local".to_string()),
            DOWNLOADED_MODELS_DIRECTORY_NAME
        );
        // Fetch the models from S3
        let models = match fetch_models(&client, bucket_name.clone(), model_store_dir.clone()).await
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
        match remove_dir_all(self.model_store_dir.clone()) {
            Ok(_) => {
                log::info!("Cleaning up temporary location for model store on disk 🧹")
            }
            Err(_) => {
                log::error!("Failed to clean up temporary location for model store on disk");
            }
        }
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
    async fn add_model(&self, model_name: ModelName, _model_path: &str) -> anyhow::Result<()> {
        // Prepare the S3 key from model_name
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
                log::warn!("Failed to download object from s3 ⚠️: {}", e.to_string());
            }
        };

        // We will not be using the model_path from the request
        // as we have all the information to load the model
        let model_path = format!("{}/{}", self.model_store_dir, model_name);

        // Load the model into memory
        match extract_framework_from_path(model_path.clone()) {
            None => {
                anyhow::bail!("Failed to extract framework from path");
            }
            Some(framework) => match load_predictor(framework, model_path.as_str()) {
                Ok(predictor) => {
                    let now = Utc::now();
                    let model = Model::new(
                        predictor,
                        model_name.clone(),
                        framework,
                        model_path.to_string(),
                        now.to_rfc2822(),
                    );
                    self.models.insert(model_name, Arc::new(model));
                    Ok(())
                }
                Err(e) => {
                    anyhow::bail!("Failed to add new model: {e}")
                }
            },
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
    async fn update_model(&self, model_name: ModelName) -> anyhow::Result<()> {
        // Here model name will be the actual name and not the s3 key as used in add_model.
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

                        match load_predictor(model_framework, model_path) {
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
}

/// Fetches models from an S3 bucket, downloads them to a local directory, and loads them into memory.
///
/// This function first retrieves object keys from the S3 bucket, downloads corresponding objects,
/// saves and unpacks them into the specified local directory, and finally loads the models into memory.
///
/// # Arguments
///
/// * `client` - An `s3::Client` instance for interacting with AWS S3.
/// * `bucket_name` - The name of the S3 bucket from which to fetch models.
/// * `model_store_dir` - The local directory where downloaded models will be stored and loaded from.
///
/// # Returns
///
/// * `Result<DashMap<ModelName, Arc<Model>>>` - A `DashMap` containing loaded models mapped by their names,
///   or an error if fetching or loading models fails.
///
/// # Errors
///
/// This function will return an error if:
/// * The object keys cannot be retrieved from the S3 bucket.
/// * Objects cannot be downloaded from S3.
/// * Downloaded tarballs cannot be unpacked.
/// * Models cannot be loaded from the local directory.
///
async fn fetch_models(
    client: &s3::Client,
    bucket_name: String,
    model_store_dir: String,
) -> anyhow::Result<DashMap<ModelName, Arc<Model>>> {
    let keys = get_keys(client, bucket_name.clone()).await;

    match download_objects(client, bucket_name, keys.unwrap(), model_store_dir.as_str()).await {
        Ok(_) => {
            log::info!("Downloaded objects from s3 ✅")
        }
        Err(_) => {
            log::warn!("Failed to download objects from s3 ⚠️")
        }
    }

    let models = match load_models(model_store_dir) {
        Ok(models) => {
            log::info!("Successfully loaded models from directory ✅");
            models
        }
        Err(e) => {
            anyhow::bail!("Failed to load models - {}", e.to_string());
        }
    };

    Ok(models)
}

/// Retrieves object keys from an S3 bucket.
///
/// This function uses an `s3::Client` instance to list object keys from the specified S3 bucket.
///
/// # Arguments
///
/// * `client` - An `s3::Client` instance for interacting with AWS S3.
/// * `bucket_name` - The name of the S3 bucket from which to retrieve object keys.
///
/// # Returns
///
/// * `Result<Vec<String>>` - A vector containing object keys retrieved from the S3 bucket,
///   or an error if object keys cannot be retrieved.
///
/// # Errors
///
/// This function will return an error if:
/// * Object keys cannot be listed from the S3 bucket.
///
async fn get_keys(client: &s3::Client, bucket_name: String) -> anyhow::Result<Vec<String>> {
    let mut keys: Vec<String> = Vec::new();

    let mut response = client
        .list_objects_v2()
        .bucket(bucket_name.clone())
        .max_keys(10)
        .into_paginator()
        .send();

    while let Some(result) = response.next().await {
        match result {
            Ok(output) => match output.contents {
                None => {
                    log::warn!("No models found in the S3 bucket hence no models will be loaded ⚠️");
                }
                Some(objects) => {
                    for object in objects {
                        match object.key {
                            None => {
                                log::warn!("Object key is empty ⚠️");
                            }
                            Some(key) => {
                                keys.push(key);
                            }
                        }
                    }
                }
            },
            Err(e) => {
                anyhow::bail!(
                    "Failed to list objects in the {} bucket: {}",
                    bucket_name,
                    e.into_service_error()
                )
            }
        }
    }
    Ok(keys)
}

/// Downloads objects from an S3 bucket and saves them to a local directory.
///
/// This function downloads objects with specified keys from the S3 bucket using an `s3::Client` instance,
/// saves them to a temporary directory, and unpacks them into the specified output directory.
///
/// # Arguments
///
/// * `client` - An `s3::Client` instance for interacting with AWS S3.
/// * `bucket_name` - The name of the S3 bucket from which to download objects.
/// * `object_keys` - A vector of object keys to download from the S3 bucket.
/// * `out_dir` - The local directory where downloaded objects will be unpacked.
///
/// # Returns
///
/// * `Result<()>` - An empty result indicating success or an error.
///
/// # Errors
///
/// This function will return an error if:
/// * Objects cannot be downloaded from the S3 bucket.
/// * Downloaded tarballs cannot be saved or unpacked.
///
async fn download_objects(
    client: &s3::Client,
    bucket_name: String,
    object_keys: Vec<String>,
    out_dir: &str,
) -> anyhow::Result<()> {
    // Create a tempdir to hold downloaded models
    // This will be deleted when the program exits
    let dir = match tempfile::Builder::new().prefix("models").tempdir() {
        Ok(dir) => dir,
        Err(e) => {
            anyhow::bail!("Failed to create temporary directory ❌: {}", e.to_string());
        }
    };
    let temp_path = dir.path().to_str().unwrap();

    for object_key in object_keys {
        let response = client
            .get_object()
            .bucket(bucket_name.clone())
            .key(object_key.clone())
            .send()
            .await;

        match response {
            Ok(output) => {
                match output.body.collect().await {
                    Ok(data) => {
                        match save_and_upack_tarball(
                            temp_path,
                            object_key.clone(),
                            data.into_bytes(),
                            out_dir,
                        )
                        .await
                        {
                            Ok(_) => {
                                // Do nothing
                            }
                            Err(e) => {
                                log::warn!(
                                    "Failed to save artefact {} ⚠️: {}",
                                    object_key,
                                    e.to_string()
                                )
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "Failed to download artefact {} ⚠️: {}",
                            object_key,
                            e.to_string()
                        )
                    }
                }
            }
            Err(e) => {
                anyhow::bail!(
                    "Failed to get object key: {} from S3 ⚠️: {}",
                    object_key,
                    e.into_service_error()
                )
            }
        }
    }
    Ok(())
}

/// Saves and unpacks a tarball file into a specified output directory.
///
/// This function saves a tarball file received as bytes to a temporary location,
/// then unpacks it into the specified output directory.
///
/// # Arguments
///
/// * `path` - The temporary directory path where the tarball will be saved.
/// * `key` - The key or name of the tarball file.
/// * `data` - The tarball file data as bytes.
/// * `out_dir` - The output directory where the tarball will be unpacked.
///
/// # Returns
///
/// * `Result<()>` - An empty result indicating success or an error.
///
/// # Errors
///
/// This function will return an error if:
/// * The tarball file cannot be saved or created.
/// * The tarball file cannot be unpacked into the output directory.
///
async fn save_and_upack_tarball(
    path: &str,
    key: String,
    data: bytes::Bytes,
    out_dir: &str,
) -> anyhow::Result<()> {
    let file_path = Path::new(path).join(key);

    // Create parent directories if they do not exist
    if let Some(parent) = file_path.parent() {
        match tokio::fs::create_dir_all(parent).await {
            Ok(_) => {}
            Err(e) => {
                anyhow::bail!("Failed to create directory ⚠️: {}", e.to_string())
            }
        }
    }

    match tokio::fs::File::create(&file_path).await {
        Ok(mut file) => match file.write_all(&data).await {
            Ok(_) => {
                log::info!("Saved file to {:?}", file_path);
            }
            Err(e) => {
                anyhow::bail!(
                    "Failed to save file to {:?} ⚠️: {}",
                    file_path,
                    e.to_string()
                )
            }
        },
        Err(e) => {
            anyhow::bail!("Failed to create file {:?} ⚠️: {}", file_path, e.to_string())
        }
    }

    match unpack_tarball(file_path.to_str().unwrap(), out_dir) {
        Ok(_) => Ok(()),
        Err(e) => {
            anyhow::bail!("Failed to unpack ⚠️: {}", e.to_string())
        }
    }
}

/// Unpacks a `.tar.gz` file into a specified output directory.
///
/// This function opens a `.tar.gz` file located at `tarball_path`, extracts its contents,
/// and unpacks them into the directory specified by `out_dir`.
///
/// # Arguments
///
/// * `tarball_path` - The path to the `.tar.gz` file to unpack.
/// * `out_dir` - The directory where the contents of the `.tar.gz` file will be unpacked.
///
/// # Returns
///
/// * `Result<()>` - An empty result indicating success or an error.
///
/// # Errors
///
/// This function will return an error if:
/// * The `.tar.gz` file cannot be opened or read.
/// * The contents of the `.tar.gz` file cannot be unpacked into the output directory.
///
fn unpack_tarball(tarball_path: &str, out_dir: &str) -> anyhow::Result<()> {
    match File::open(tarball_path) {
        Ok(tar_gz) => {
            let tar = GzDecoder::new(tar_gz);
            let mut archive = Archive::new(tar);

            match archive.unpack(out_dir) {
                Ok(_) => {
                    log::info!(
                        "Unpacked tarball: {:?} at location: {}",
                        tarball_path,
                        out_dir
                    );
                    Ok(())
                }
                Err(_) => {
                    anyhow::bail!(
                        "Failed to unpack tarball ⚠️: {:?} at location: {}",
                        tarball_path,
                        out_dir
                    )
                }
            }
        }
        Err(e) => {
            anyhow::bail!("Failed to open tarball ⚠️: {}", e.to_string())
        }
    }
}

// todo: use Mocks
#[cfg(test)]
mod tests {
    // #[tokio::test]
    // async fn successfully_load_models_from_s3_model_store() {
    //     // Arrange
    //     let bucket_name = "jams-model-store".to_string();
    //
    //     // Act
    //     let model_store = S3ModelStore::new(bucket_name).await;
    //
    //     // Assert
    //     assert!(model_store.is_ok());
    //
    //     // Cleanup
    //     std::fs::remove_dir_all(model_store.unwrap().model_store_dir).unwrap();
    // }
}
