use aws_sdk_s3::client as s3;

use crate::model_store::common::save_and_upack_tarball;

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
#[tracing::instrument(skip(client, object_keys, out_dir))]
pub async fn download_objects(
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
            tracing::error!("Failed to create temporary directory ❌: {}", e.to_string());
            anyhow::bail!("Failed to create temporary directory ❌: {}", e.to_string());
        }
    };
    let temp_path = match dir.path().to_str() {
        None => {
            tracing::error!("failed to convert path to str ❌");
            anyhow::bail!("failed to convert path to str ❌")
        }
        Some(path) => path,
    };

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
                        ) {
                            Ok(_) => {
                                // Do nothing
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to save artefact {} ⚠️: {}",
                                    object_key,
                                    e.to_string()
                                )
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to download artefact {} ⚠️: {}",
                            object_key,
                            e.to_string()
                        )
                    }
                }
            }
            Err(e) => {
                tracing::error!(
                    "Failed to get object key: {} from S3 ⚠️: {}",
                    object_key,
                    e.into_service_error()
                );
                anyhow::bail!("Failed to get object key: {} from S3 ⚠️", object_key,)
            }
        }
    }
    Ok(())
}
