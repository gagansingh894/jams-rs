use crate::model_store::common::save_and_upack_tarball;
use azure_storage_blobs::prelude::ContainerClient;
use bytes::Bytes;
use futures::StreamExt;

/// Asynchronously downloads a blob from Azure Blob Storage, saves it to a temporary path, and unpacks it.
///
/// This function performs the following steps:
/// 1. Creates a `BlobClient` for the specified blob.
/// 2. Streams the blob data in chunks and collects it into a complete byte vector.
/// 3. Saves the collected data to a temporary path and unpacks it into the specified model storage directory.
///
/// # Arguments
///
/// * `client` - A reference to an Azure Blob Storage `ContainerClient`.
/// * `blob_name` - A `String` specifying the name of the blob to be downloaded.
/// * `temp_path` - A `String` specifying the temporary directory path where the blob will be saved.
/// * `model_store_dir` - A `String` specifying the directory where the unpacked model will be stored.
///
/// # Returns
///
/// A `Result` which is `Ok(())` if the operation is successful, or an `anyhow::Error` if the operation fails.
///
/// # Errors
///
/// This function will return an error if:
/// * Streaming or collecting the blob data fails.
/// * Saving and unpacking the blob data fails.
pub async fn download_blob(
    client: &ContainerClient,
    blob_name: String,
    model_store_dir: String,
) -> anyhow::Result<()> {
    let temp_path = match tempfile::Builder::new().prefix("models").tempdir() {
        Ok(dir) => match dir.path().to_str() {
            None => {
                anyhow::bail!("Failed to convert TempDir to String ❌")
            }
            Some(p) => p.to_string(),
        },
        Err(e) => {
            anyhow::bail!("Failed to create temp directory: {}", e)
        }
    };
    let temp_save_path = format!("{}/{}", temp_path, blob_name.clone());

    let blob_client = client.blob_client(blob_name.clone());
    let mut blob_stream = blob_client.get().into_stream();
    let mut complete_response: Vec<u8> = vec![];
    while let Some(value) = blob_stream.next().await {
        let data = match value {
            Ok(response) => match response.data.collect().await {
                Ok(data) => data.to_vec(),
                Err(e) => {
                    anyhow::bail!("Failed to convert bytes: {}", e)
                }
            },
            Err(e) => {
                anyhow::bail!("Failed to collect data to bytes: {}", e)
            }
        };
        complete_response.extend(&data);
    }

    match save_and_upack_tarball(
        temp_save_path.as_str(),
        blob_name.clone(),
        Bytes::copy_from_slice(&complete_response[..]),
        model_store_dir.as_str(),
    ) {
        Ok(_) => {
            // Do nothing
        }
        Err(e) => {
            tracing::warn!(
                "Failed to save artefact {} ⚠️: {}",
                blob_name,
                e.to_string()
            )
        }
    }

    Ok(())
}
