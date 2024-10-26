use flate2::read::GzDecoder;
use std::fs::{remove_dir_all, File};
use std::io::Write;
use std::path::Path;
use tar::Archive;

pub const DOWNLOADED_MODELS_DIRECTORY_NAME_PREFIX: &str = "model_store";

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
#[tracing::instrument(skip(path, key, data, out_dir))]
pub fn save_and_upack_tarball(
    path: &str,
    key: String,
    data: bytes::Bytes,
    out_dir: &str,
) -> anyhow::Result<()> {
    let file_path = Path::new(path).join(key);

    // Create parent directories if they do not exist
    if let Some(parent) = file_path.parent() {
        match std::fs::create_dir_all(parent) {
            Ok(_) => {}
            Err(e) => {
                anyhow::bail!("Failed to create directory ⚠️: {}", e.to_string())
            }
        }
    }

    match File::create(&file_path) {
        Ok(mut file) => match file.write_all(&data) {
            Ok(_) => {
                tracing::info!("Saved file to {:?}", file_path);
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
            anyhow::bail!(
                "Failed to create file {:?} ⚠️: {}",
                file_path,
                e.to_string()
            )
        }
    }

    let tarball_path = match file_path.to_str() {
        None => {
            anyhow::bail!("failed to convert file path to str ❌")
        }
        Some(path) => path,
    };

    match unpack_tarball(tarball_path, out_dir) {
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
#[tracing::instrument(skip(tarball_path, out_dir))]
pub fn unpack_tarball(tarball_path: &str, out_dir: &str) -> anyhow::Result<()> {
    match File::open(tarball_path) {
        Ok(tar_gz) => {
            let tar = GzDecoder::new(tar_gz);
            let mut archive = Archive::new(tar);

            match archive.unpack(out_dir) {
                Ok(_) => {
                    tracing::info!(
                        "Unpacked tarball: {:?} at location: {}",
                        tarball_path,
                        out_dir
                    );
                    Ok(())
                }
                Err(e) => {
                    anyhow::bail!(
                        "Failed to unpack tarball ⚠️: {:?} at location: {} - {}",
                        tarball_path,
                        out_dir,
                        e.to_string()
                    )
                }
            }
        }
        Err(e) => {
            anyhow::bail!("Failed to open tarball ⚠️: {}", e.to_string())
        }
    }
}

#[tracing::instrument]
pub fn cleanup(dir: String) {
    match remove_dir_all(dir) {
        Ok(_) => {
            tracing::info!("Cleaning up temporary location for model store on disk 🧹")
        }
        Err(_) => {
            tracing::error!("Failed to clean up temporary location for model store on disk");
        }
    }
}
