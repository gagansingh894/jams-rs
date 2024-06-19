use std::fs::{File, remove_dir_all};
use std::sync::Arc;
use dashmap::DashMap;
use flate2::read::GzDecoder;
use tar::Archive;
use crate::model_store::storage::{Model, ModelName};

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
pub fn unpack_tarball(tarball_path: &str, out_dir: &str) -> anyhow::Result<()> {
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

pub fn cleanup(dir: String) {
    match remove_dir_all(dir) {
        Ok(_) => {
            log::info!("Cleaning up temporary location for model store on disk 🧹")
        }
        Err(_) => {
            log::error!("Failed to clean up temporary location for model store on disk");
        }
    }
}