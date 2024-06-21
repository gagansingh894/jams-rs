use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    tonic_build::configure()
        .file_descriptor_set_path(out_dir.join("jams_v1_descriptor.bin"))
        .compile(&["proto/api/v1/jams.proto"], &["jams_v1"])
        .unwrap();

    tonic_build::compile_protos("proto/api/v1/jams.proto")?;
    Ok(())
}
