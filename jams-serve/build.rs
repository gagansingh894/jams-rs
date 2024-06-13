fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::compile_protos("proto/api/v1/jams.proto")?;
    Ok(())
}
