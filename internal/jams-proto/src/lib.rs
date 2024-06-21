// load proto gen code as module
pub mod jams_v1 {
    tonic::include_proto!("jams_v1");

    pub const FILE_DESCRIPTOR_SET: &[u8] =
        tonic::include_file_descriptor_set!("jams_v1_descriptor");
}