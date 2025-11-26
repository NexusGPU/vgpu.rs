fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .build_server(false) // We only need the client
        .compile_protos(&["proto/pod_resources.proto"], &["proto"])?;

    tonic_prost_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&["proto/device_plugin.proto"], &["proto"])?;
    Ok(())
}
