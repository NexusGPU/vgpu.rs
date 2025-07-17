fn main() {
    tonic_build::configure()
        .compile_protos(&["protos/v1beta1/api.proto"], &["protos/v1beta1/"])
        .unwrap();
}
