extern crate cbindgen;

use std::env;

fn main() -> Result<(), cbindgen::Error> {
    let crate_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR env var is not defined");

    let config = cbindgen::Config::from_file("./cbindgen.toml")
        .expect("Unable to find cbindgen.toml configuration file");

    cbindgen::generate_with_config(&crate_dir, config)?.write_to_file("./include/cuda_limiter.h");

    Ok(())
}
