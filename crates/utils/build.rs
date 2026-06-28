use std::env;

fn main() {
    if let Err(err) = emit_git_variables() {
        println!("cargo:warning=vergen: {err}");
    }

    let version = env::var("VERSION").unwrap_or_else(|_| "latest".to_string());
    println!("cargo:rustc-env=IMAGE_VERSION={version}");
}

fn emit_git_variables() -> anyhow::Result<()> {
    let git2 = vergen_git2::Git2::builder()
        .branch(true)
        .commit_message(true)
        .describe(true, true, None)
        .sha(true)
        .dirty(true)
        .build();

    vergen_git2::Emitter::default()
        .add_instructions(&git2)?
        .emit()
}
