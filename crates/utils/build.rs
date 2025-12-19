use std::env;

fn main() {
    if let Err(err) = emit_git_variables() {
        println!("cargo:warning=vergen: {err}");
    }

    let version = env::var("VERSION").unwrap_or_else(|_| "latest".to_string());
    println!("cargo:rustc-env=IMAGE_VERSION={version}");

    
    // Add library search path for frida-gum
    println!("cargo:rustc-link-search=native=/usr/local/lib");
}

fn emit_git_variables() -> anyhow::Result<()> {
    let mut builder = vergen_git2::Git2Builder::default();

    builder.branch(true);
    builder.commit_message(true);
    builder.describe(true, true, None);
    builder.sha(true);
    builder.dirty(true);

    let git2 = builder.build()?;

    vergen_git2::Emitter::default()
        .fail_on_error()
        .add_instructions(&git2)?
        .emit()
}
