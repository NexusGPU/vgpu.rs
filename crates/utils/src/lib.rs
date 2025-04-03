pub mod hooks;
pub mod logging;
pub mod macros;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to find module for name `{0}`")]
    NoModuleName(String),

    #[error("Failed to find symbol for name `{0}`")]
    NoSymbolName(String),

    #[error("Frida failed with `{0}`")]
    Frida(frida_gum::Error),
}

impl From<frida_gum::Error> for Error {
    fn from(err: frida_gum::Error) -> Self {
        Error::Frida(err)
    }
}
