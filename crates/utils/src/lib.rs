mod build_info;
pub mod hooks;
pub mod logging;
pub mod macros;
pub mod shared_memory;
pub mod version;

use std::borrow::Cow;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to find module for name `{0}`")]
    NoModuleName(Cow<'static, str>),

    #[error("Failed to find symbol for name `{0}`")]
    NoSymbolName(Cow<'static, str>),

    #[error("Frida failed with `{0}`")]
    Frida(frida_gum::Error),
}

impl From<frida_gum::Error> for Error {
    fn from(err: frida_gum::Error) -> Self {
        Error::Frida(err)
    }
}
