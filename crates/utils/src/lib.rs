mod build_info;
pub mod hooks;
pub mod keyed_lock;
pub mod logging;
pub mod macros;
pub mod shared_memory;
pub mod version;

use std::borrow::Cow;

use frida_gum::Error as FridaError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to find module for name `{0}`")]
    NoModuleName(Cow<'static, str>),

    #[error("Failed to find symbol for name `{0}`")]
    NoSymbolName(Cow<'static, str>),

    #[error("Frida failed with `{0}`")]
    Frida(FridaError),
}

impl From<FridaError> for Error {
    fn from(err: FridaError) -> Self {
        Error::Frida(err)
    }
}
