use crate::Error;
use frida_gum::{interceptor::Interceptor, Gum, Module, NativePointer};
use std::{
    ffi::c_void,
    ops::Deref,
    sync::{LazyLock, OnceLock},
};

static GUM: LazyLock<Gum> = LazyLock::new(Gum::obtain);

pub struct Hooker<'a> {
    interceptor: &'a mut Interceptor,
    module: Option<&'a str>,
}
impl<'a> Hooker<'a> {
    pub fn hook_export(
        &mut self,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, Error> {
        let function = Module::obtain(&GUM)
            .find_export_by_name(self.module, symbol)
            .ok_or_else(|| Error::NoSymbolName(symbol.to_string()))?;
        self.interceptor
            .replace(
                function,
                NativePointer(detour),
                NativePointer(std::ptr::null_mut()),
            )
            .map_err(Into::into)
    }

    pub fn hook_export_fast(
        &mut self,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, Error> {
        let function = Module::obtain(&GUM)
            .find_export_by_name(self.module, symbol)
            .ok_or_else(|| Error::NoSymbolName(symbol.to_string()))?;

        // we use `replace_fast` since we don't use the original function.
        self.interceptor
            .replace_fast(function, NativePointer(detour))
            .map_err(Into::into)
    }
}

/// Struct for managing the hooks using Frida.
pub struct HookManager {
    interceptor: Interceptor,
    pub module_names: Vec<String>,
}

impl HookManager {
    pub fn collect_module_names(&mut self) {
        self.module_names = Module::obtain(&GUM)
            .enumerate_modules()
            .iter()
            .map(|m| m.name.clone())
            .collect();
        // sort by length to avoid matching a longer module name as a substring of a shorter one
        self.module_names
            .sort_by_key(|b| std::cmp::Reverse(b.len()));
    }

    pub fn hooker<'a>(&'a mut self, module: Option<&'a str>) -> Result<Hooker<'a>, Error> {
        let found_module =
            module.map(|m: &str| self.module_names.iter().find(|x| x.starts_with(m)));

        let module = match found_module {
            Some(None) => return Err(Error::NoModuleName(module.unwrap().to_string())),
            Some(m) => m.map(|s| s.as_str()),
            None => None,
        };

        tracing::debug!("start hook module: {:?}", module);

        Ok(Hooker {
            interceptor: &mut self.interceptor,
            module,
        })
    }

    pub fn hook_export(
        &mut self,
        module: Option<&str>,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, Error> {
        self.hooker(module)?.hook_export(symbol, detour)
    }

    #[allow(dead_code)]
    pub fn hook_export_fast(
        &mut self,
        module: Option<&str>,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, Error> {
        self.hooker(module)?.hook_export_fast(symbol, detour)
    }
}

impl Default for HookManager {
    fn default() -> Self {
        let mut interceptor = Interceptor::obtain(&GUM);
        interceptor.begin_transaction();
        Self {
            interceptor,
            module_names: vec![],
        }
    }
}

impl Drop for HookManager {
    fn drop(&mut self) {
        self.interceptor.end_transaction()
    }
}

#[derive(Debug)]
pub struct HookFn<T>(OnceLock<T>);

impl<T> Deref for HookFn<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0.get().unwrap()
    }
}

impl<T> HookFn<T> {
    /// Helper function to set the inner [`OnceLock`] `T` of `self`.
    pub fn set(&self, value: T) -> Result<(), T> {
        self.0.set(value)
    }

    /// Until we can impl Default as const.
    pub const fn default_const() -> Self {
        Self(OnceLock::new())
    }
}
