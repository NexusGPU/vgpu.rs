use std::borrow::Cow;
use std::ffi::c_void;
use std::ops::Deref;
use std::ptr::null_mut;
use std::sync::LazyLock;
use std::sync::OnceLock;

use frida_gum::interceptor::Interceptor;
pub use frida_gum::interceptor::InvocationContext;
pub use frida_gum::interceptor::InvocationListener;
pub use frida_gum::interceptor::Listener;
pub use frida_gum::NativePointer;
use frida_gum::{Gum, Module, Process};

use crate::HookError;

static GUM: LazyLock<Gum> = LazyLock::new(Gum::obtain);

/// Check if a module with the given prefix is loaded
pub fn is_module_loaded(prefix: &str) -> bool {
    let process = Process::obtain(&GUM);
    let modules = process.enumerate_modules();
    modules.iter().any(|m| m.name().starts_with(prefix))
}

/// Struct for managing the hooks using Frida.
pub struct HookManager<'a> {
    interceptor: Interceptor,
    modules: Vec<Module>,
    #[allow(dead_code)]
    process: Process<'a>,
}

impl<'a> HookManager<'a> {
    /// Hook the first function exported from a lib that is in modules and is hooked successfully
    pub fn hook_any_lib_export(
        &mut self,
        symbol: &str,
        detour: *mut c_void,
        filter: Option<&str>,
    ) -> Result<NativePointer, HookError> {
        for module in &self.modules {
            // In this case we only want libs, no "main binaries"
            let module_name = module.name();
            if !module_name.starts_with(filter.unwrap_or("lib")) {
                continue;
            }

            if let Some(function) = module.find_export_by_name(symbol) {
                tracing::trace!("found {symbol:?} in {module_name:?}, hooking");
                match self.interceptor.replace(
                    function,
                    NativePointer(detour),
                    NativePointer(null_mut()),
                ) {
                    Ok(original) => return Ok(original),
                    Err(err) => {
                        tracing::trace!(
                            "hook {symbol:?} in {module_name:?} failed with err {err:?}"
                        )
                    }
                }
            }
        }
        Err(HookError::NoSymbolName(Cow::Owned(symbol.to_string())))
    }

    /// Hook an exported symbol, suitable for most libc use cases.
    /// If it fails to hook the first one found, it will try to hook each matching export
    /// until it succeeds.
    pub fn hook_export_or_any(
        &mut self,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, HookError> {
        // First try to hook the default exported one, if it fails, fallback to first lib that
        // provides it.
        let function = Module::find_global_export_by_name(symbol);
        match function {
            Some(func) => self
                .interceptor
                .replace(func, NativePointer(detour), NativePointer(null_mut()))
                .or_else(|_| self.hook_any_lib_export(symbol, detour, None)),
            None => self.hook_any_lib_export(symbol, detour, None),
        }}

    /// Hook an export from a specific module or globally if module is None
    pub fn hook_export(
        &mut self,
        module: Option<&str>,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, HookError> {
        let function = if let Some(module_name) = module {
            self.modules
                .iter()
                .find(|m| m.name() == module_name)
                .and_then(|m| m.find_export_by_name(symbol))
        } else {
            Module::find_global_export_by_name(symbol)
        }
        .ok_or_else(|| HookError::NoSymbolName(Cow::Owned(symbol.to_string())))?;

        tracing::debug!(
            "Found function at {:p} for symbol {}, calling interceptor.replace",
            function.0,
            symbol
        );
        let result = self
            .interceptor
            .replace(function, NativePointer(detour), NativePointer(null_mut()))
            .map_err(Into::into);

        match &result {
            Ok(original) => tracing::debug!(
                "Successfully hooked symbol {}, original at {:p}",
                symbol,
                original.0
            ),
            Err(e) => tracing::error!("Failed to hook symbol {}: {:?}", symbol, e),
        }

        result
    }

    #[cfg(target_os = "linux")]
    /// Hook a symbol in the first module (main module, binary)
    pub fn hook_symbol_main_module(
        &mut self,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, HookError> {
        let function = self
            .process
            .main_module
            .find_symbol_by_name(symbol)
            .ok_or_else(|| HookError::NoSymbolName(Cow::Owned(symbol.to_string())))?;

        // on Go we use `replace_fast` since we don't use the original function.
        self.interceptor
            .replace_fast(function, NativePointer(detour))
            .map_err(Into::into)
    }

    /// Resolve symbol in main module
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    pub fn resolve_symbol_main_module(&self, symbol: &str) -> Option<NativePointer> {
        self.process.main_module.find_symbol_by_name(symbol)
    }

    /// Resolve symbol in the given module
    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    pub fn resolve_symbol_in_module(
        &self,
        module_name: &str,
        symbol: &str,
    ) -> Option<NativePointer> {
        let Some(module) = self.modules.iter().find(|m| m.name() == module_name) else {
            tracing::trace!(module_name, "Module not found");
            return None;
        };
        module.find_symbol_by_name(symbol)
    }

    #[cfg(all(
        target_os = "linux",
        any(target_arch = "x86_64", target_arch = "aarch64")
    ))]
    pub fn hook_symbol_in_module(
        &mut self,
        module: &str,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, HookError> {
        let Some(module) = self.modules.iter().find(|m| m.name() == module) else {
            return Err(HookError::NoModuleName(Cow::Owned(module.to_string())));
        };

        let function = module
            .find_symbol_by_name(symbol)
            .ok_or_else(|| HookError::NoSymbolName(Cow::Owned(symbol.to_string())))?;

        // on Go we use `replace_fast` since we don't use the original function.
        self.interceptor
            .replace_fast(function, NativePointer(detour))
            .map_err(Into::into)
    }

    pub fn attach<I: InvocationListener>(
        &mut self,
        function: NativePointer,
        listener: &mut I,
    ) -> Result<Listener, HookError> {
        self.interceptor
            .attach(function, listener)
            .map_err(Into::into)
    }

    pub fn detach(&mut self, listener: Listener) {
        self.interceptor.detach(listener);
    }

    /// Get module names currently loaded
    pub fn module_names(&self) -> Vec<String> {
        self.modules.iter().map(|m| m.name().to_string()).collect()
    }

    /// Check if a module with the given prefix is loaded
    pub fn is_module_loaded(&self, prefix: &str) -> bool {
        self.modules.iter().any(|m| m.name().starts_with(prefix))
    }
}

impl<'a> Default for HookManager<'a> {
    fn default() -> Self {
        let mut interceptor = Interceptor::obtain(&GUM);
        interceptor.begin_transaction();
        let process = Process::obtain(&GUM);
        let modules = process.enumerate_modules();
        Self {
            interceptor,
            modules,
            process,
        }
    }
}

impl<'a> Drop for HookManager<'a> {
    fn drop(&mut self) {
        self.interceptor.end_transaction()
    }
}

#[derive(Debug)]
pub struct HookFn<T>(OnceLock<T>);

impl<T> Deref for HookFn<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.0
            .get()
            .expect("hook function not initialized before use")
    }
}

impl<T> HookFn<T> {
    /// Helper function to set the inner [`OnceLock`] `T` of `self`.
    pub fn set(&self, value: T) -> Result<(), T> {
        self.0.set(value)
    }

    pub fn get(&self) -> Option<&T> {
        self.0.get()
    }

    pub fn is_none(&self) -> bool {
        self.0.get().is_none()
    }

    /// Until we can impl Default as const.
    pub const fn default_const() -> Self {
        Self(OnceLock::new())
    }
}
