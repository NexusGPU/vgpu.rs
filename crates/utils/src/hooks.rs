use std::borrow::Cow;
use std::ffi::c_void;
use std::ops::Deref;
use std::sync::LazyLock;
use std::sync::OnceLock;

use frida_gum::interceptor::Interceptor;
use frida_gum::Gum;
use frida_gum::Module;
use frida_gum::ModuleMap;
use frida_gum::NativePointer;

use crate::Error;

static GUM: LazyLock<Gum> = LazyLock::new(Gum::obtain);

pub struct Hooker<'a> {
    interceptor: &'a mut Interceptor,
    module: Option<&'a str>,
}
impl Hooker<'_> {
    pub fn hook_export(
        &mut self,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, Error> {
        tracing::debug!("Starting hook_export for symbol: {}", symbol);
        let function = if let Some(module_name) = self.module {
            tracing::debug!("Loading module {} for symbol {}", module_name, symbol);
            Module::load(&GUM, module_name).find_export_by_name(symbol)
        } else {
            tracing::debug!("Finding global export for symbol {}", symbol);
            Module::find_global_export_by_name(symbol)
        }
        .ok_or_else(|| Error::NoSymbolName(Cow::Owned(symbol.to_string())))?;
        
        tracing::debug!("Found function at {:p} for symbol {}, calling interceptor.replace", function.0, symbol);
        let result = self.interceptor
            .replace(
                function,
                NativePointer(detour),
                NativePointer(std::ptr::null_mut()),
            )
            .map_err(Into::into);
        
        match &result {
            Ok(original) => tracing::debug!("Successfully hooked symbol {}, original at {:p}", symbol, original.0),
            Err(e) => tracing::error!("Failed to hook symbol {}: {:?}", symbol, e),
        }
        
        result
    }
}

/// Struct for managing the hooks using Frida.
pub struct HookManager {
    interceptor: Interceptor,
    pub module_names: Vec<String>,
}

impl HookManager {
    pub fn collect_module_names(&mut self) {
        let mut module_map = ModuleMap::new();
        module_map.update();
        self.module_names = module_map
            .values()
            .iter()
            .filter(|m| !m.path().starts_with("/tensor-fusion"))
            .map(|m| m.name().to_string())
            .collect();
        // sort by length to avoid matching a longer module name as a substring of a shorter one
        self.module_names
            .sort_by_key(|b| std::cmp::Reverse(b.len()));
    }

    pub fn hooker<'a>(&'a mut self, module: Option<&'a str>) -> Result<Hooker<'a>, Error> {
        let found_module =
            module.map(|m: &str| self.module_names.iter().find(|x| x.starts_with(m)));

        let module = match found_module {
            Some(None) => return Err(Error::NoModuleName(Cow::Owned(module.unwrap().to_string()))),
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
}

impl Default for HookManager {
    fn default() -> Self {
        let mut interceptor = Interceptor::obtain(&GUM);
        tracing::debug!("Starting interceptor transaction");
        interceptor.begin_transaction();
        Self {
            interceptor,
            module_names: vec![],
        }
    }
}

impl Drop for HookManager {
    fn drop(&mut self) {
        tracing::debug!("Ending interceptor transaction");
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
