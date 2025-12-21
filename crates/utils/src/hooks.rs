use std::borrow::Cow;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ffi::CString;
use std::ops::Deref;
use std::sync::LazyLock;
use std::sync::OnceLock;

use frida_gum::interceptor::Interceptor;
pub use frida_gum::interceptor::InvocationContext;
pub use frida_gum::interceptor::InvocationListener;
pub use frida_gum::interceptor::Listener;
use frida_gum::Gum;
#[cfg(not(target_os = "linux"))]
use frida_gum::ModuleMap;
pub use frida_gum::NativePointer;

use crate::HookError;

/// 原始 dlsym 函数指针，用于避免递归调用
type DlsymFn = unsafe extern "C" fn(*const c_void, *const c_char) -> *const c_void;
static ORIGINAL_DLSYM: OnceLock<DlsymFn> = OnceLock::new();

/// 设置原始 dlsym 函数指针
pub fn set_original_dlsym(dlsym_fn: DlsymFn) {
    let _ = ORIGINAL_DLSYM.set(dlsym_fn);
}

static GUM: LazyLock<Gum> = LazyLock::new(Gum::obtain);

pub struct Hooker<'a> {
    interceptor: &'a mut Interceptor,
    #[allow(dead_code)] // 保留以兼容，但现在总是用全局查找
    module: Option<&'a str>,
}
impl Hooker<'_> {
    pub fn hook_export(
        &mut self,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, HookError> {
        // 完全绕过 Frida 的模块查找，直接用 dlsym(RTLD_DEFAULT)
        // 原因：
        // 1. Module::load() 会触发 gum_module_load -> gum_module_registry_obtain -> module registry activation
        // 2. Module::find_global_export_by_name() 也可能内部触发 module registry
        // 3. 在 dlopen 上下文中调用 module registry 会导致 dl_iterate_phdr 重入死锁
        //
        // 解决方案：用原生 dlsym，优先使用提供的原始 dlsym（避免触发 dlsym hook）
        let function = unsafe {
            let symbol_cstr = CString::new(symbol).map_err(|_| {
                HookError::NoSymbolName(Cow::Owned(symbol.to_string()))
            })?;
            
            // 优先使用原始 dlsym（如果已设置），否则使用 libc::dlsym
            let sym_ptr = if let Some(original_dlsym) = ORIGINAL_DLSYM.get() {
                // 使用提供的原始 dlsym，不会触发 dlsym hook
                original_dlsym(libc::RTLD_DEFAULT, symbol_cstr.as_ptr())
            } else {
                // Fallback 到 libc::dlsym（可能会触发 dlsym hook，但有递归保护）
                libc::dlsym(libc::RTLD_DEFAULT, symbol_cstr.as_ptr())
            };
            
            if sym_ptr.is_null() {
                return Err(HookError::NoSymbolName(Cow::Owned(symbol.to_string())));
            }
            
            NativePointer(sym_ptr as *mut c_void)
        };

        tracing::debug!(
            "Found function at {:p} for symbol {}, calling interceptor.replace",
            function.0,
            symbol
        );
        let result = self
            .interceptor
            .replace(
                function,
                NativePointer(detour),
                NativePointer(std::ptr::null_mut()),
            )
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
}

/// Struct for managing the hooks using Frida.
pub struct HookManager {
    interceptor: Interceptor,
    pub module_names: Vec<String>,
}

fn find_module_by_prefix(prefix: &str) -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let process = match procfs::process::Process::myself() {
            Ok(p) => p,
            Err(e) => {
                tracing::error!("Failed to access /proc/self: {}", e);
                return None;
            }
        };

        let maps = match process.maps() {
            Ok(m) => m,
            Err(e) => {
                tracing::error!("Failed to read /proc/self/maps: {}", e);
                return None;
            }
        };

        for map in maps {
            // Skip non-path mappings (e.g., anonymous, heap, stack)
            let procfs::process::MMapPath::Path(path) = map.pathname else {
                continue;
            };

            // Skip if no filename
            let Some(filename) = path.file_name() else {
                continue;
            };

            // Skip if filename is not valid UTF-8
            let Some(filename_str) = filename.to_str() else {
                continue;
            };

            // Check if this library matches our prefix
            if filename_str.starts_with(prefix) {
                tracing::debug!(
                    "Found module '{}' at path: {}",
                    filename_str,
                    path.display()
                );
                return Some(filename_str.to_string());
            }
        }

        tracing::debug!(
            "Module with prefix '{}' not found in process memory maps",
            prefix
        );
        None
    }

    #[cfg(not(target_os = "linux"))]
    {
        // On non-Linux platforms, fall back to Frida's module enumeration
        // This is less safe but necessary for cross-platform support
        let mut module_map = ModuleMap::new();
        module_map.update();

        for module in module_map.values() {
            let name = module.name();
            if name.starts_with(prefix) {
                tracing::debug!("Found module via Frida: {}", name);
                return Some(name.to_string());
            }
        }

        tracing::debug!("Module with prefix '{}' not found via Frida", prefix);
        None
    }
}

pub fn is_module_loaded(prefix: &str) -> bool {
    find_module_by_prefix(prefix).is_some()
}

impl HookManager {
    pub fn hooker<'a>(&'a mut self, module: Option<&'a str>) -> Result<Hooker<'a>, HookError> {
        let module = if let Some(module_prefix) = module {
            match find_module_by_prefix(module_prefix) {
                Some(name) => {
                    tracing::debug!("Found module: {}", name);
                    // Store it if not already present
                    if !self.module_names.contains(&name) {
                        self.module_names.push(name);
                    }
                    // Return reference to the stored string
                    self.module_names
                        .iter()
                        .find(|m| m.starts_with(module_prefix))
                        .map(|s| s.as_str())
                }
                None => {
                    return Err(HookError::NoModuleName(Cow::Owned(
                        module_prefix.to_string(),
                    )))
                }
            }
        } else {
            None
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
    ) -> Result<NativePointer, HookError> {
        self.hooker(module)?.hook_export(symbol, detour)
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
