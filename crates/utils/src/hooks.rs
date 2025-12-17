use std::borrow::Cow;
use std::ffi::c_void;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;
use std::sync::OnceLock;

use frida_gum::interceptor::Interceptor;
pub use frida_gum::interceptor::InvocationContext;
pub use frida_gum::interceptor::InvocationListener;
pub use frida_gum::interceptor::Listener;
use frida_gum::Gum;
use frida_gum::Module;
#[cfg(not(target_os = "linux"))]
use frida_gum::ModuleMap;
pub use frida_gum::NativePointer;
use frida_gum::Process;

use crate::HookError;

static GUM: LazyLock<Gum> = LazyLock::new(Gum::obtain);

// Flag to ensure module registry is synchronized only once
static MODULE_REGISTRY_SYNCED: AtomicBool = AtomicBool::new(false);

/// Force Frida's module registry to synchronize by enumerating all modules.
/// This must be called before any hooks are installed to prevent crashes in
/// multiprocess environments (e.g., Python multiprocessing spawn mode).
///
/// The function intentionally leaks the Process and Module objects to ensure
/// Frida's internal state remains valid for the entire process lifetime.
fn ensure_module_registry_synced() {
    if MODULE_REGISTRY_SYNCED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
        .is_ok()
    {
        // First, warm up dl_iterate_phdr directly via libc before Frida touches it.
        // This ensures the dynamic linker's internal state is fully initialized.
        #[cfg(target_os = "linux")]
        unsafe {
            extern "C" fn count_modules(
                _info: *mut libc::dl_phdr_info,
                _size: libc::size_t,
                data: *mut libc::c_void,
            ) -> libc::c_int {
                let count = data as *mut usize;
                *count += 1;
                0
            }
            let mut module_count: usize = 0;
            libc::dl_iterate_phdr(
                Some(count_modules),
                &mut module_count as *mut usize as *mut libc::c_void,
            );
            tracing::debug!("Pre-warmed dl_iterate_phdr, found {} modules", module_count);
        }

        // Memory fence to ensure all dl_iterate_phdr operations complete
        std::sync::atomic::fence(Ordering::SeqCst);

        // Now let Frida do its module enumeration
        let process = Process::obtain(&GUM);
        let modules = process.enumerate_modules();

        tracing::debug!(
            "Frida module registry synchronized with {} modules",
            modules.len()
        );

        // Intentionally leak these to keep Frida's internal state valid.
        // This prevents crashes when hooked functions are called later.
        std::mem::forget(modules);
        std::mem::forget(process);

        // Final memory fence
        std::sync::atomic::fence(Ordering::SeqCst);

        // Force actual I/O operations to ensure all system resources are initialized.
        // This mimics what trace logging does when enabled.
        // Writing to /dev/null triggers real system calls without visible output.
        #[cfg(target_os = "linux")]
        {
            use std::fs::OpenOptions;
            use std::io::Write;
            if let Ok(mut f) = OpenOptions::new().write(true).open("/dev/null") {
                let _ = f.write_all(b"warmup");
                let _ = f.flush();
            }
        }
    }
}

pub struct Hooker<'a> {
    interceptor: &'a mut Interceptor,
    module: Option<&'a str>,
}
impl Hooker<'_> {
    pub fn hook_export(
        &mut self,
        symbol: &str,
        detour: *mut c_void,
    ) -> Result<NativePointer, HookError> {
        let function = if let Some(module_name) = self.module {
            Module::load(&GUM, module_name).find_export_by_name(symbol)
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
        // Ensure Frida's module registry is synchronized before any hooks are installed
        ensure_module_registry_synced();

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
