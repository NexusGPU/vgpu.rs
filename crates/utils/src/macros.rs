#[macro_export]
macro_rules! replace {
    ($hooker:expr, $func:expr, $detour:expr) => {{
        match $hooker.hook_export_fast($func, $detour as *mut std::ffi::c_void) {
            Ok(_) => {
                tracing::trace!("hooked {:?}", $func);
            }
            Err(err) => {
                tracing::debug!("hook {:?} failed with err {err:?}", $func);
            }
        }
    }};
}

#[macro_export]
macro_rules! replace_symbol {
    ($hook_manager:expr, $mod:expr, $func:expr, $detour_function:expr, $detour_type:ty, $hook_fn:expr) => {{
        let intercept = |hook_manager: &mut $crate::hooks::HookManager,
                         symbol_name,
                         detour: $detour_type|
         -> Result<$detour_type, $crate::Error> {
            let replaced = hook_manager
                .hook_export($mod, symbol_name, detour as *mut std::ffi::c_void)?
                .0;
            let original_fn: $detour_type = std::mem::transmute(replaced);

            tracing::trace!("hooked {symbol_name:?}");
            Ok(original_fn)
        };

        let _ = intercept($hook_manager, $func, $detour_function)
            .and_then(|hooked| Ok($hook_fn.set(hooked).unwrap()));
    }};
}
