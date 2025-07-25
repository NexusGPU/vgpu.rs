#![warn(clippy::indexing_slicing)]

use proc_macro2::Span;
use quote::quote;
use syn::Ident;
use syn::ItemFn;
use syn::Type;

/// `#[hook_fn]` annotates the C ffi functions (mirrord's `_detour`s), and is used to generate the
/// following boilerplate (using `close_detour` as an example):
///
/// 1. `type FnClose = unsafe extern "C" fn(c_int) -> c_int`;
/// 2. `static FN_CLOSE: HookFn<FnClose> = HookFn(OnceLock::new())`;
///
/// Where (1) is the type alias of the ffi function, and (2) is where we'll store the original
/// function after calling replace with frida. `HookFn` is defined in `mirrord-layer` as a newtype
/// wrapper around `std::sync::Oncelock`.
///
///
/// The visibility of both (1) and (2) are based on the visibility of the annotated function.
///
/// `_args`: So far we're just ignoring this.
///
/// `input`: The ffi function, including docstrings, and other annotations.
///
/// -> Returns the `input` function with no loss of information (keeps docstrings and other
/// annotations intact).
#[proc_macro_attribute]
pub fn hook_fn(
    _args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let output: proc_macro2::TokenStream = {
        let proper_function = syn::parse_macro_input!(input as ItemFn);

        let signature = &proper_function.sig;
        let visibility = &proper_function.vis;

        let ident_string = signature.ident.to_string();
        let type_name = ident_string.split("_detour").next().map(|fn_name| {
            let (uppercase, lowercase) = fn_name.split_at(1);
            let name = format!("Fn{}{}", uppercase.to_uppercase(), lowercase);
            Ident::new(&name, Span::call_site())
        });

        let static_name = ident_string.split("_detour").next().map(|fn_name| {
            let name = format!("FN_{}", fn_name.to_uppercase());
            Ident::new(&name, Span::call_site())
        });

        let unsafety = signature.unsafety;
        let abi = signature.abi.clone();

        // Function arguments without taking into account variadics!
        let mut fn_args = signature
            .inputs
            .iter()
            .map(|fn_arg| match fn_arg {
                syn::FnArg::Receiver(_) => panic!("Hooks should not take any form of `self`!"),
                syn::FnArg::Typed(arg) => arg.ty.clone(),
            })
            .collect::<Vec<_>>();

        // If we have `VaListImpl` args, then we push it to the end of the `fn_args` as
        // just `...`.
        if signature.variadic.is_some() {
            let fixed_arg = quote! {
                ...
            };

            fn_args.push(Box::new(Type::Verbatim(fixed_arg)));
        }

        let return_type = &signature.output;

        // `unsafe extern "C" fn(i32) -> i32`
        let bare_fn = quote! {
            #unsafety #abi fn(#(#fn_args),*) #return_type
        };

        // `pub(crate) type FnClose = unsafe extern "C" fn(i32) -> i32`
        let type_alias = quote! {
            #visibility type #type_name = #bare_fn
        };

        // `pub(crate) static FN_CLOSE: HookFn<FnClose> = HookFn::default()`
        let original_fn = quote! {
            #visibility static #static_name: utils::hooks::HookFn<#type_name> =
                utils::hooks::HookFn::default_const()
        };

        // Extract the function name
        let fn_name = &signature.ident;

        // Create a modified function with tracing at the beginning
        let mut modified_function = proper_function.clone();

        // Access the block through the reference
        let block = &modified_function.block;

        // Create a new body with tracing at the beginning - just log the function name
        let new_body = quote! {
            {
                tracing::debug!(target: "hook_fn", "Called {}", stringify!(#fn_name));
                #block
            }
        };

        // Parse the new body
        if let Ok(parsed_block) = syn::parse2::<syn::Block>(new_body) {
            modified_function.block = Box::new(parsed_block);
        }

        let output = quote! {
            #[allow(non_camel_case_types)]
            #type_alias;

            #[allow(non_upper_case_globals)]
            #original_fn;

            #[allow(non_upper_case_globals)]
            #modified_function
        };

        output
    };

    // Here we return the equivalent of (1) and (2) for the ffi function, plus the annotated
    // function we received as `input`.
    proc_macro::TokenStream::from(output)
}
