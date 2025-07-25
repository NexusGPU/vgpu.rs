# Rust Coding Style

## Project-Specific Patterns

- Use the 2024 edition of Rust
- Prefer `derive_more` over manual trait implementations
- Feature flags in this codebase use the `#[cfg(feature = "...")]` pattern
- Invoke `cargo clippy` with `--all-features`, `--all-targets`, and `--no-deps` from the root
- Use `cargo doc --no-deps --all-features` for checking documentation
- Use `rustfmt` to format the code
- Use `#[expect(lint, reason = "...")]` over `#[allow(lint)]`
### dependencies
When modifying Cargo.toml for a crate that is part of a Cargo workspace, always check if the dependency can be moved to the root-level [workspace.dependencies].
If the dependency already exists there, reuse it and avoid redeclaring version or source information in the individual crate.
If it does not exist yet, add the dependency to [workspace.dependencies] at the workspace root.
⚠️ Note: workspace.dependencies does not support setting features, default-features, or optional.
These configurations must still be specified in the local crate's [dependencies] section using workspace = true
In short, centralize dependency source/version management in the workspace root, but configure features locally.

## Type System

- Create strong types with newtype patterns for domain entities
- Implement `hash_graph_types` traits for custom domain types
- Consider visibility carefully (avoid unnecessary `pub`)

```rust
#[derive(Debug, Copy, Clone, Eq, Hash, PartialEq, derive_more::Display)]
pub struct UserId(Uuid);
```

### When implementing methods for a generic struct in Rust, prefer placing methods that do not
depend on trait bounds in impl blocks without where clauses.

Purpose: This improves code readability, avoids unnecessary repetition of bounds, and helps separate general methods from those requiring specific constraints.
```rust
struct MyStruct<T> {
    value: T,
}

// Methods without trait bounds go here
impl<T> MyStruct<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

// Methods that require trait bounds go here
impl<T> MyStruct<T>
where
    T: std::fmt::Display,
{
    pub fn print(&self) {
        println!("{}", self.value);
    }
}
```
## Async Patterns

- Use `impl Future<Output = T> + Send` in trait definitions:

```rust
fn get_data(
    &self,
    id: String,
) -> impl Future<Output = Result<Data, Report<DataError>>> + Send {
    async move {
        // Implementation
    }
}
```

## Function Arguments

- Functions should **never** take more than 7 arguments. If a function requires more than 7 arguments, encapsulate related parameters in a struct.
- Functions that use data immutably should take a reference to the data, while functions that modify data should take a mutable reference. Never take ownership of data unless the function explicitly consumes it.
- Make functions `const` whenever possible.
- Prefer the following argument types when applicable, but only if this does not reduce performance:
  - `impl AsRef<str>` instead of `&str` or `&String`
  - `impl AsRef<Path>` instead of `&Path` or `&PathBuf`
  - `impl IntoIterator<Item = &T>` when only iterating over the data
  - `&[T]` instead of `&Vec<T>`
  - `&mut [T]` instead of `&mut Vec<T>` when the function doesn't need to resize the vector
  - `impl Into<Cow<T>>` instead of `Cow<T>`
  - `impl Into<Arc<T>>` instead of `Arc<T>`
  - `impl Into<Rc<T>>` instead of `Rc<T>`
  - `impl Into<Box<T>>` instead of `Box<T>`
- Never use `impl Into<Option<_>>` as from reading the caller site, it's not visible that `None` could potentially be passed

## `From` and `Into`

- Generally prefer `From` implementations over `Into` implementations. The Rust compiler will automatically derive `Into` from `From`, but not vice versa.
- When converting between types, prefer using the `from` method over `into` for clarity. The `from` method makes the target type explicit in the code, while `into` requires type inference.
- For wrapper types like `Cow`, `Arc`, `Rc`, `Report`, and `Box`, prefer using explicit constructors (e.g., `Cow::from`, `Arc::new`) instead of `.into()`. This improves readability by clearly indicating the target type.

## Smart Pointers

- When cloning smart pointers such as `Arc` and `Rc`, **always** use `Arc::clone(&pointer)` and `Rc::clone(&pointer)` instead of `pointer.clone()`. This explicitly indicates you're cloning the reference, not the underlying data.

## Instrumentation

- Annotate functions that perform significant work with `#[tracing::instrument]`
- Use `tracing` macros (e.g., `trace!`, `debug!`, `info!`, `warn!`, `error!`) instead of `println!` or `eprintln!` for logging

## Allocations

- Minimize allocations when possible. For example, reuse a `Vec` in a loop instead of creating a new one in each iteration.
- Prefer borrowed data over owned data where appropriate.
- Balance performance and readability—if an allocation makes code significantly more readable or maintainable, the trade-off may be worthwhile.

## Types

- Use newtypes when a value should carry specific semantics beyond its underlying type. This improves type safety and code clarity.

For example:

```rust
struct UserId(u64);  // instead of `type UserId = u64;` or `u64`
```

## Naming Conventions

When suggesting names for variables, functions, or types:

- Do not prefix test-function names with `test_`, this would otherwise result in `test::test_<name>` names.
- Provide a concise list of naming options with brief explanations of why each fits the context
- Choose names of appropriate length—avoid names that are too long or too short
- Avoid abbreviations unless they are widely recognized in the domain (e.g., `Http` or `Json` is acceptable, but `Ctx` instead of `Context` is not)
- Do not suffix names with their types (e.g., use `users` instead of `usersList`)
- Do not repeat the type name in variable names (e.g., use `user` instead of `userUser`)

## Crate Preferences

- Use `similar_asserts` for test assertions
- Use `insta` for snapshot tests
- Use `test_log` for better test output (`#[test_log::test]`)
- Use `tracing` macros, not `log` macros
- Prefer `tracing::instrument` for function instrumentation

## Import Style

- Don't use local imports within functions, or blocks
- Avoid wildcard imports like `use super::*;`, or `use crate::module::*;`
- Never use a prelude `use crate::prelude::*`
- Prefer explicit imports to make dependencies clear and improve code readability
- We prefer `core` over `alloc` over `std` for imports to minimize dependencies
  - Use `core` for functionality that doesn't require allocation
  - Use `alloc` when you need allocation but not OS-specific features
  - Only use `std` when necessary for OS interactions or when using `core`/`alloc` would be unnecessarily complex
- Prefer qualified imports (`use foo::Bar; let x = Bar::new()`) over fully qualified paths (`let x = foo::Bar::new()`) for frequently used types
- Use `pub use` re-exports in module roots to create a clean public API
- Avoid importing items with the same name from different modules; use qualified imports
- Import traits using `use module::Trait as _;` when you only need the trait's methods and not the trait name itself
  - This pattern brings trait methods into scope without name conflicts
  - Use this especially for extension traits or when implementing foreign traits on local types

```rust
// Good - Importing a trait just for its methods:
use std::io::Read as _;

// Example with trait methods:
fn read_file(file: &mut File) -> Result<String, std::io::Error> {
    // Read methods available without importing the Read trait name
    let mut content = String::new();
    file.read_to_string(&mut content)?;
    Ok(content)
}

// Bad - Directly importing trait when only methods are needed:
use std::io::Read;

// Good - Importing trait for implementing it:
use std::io::Write;
impl Write for MyWriter { /* implementation */ }

// Bad - Wildcard import:
mod tests {
    use super::*;  // Wildcard import

    #[test]
    fn test_something() {
        // Test implementation
    }
}

// Good - Explicit imports:
mod tests {
    use crate::MyStruct;
    use crate::my_function;

    #[test]
    fn test_something() {
        // Test implementation
    }
}

// Bad - Local import:
fn process_data() {
    use std::collections::HashMap;  // Local import
    let map = HashMap::new();
    // Implementation
}

// Good - Module-level import:
use std::collections::HashMap;

fn process_data() {
    let map = HashMap::new();
    // Implementation
}

// Bad - Using std when core would suffice:
use std::fmt::Display;

// Good - Using core for non-allocating functionality:
use core::fmt::Display;

// Bad - Using std when alloc would suffice:
use std::collections::BTreeSet;

// Good - Using alloc for allocation without full std dependency:
use alloc::vec::Vec;

// Appropriate - Using std when needed:
use std::fs::File;  // OS-specific functionality requires std
```

## Libraries and Components

- Abstract integrations with third-party systems behind traits to maintain clean separation of concerns

## Comments and Assertions

- Do not add comments after a line of code; place comments on separate lines above the code they describe
- When using assertions, include descriptive messages using the optional description parameter rather than adding a comment
- All `expect()` messages should follow the format "should ..." to clearly indicate the expected behavior

For example:

```rust
// Bad:
assert_eq!(result, expected); // This should match the expected value

// Good:
assert_eq!(result, expected, "Values should match expected output");

// Bad:
some_value.expect("The value is not None"); // This should never happen

// Good:
some_value.expect("should contain a valid value");
```