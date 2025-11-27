# Rust coding guidelines

* Prioritize code correctness and clarity. Speed and efficiency are secondary priorities unless otherwise specified.
* Do not write organizational or comments that summarize the code. Comments should only be written in order to explain "why" the code is written in some way in the case there is a reason that is tricky / non-obvious.
* Prefer implementing functionality in existing files unless it is a new logical component. Avoid creating many small files.
* Avoid using functions that panic like `unwrap()`, instead use mechanisms like `?` to propagate errors.
* Be careful with operations like indexing which may panic if the indexes are out of bounds.
* Never silently discard errors with `let _ =` on fallible operations. Always handle errors appropriately:
  - Propagate errors with `?` when the calling function should handle them
  - Use `.log_err()` or similar when you need to ignore errors but want visibility
  - Use explicit error handling with `match` or `if let Err(...)` when you need custom logic
  - Example: avoid `let _ = client.request(...).await?;` - use `client.request(...).await?;` instead
* When implementing async operations that may fail, ensure errors propagate to the UI layer so users get meaningful feedback.
* Never create files with `mod.rs` paths - prefer `src/some_module.rs` instead of `src/some_module/mod.rs`.
* When creating new crates, prefer specifying the library root path in `Cargo.toml` using `[lib] path = "...rs"` instead of the default `lib.rs`, to maintain consistent and descriptive naming (e.g., `gpui.rs` or `main.rs`).
* Avoid creative additions unless explicitly requested
* Use full words for variable names (no abbreviations like "q" for "queue")
* Use variable shadowing to scope clones in async contexts for clarity, minimizing the lifetime of borrowed references.
  Example:
  ```rust
  executor.spawn({
      let task_ran = task_ran.clone();
      async move {
          *task_ran.borrow_mut() = true;
      }
  });
  ```

# Code-generation rules
When generating Rust code, you must always follow these rules:

## **Project Structure**

* Keep code simple and avoid over-engineering
* Keep modules cohesive and loosely coupled
* Avoid unnecessary traits, generics, or abstractions
* Prefer reusing existing crates instead of writing utilities by hand

## **Rust Idioms**

* Avoid unnecessary `clone`, `Arc`, `String`, `to_string`
* Prefer zero-cost abstractions
* Prefer borrowing (`&T`, `&str`, `Cow`)
* Prefer iterators and combinators
* Avoid `unwrap()` / `expect()`; use `?` and proper error types
* Use `Result` / `Option` to express correct semantics

## **Naming Rules**

* Names must be clear, meaningful, consistent, domain-driven
* Consistent naming across the project (files, modules, functions, parameters)
* Asynchronous functions must follow a consistent naming rule (`_async` or uniformly omitted)
* Parameter names must be globally consistent (e.g., always use `user_id`, not `id` in one file and `uid` in another)

## **Avoid Duplicate Code**

* Do not repeat logic
* Extract common behavior into helpers
* Centralize reusable functions

## **Avoid Reinventing the Wheel**

Before writing new code, check if the functionality exists in crates such as:

* `serde`, `anyhow`, `thiserror`, `regex`, `chrono`, `serde_json`, `tokio`, etc.
  If so, prefer the crate.
  Only implement manually when necessary.

## **Comments**

* Code should be self-explanatory
* Comment only when explaining intent (“why”), not behavior (“what”)

## **Tests**

* Write only meaningful tests
* Test critical paths, boundaries, and error behavior
* Do not test trivial getters/setters or obvious logic
* Do not couple tests to internal implementation
