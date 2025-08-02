# Rust Error Handling Practices

## Error Handling Basics

- Apply the fail-fast principle: functions should return early on errors instead of nesting logic or continuing execution after an error has occurred.
- Use early return Err(...) or ? for propagating errors to simplify control flow and improve readability.
- Avoid deeply nested if let or match blocks for error handling—prefer early exits.
- Use error-stack (`Report<MyError>`) instead of anyhow or eyre for error handling
- Use `Box<dyn Error>` only in tests or prototyping code
- Use concrete error types with `Report<E>`, not `Report<Box<dyn Error>>`
- Use error-stack macros for early returns:
  - `ensure!(condition, MyError::SomeVariant)` (from `error_stack::ensure`)
  - `bail!(MyError::SomeVariant)`
- Import `Error` from `core::error::` instead of `std::error::`

## Error Context Enhancement

- Use `change_context()` to map error types consistently
- Ensure that errors include sufficient context for debugging
- Add `attach_printable()` or `attach_printable_lazy()` to include relevant debug information
- When reporting errors to users, provide actionable guidance where possible

## Specialized Error Handling

### Streaming Decoders

- Return `Ok(None)` for incomplete data rather than an error
- Return errors only for actual error conditions like malformed data
- Document buffer management behavior in error cases

### Async Operations

- Propagate errors with proper context across async boundaries
- Consider using structured error types for complex async workflows
- Design error types to preserve causal relationships across async tasks

## Error Documentation

For comprehensive guidance on documenting errors, refer to the [Rust Documentation Practices](mdc:.cursor/rules/rust-documentation.mdc) file.

Key points to remember:

- Document errors with an "# Errors" section for all fallible functions
- Link error variants with proper reference syntax
- Document potential panics with a "# Panics" section

## Testing Error Conditions

For guidance on testing error conditions, refer to the [Rust Testing Strategy](mdc:.cursor/rules/rust-testing-strategy.mdc) file.

Key points:

- Write tests for each error case in your code
- Assert on specific error types or message contents
- Use `expect_err()` with clear messages following the "should..." format
