# Rust Testing Strategy

## Verification Commands

- When modifying Rust code, run clippy to detect issues:

  ```bash
  cargo clippy --all-features --all-targets --workspace --no-deps
  ```

- If changes affect the `hash-graph-api` crate or any representation influencing the public interface, regenerate the OpenAPI specifications:

  ```bash
  cd libs/@local/graph/api
  cargo run --bin openapi-spec-generator
  ```

- Use `cargo doc --no-deps --all-features` to check documentation

## Test Execution

- Use `cargo-nextest` for running unit and integration tests
- Use the default test runner for documentation tests
- For comprehensive verification, run the complete test suite
- Database seeding can be performed using yarn commands in the project's package.json

## Test Design Principles

- Test both happy paths and error conditions for each function
- For each error case in your code, write a corresponding test
- Test boundary conditions and edge cases explicitly
- Include tests for invalid or malformed inputs
- For streaming encoders/decoders, test partial data handling and buffer management
- Aim for high test coverage but prioritize test quality over quantity
- Structure tests following the Arrange-Act-Assert pattern

## Assertion Standards

- Use descriptive assertion messages that explain the expected behavior
- All assertion messages (including `expect()`, `unwrap()` and `assert*()`) should follow the "should..." format:

  ```rust
  // Good examples:
  value.expect("should contain a valid configuration");
  assert_eq!(result, expected, "Result should match the expected value");
  ```

- Use `expect()` or `expect_err()` with clear messages instead of `unwrap()` or `unwrap_err()`
- Prefer `assert_eq!` with custom messages over bare assertions when comparing values
- When testing errors, assert on specific error types or message contents, not just that an error occurred
- Balance assertions to verify functionality without creating brittle tests

## Test Organization

- Group related tests into appropriate modules
- Use descriptive test names that explain the test scenario and expected outcome
- Do not prefix test function names with `test_` (avoid `test::test_name` patterns)
- Use helper functions to avoid code duplication in tests
- Consider using parameterized tests for testing similar functionality with different inputs

## Test Code Quality

- Follow the same code quality standards in test code as in production code
- Add appropriate assertions for array/slice access to avoid clippy warnings
- Document test scenarios with clear comments explaining:
  - The setup (input and environment)
  - The action being tested
  - The expected outcome
  - Why the outcome is expected
- Consider adding custom test utilities to simplify common testing patterns
- Use the `json!` macro from `serde_json` instead of constructing JSON as raw strings

```rust
// Bad:
let json_str = "{\"name\":\"value\",\"nested\":{\"key\":42}}";

// Good:
use serde_json::json;
let json_value = json!({
    "name": "value",
    "nested": {
        "key": 42
    }
});
```

## Integration With Other Practices

- For documentation practices in tests, refer to the [Rust Documentation Practices](mdc:.cursor/rules/rust-documentation.mdc) file
- For error handling in tests, refer to the [Rust Error Handling Practices](mdc:.cursor/rules/rust-error-handling.mdc) file