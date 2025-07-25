# Rust Tracing Practices

## Event Recording

- Use level-specific macros for appropriate visibility: `info!`, `debug!`, `trace!`, `warn!`, `error!`
- Provide structured data as named arguments to facilitate automated processing and filtering
- Format log messages consistently, beginning with verbs in present tense
- Include relevant context through key-value pairs:

```rust
tracing::info!(user_id = user.id, action = "login", "User logged in successfully");
tracing::error!(error = %e, user_id = user.id, "Failed to authenticate user");
```

## Instrumentation

- Apply the `#[tracing::instrument]` attribute to functions to automatically create trace spans
- Include the `err` parameter to capture error returns in the span
- Exclude sensitive or large parameters using `skip(password)` or `skip_all`
- Add custom context fields with `fields(request_id = req.id())`

```rust
#[tracing::instrument(level = "debug", err, skip(password), fields(user_id = user.id))]
async fn authenticate_user(user: &User, password: &[u8]) -> Result<AuthToken, AuthError> {
    // Implementation
}
```
