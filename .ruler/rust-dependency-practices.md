1. If a dependency is used in multiple crates in the workspace, move it into the `[workspace.dependencies]` section of the top-level Cargo.toml if it's not already there.

2. In the crate's own `[dependencies]`, replace the full dependency declaration with a workspace reference:

```toml
dep_name = { workspace = true }
```
3. Do NOT include features, optional, default-features, or other configuration fields in the [workspace.dependencies]. These settings should remain in the individual crates.
4. Preserve crate-specific features and customizations within each crate's local dependency entry.
5. Do not overwrite existing [workspace.dependencies] entries unless necessary for consistency.
6. Always ensure that the dependency version used in [workspace.dependencies] is the latest available compatible version published on crates.io, following semantic versioning rules. If a newer version is available, update it.

