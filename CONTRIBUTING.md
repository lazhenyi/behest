# Contributing

Thanks for helping improve `agents`.

## Development loop

1. Keep changes focused and atomic.
2. Add or update tests for behavior changes.
3. Run the full local check before opening a pull request:

```bash
cargo fmt --all --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-features --locked
cargo doc --all-features --no-deps --locked
```

## Code style

- Prefer typed errors over stringly errors.
- Avoid `unwrap` and `expect` outside tests and examples.
- Keep public APIs documented.
- Keep provider adapters behind feature flags when they add heavy dependencies.

## Commit style

Use Conventional Commits, for example:

- `feat(provider): add streaming adapter contract`
- `fix(registry): return unsupported for missing providers`
- `docs: document feature flags`
