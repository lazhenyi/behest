//! Artifact store implementations using the `object_store` crate.
//!
//! Provides local filesystem and Amazon S3-compatible backends.

pub mod artifact;

pub use artifact::{DiskArtifactStore, S3ArtifactStore};
