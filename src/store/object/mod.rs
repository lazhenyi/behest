//! Object store artifact implementations (local disk and S3).

pub mod artifact;

pub use artifact::{DiskArtifactStore, S3ArtifactStore};
