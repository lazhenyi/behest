//! MongoDB session store implementation using document-per-session model.

pub mod session;

pub use session::MongodbSessionStore;
