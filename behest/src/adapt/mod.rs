//! Concrete provider adapters.

#[cfg(feature = "anthropic")]
pub mod anthropic {
    //! Anthropic provider adapter.

    pub use behest_adapter_anthropic::*;
}

#[cfg(feature = "openai")]
pub mod openai {
    //! OpenAI-compatible provider adapters.

    pub use behest_adapter_openai::*;
}
