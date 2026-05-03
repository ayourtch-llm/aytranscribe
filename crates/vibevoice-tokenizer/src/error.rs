use thiserror::Error;

#[derive(Debug, Error)]
pub enum VibeVoiceTokenizerError {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Core(#[from] vibevoice_core::VibeVoiceCoreError),

    #[error("unsupported operation: {0}")]
    Unsupported(&'static str),

    #[error("invalid shape: {0}")]
    InvalidShape(String),
}

pub type Result<T> = std::result::Result<T, VibeVoiceTokenizerError>;

