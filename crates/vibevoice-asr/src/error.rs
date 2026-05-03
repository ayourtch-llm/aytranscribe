use thiserror::Error;

#[derive(Debug, Error)]
pub enum VibeVoiceAsrError {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),

    #[error(transparent)]
    Core(#[from] vibevoice_core::VibeVoiceCoreError),

    #[error(transparent)]
    HfHub(#[from] hf_hub::api::sync::ApiError),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Tokenizer(#[from] tokenizers::Error),

    #[error(transparent)]
    TokenizerModel(#[from] vibevoice_tokenizer::VibeVoiceTokenizerError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, VibeVoiceAsrError>;
