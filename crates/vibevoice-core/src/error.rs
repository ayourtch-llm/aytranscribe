use thiserror::Error;

pub type Result<T> = std::result::Result<T, VibeVoiceCoreError>;

#[derive(Debug, Error)]
pub enum VibeVoiceCoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("tensor error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("huggingface hub error: {0}")]
    HfHub(#[from] hf_hub::api::sync::ApiError),

    #[error("audio decode error: {0}")]
    AudioDecode(String),

    #[error("unsupported configuration: {0}")]
    UnsupportedConfig(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),
}

