use thiserror::Error;

use vibevoice_core::VibeVoiceConfig;

#[derive(Debug, Error)]
pub enum VibeVoiceTtsError {
    #[error("tts is not implemented yet")]
    NotImplemented,
}

pub type Result<T> = std::result::Result<T, VibeVoiceTtsError>;

#[derive(Debug)]
pub struct VibeVoiceTtsModel {
    pub config: VibeVoiceConfig,
}

impl VibeVoiceTtsModel {
    pub fn new(config: VibeVoiceConfig) -> Self {
        Self { config }
    }

    pub fn synthesize(&self, _text: &str) -> Result<Vec<f32>> {
        Err(VibeVoiceTtsError::NotImplemented)
    }
}

