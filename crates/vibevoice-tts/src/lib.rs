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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_model_constructs() {
        let model = VibeVoiceTtsModel::new(VibeVoiceConfig::default());
        assert_eq!(model.config.decoder_config.hidden_size, 3584);
    }

    #[test]
    fn tts_model_returns_not_implemented() {
        let model = VibeVoiceTtsModel::new(VibeVoiceConfig::default());
        assert!(matches!(model.synthesize("hello"), Err(VibeVoiceTtsError::NotImplemented)));
    }
}
