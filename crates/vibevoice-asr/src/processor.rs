use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use vibevoice_core::{TARGET_SAMPLE_RATE, load_audio_file};

use crate::error::{Result, VibeVoiceAsrError};

pub const SYSTEM_PROMPT: &str =
    "You are a helpful assistant that transcribes audio input into text output in JSON format.";

#[derive(Debug)]
pub struct VibeVoiceAsrProcessor {
    tokenizer: Tokenizer,
    speech_tok_compress_ratio: usize,
}

#[derive(Debug)]
pub struct VibeVoiceAsrInputs {
    pub prompt_token_ids: Vec<u32>,
    pub speech_tensor: Tensor,
    pub sample_rate: u32,
}

impl VibeVoiceAsrProcessor {
    pub fn new(tokenizer: Tokenizer, speech_tok_compress_ratio: usize) -> Self {
        Self {
            tokenizer,
            speech_tok_compress_ratio,
        }
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(VibeVoiceAsrError::Tokenizer)?;
        Ok(Self::new(tokenizer, 3200))
    }

    pub fn prepare_audio_file(
        &self,
        path: impl AsRef<std::path::Path>,
        device: &Device,
        context_info: Option<&str>,
    ) -> Result<VibeVoiceAsrInputs> {
        let audio = load_audio_file(path, true)?;
        let prompt = match context_info {
            Some(context) if !context.is_empty() => {
                format!("{SYSTEM_PROMPT}\nContext: {context}\n<|speech_start|>")
            }
            _ => format!("{SYSTEM_PROMPT}\n<|speech_start|>"),
        };
        let encoding = self.tokenizer.encode(prompt, true)?;
        let n_samples = audio.samples.len();
        let speech_tensor = Tensor::from_vec(
            audio.samples,
            (1, 1, n_samples),
            device,
        )?;

        Ok(VibeVoiceAsrInputs {
            prompt_token_ids: encoding.get_ids().to_vec(),
            speech_tensor,
            sample_rate: TARGET_SAMPLE_RATE,
        })
    }

    pub fn speech_tok_compress_ratio(&self) -> usize {
        self.speech_tok_compress_ratio
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

