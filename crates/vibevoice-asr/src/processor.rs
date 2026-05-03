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
    speech_start_id: u32,
    speech_pad_id: u32,
    speech_end_id: u32,
}

#[derive(Debug)]
pub struct VibeVoiceAsrInputs {
    pub prompt_token_ids: Vec<u32>,
    pub acoustic_input_mask: Vec<bool>,
    pub speech_tensor: Tensor,
    pub sample_rate: u32,
}

impl VibeVoiceAsrProcessor {
    pub fn new(tokenizer: Tokenizer, speech_tok_compress_ratio: usize) -> Result<Self> {
        let speech_start_id =
            token_id_any(&tokenizer, &["<|speech_start|>", "<|object_ref_start|>"])?;
        let speech_pad_id = token_id_any(&tokenizer, &["<|speech_pad|>", "<|box_start|>"])?;
        let speech_end_id =
            token_id_any(&tokenizer, &["<|speech_end|>", "<|object_ref_end|>"])?;
        Ok(Self {
            tokenizer,
            speech_tok_compress_ratio,
            speech_start_id,
            speech_pad_id,
            speech_end_id,
        })
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path).map_err(VibeVoiceAsrError::Tokenizer)?;
        Self::new(tokenizer, 3200)
    }

    pub fn prepare_audio_file(
        &self,
        path: impl AsRef<std::path::Path>,
        device: &Device,
        context_info: Option<&str>,
    ) -> Result<VibeVoiceAsrInputs> {
        let audio = load_audio_file(path, true)?;
        let speech_len = audio.samples.len();
        let speech_tensor = Tensor::from_vec(audio.samples, (1, 1, speech_len), device)?;
        self.prepare_audio_tensor(speech_tensor, speech_len, context_info)
    }

    pub fn prepare_audio_tensor(
        &self,
        speech_tensor: Tensor,
        speech_samples: usize,
        context_info: Option<&str>,
    ) -> Result<VibeVoiceAsrInputs> {
        let (batch, channels, _) = speech_tensor.dims3()?;
        if batch != 1 || channels != 1 {
            return Err(VibeVoiceAsrError::InvalidInput(format!(
                "expected speech tensor shape [1, 1, samples], got [{batch}, {channels}, ...]"
            )));
        }
        let vae_tok_len = speech_samples.div_ceil(self.speech_tok_compress_ratio);
        let (prompt_token_ids, acoustic_input_mask) =
            self.build_prompt(vae_tok_len, speech_samples as f32 / TARGET_SAMPLE_RATE as f32, context_info)?;

        Ok(VibeVoiceAsrInputs {
            prompt_token_ids,
            acoustic_input_mask,
            speech_tensor,
            sample_rate: TARGET_SAMPLE_RATE,
        })
    }

    pub fn build_prompt(
        &self,
        vae_tok_len: usize,
        audio_duration_secs: f32,
        context_info: Option<&str>,
    ) -> Result<(Vec<u32>, Vec<bool>)> {
        let system_prefix = format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        );
        let mut ids = self.tokenizer.encode(system_prefix, true)?.get_ids().to_vec();
        let mut mask = vec![false; ids.len()];

        ids.push(self.speech_start_id);
        mask.push(false);
        ids.extend(std::iter::repeat_n(self.speech_pad_id, vae_tok_len));
        mask.extend(std::iter::repeat_n(true, vae_tok_len));
        ids.push(self.speech_end_id);
        mask.push(false);

        let suffix = format!(
            "{}<|im_end|>\n<|im_start|>assistant\n",
            build_user_suffix(audio_duration_secs, context_info)
        );
        let suffix_ids = self.tokenizer.encode(suffix, true)?.get_ids().to_vec();
        mask.extend(std::iter::repeat_n(false, suffix_ids.len()));
        ids.extend(suffix_ids);
        Ok((ids, mask))
    }

    pub fn speech_tok_compress_ratio(&self) -> usize {
        self.speech_tok_compress_ratio
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

fn token_id_any(tokenizer: &Tokenizer, tokens: &[&str]) -> Result<u32> {
    for token in tokens {
        if let Some(id) = tokenizer
            .token_to_id(token)
            .or_else(|| tokenizer.get_vocab(true).get(*token).copied())
        {
            return Ok(id);
        }
    }
    Err(VibeVoiceAsrError::InvalidInput(format!(
        "missing tokenizer tokens {:?}",
        tokens
    )))
}

fn build_user_suffix(audio_duration_secs: f32, context_info: Option<&str>) -> String {
    let show_keys = "Start time, End time, Speaker ID, Content";
    match context_info {
        Some(context) if !context.trim().is_empty() => format!(
            " This is a {:.2} seconds audio, with extra info: {}\n\nPlease transcribe it with these keys: {}",
            audio_duration_secs,
            context.trim(),
            show_keys
        ),
        _ => format!(
            " This is a {:.2} seconds audio, please transcribe it with these keys: {}",
            audio_duration_secs,
            show_keys
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{
        AddedToken, Tokenizer, models::wordlevel::WordLevel,
        pre_tokenizers::whitespace::Whitespace,
    };

    fn test_tokenizer() -> Tokenizer {
        let vocab = [
            ("[UNK]", 0u32),
            ("You", 1),
            ("are", 2),
            ("a", 3),
            ("helpful", 4),
            ("assistant", 5),
            ("that", 6),
            ("transcribes", 7),
            ("audio", 8),
            ("input", 9),
            ("into", 10),
            ("text", 11),
            ("output", 12),
            ("in", 13),
            ("JSON", 14),
            ("format.", 15),
            ("This", 16),
            ("is", 17),
            ("0.50", 18),
            ("seconds", 19),
            ("please", 20),
            ("transcribe", 21),
            ("it", 22),
            ("with", 23),
            ("these", 24),
            ("keys:", 25),
            ("Start", 26),
            ("time,", 27),
            ("End", 28),
            ("Speaker", 29),
            ("ID,", 30),
            ("Content", 31),
            ("extra", 32),
            ("info:", 33),
            ("hotword", 34),
            ("<|speech_start|>", 35),
            ("<|speech_pad|>", 36),
            ("<|speech_end|>", 37),
            ("<|im_start|>", 38),
            ("<|im_end|>", 39),
            ("system", 40),
            ("user", 41),
            ("assistant\n", 42),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|speech_start|>", true),
            AddedToken::from("<|speech_pad|>", true),
            AddedToken::from("<|speech_end|>", true),
            AddedToken::from("<|im_start|>", true),
            AddedToken::from("<|im_end|>", true),
        ]);
        tokenizer
    }

    #[test]
    fn processor_builds_prompt_without_context() {
        let processor = VibeVoiceAsrProcessor::new(test_tokenizer(), 3200).unwrap();
        let (ids, mask) = processor.build_prompt(3, 0.5, None).unwrap();
        assert_eq!(ids.len(), mask.len());
        assert_eq!(mask.iter().filter(|v| **v).count(), 3);
    }

    #[test]
    fn processor_builds_prompt_with_context() {
        let processor = VibeVoiceAsrProcessor::new(test_tokenizer(), 3200).unwrap();
        let (ids, mask) = processor.build_prompt(2, 0.5, Some("hotword")).unwrap();
        assert_eq!(ids.len(), mask.len());
        assert_eq!(mask.iter().filter(|v| **v).count(), 2);
    }

    #[test]
    fn processor_validates_tensor_shape() {
        let processor = VibeVoiceAsrProcessor::new(test_tokenizer(), 3200).unwrap();
        let speech_tensor = Tensor::zeros((2, 1, 100), candle_core::DType::F32, &Device::Cpu).unwrap();
        let err = processor.prepare_audio_tensor(speech_tensor, 100, None).unwrap_err();
        assert!(matches!(err, VibeVoiceAsrError::InvalidInput(_)));
    }
}
