use std::{fs, path::{Path, PathBuf}};

use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};
use candle_transformers::models::qwen2;
use tokenizers::Tokenizer;
use vibevoice_core::{DTypeName, Qwen2DecoderConfig, VibeVoiceASRConfig};
use vibevoice_tokenizer::{VibeVoiceAcousticTokenizerModel, VibeVoiceSemanticTokenizerModel};

use crate::error::{Result, VibeVoiceAsrError};

#[derive(Debug)]
pub struct SpeechConnector {
    fc1: Linear,
    fc2: Linear,
}

impl SpeechConnector {
    pub fn load(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(input_dim, output_dim, vb.pp("fc1"))?,
            fc2: linear(output_dim, output_dim, vb.pp("fc2"))?,
        })
    }

    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(features)?;
        Ok(self.fc2.forward(&xs)?)
    }
}

#[derive(Debug)]
pub struct VibeVoiceAsrModel {
    pub config: VibeVoiceASRConfig,
    pub acoustic_tokenizer: VibeVoiceAcousticTokenizerModel,
    pub semantic_tokenizer: VibeVoiceSemanticTokenizerModel,
    pub acoustic_connector: SpeechConnector,
    pub semantic_connector: SpeechConnector,
    pub decoder: qwen2::ModelForCausalLM,
    pub decoder_tokenizer: Tokenizer,
    pub device: Device,
}

#[derive(Debug)]
pub struct VibeVoiceAsrSession {
    pub acoustic_features: Tensor,
    pub semantic_features: Tensor,
}

impl VibeVoiceAsrModel {
    pub fn from_local_dir(
        model_dir: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        device: Device,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let config = VibeVoiceASRConfig::from_path(model_dir.join("config.json"))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(VibeVoiceAsrError::Tokenizer)?;
        let dtype = config
            .torch_dtype
            .or(config.decoder_config.torch_dtype)
            .unwrap_or(DTypeName::F32)
            .into_candle();

        let weight_files = find_safetensors(model_dir)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };
        let acoustic_tokenizer =
            VibeVoiceAcousticTokenizerModel::load(&config.acoustic_tokenizer_config, vb.pp("acoustic_tokenizer"))?;
        let semantic_tokenizer =
            VibeVoiceSemanticTokenizerModel::load(&config.semantic_tokenizer_config, vb.pp("semantic_tokenizer"))?;
        let acoustic_connector = SpeechConnector::load(
            config.acoustic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp("acoustic_connector"),
        )?;
        let semantic_connector = SpeechConnector::load(
            config.semantic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp("semantic_connector"),
        )?;
        let decoder_cfg = to_candle_qwen2_config(&config.decoder_config)?;
        let decoder = qwen2::ModelForCausalLM::new(&decoder_cfg, vb.pp("language_model"))?;

        Ok(Self {
            config,
            acoustic_tokenizer,
            semantic_tokenizer,
            acoustic_connector,
            semantic_connector,
            decoder,
            decoder_tokenizer: tokenizer,
            device,
        })
    }

    pub fn encode_speech(&self, speech_tensor: &Tensor) -> Result<VibeVoiceAsrSession> {
        let acoustic_latents = self.acoustic_tokenizer.encode(speech_tensor)?.sample();
        let acoustic_features = self
            .acoustic_connector
            .forward(&transpose_bct_to_btc(&acoustic_latents)?)?;

        let semantic_latents = self.semantic_tokenizer.encode(speech_tensor)?.sample();
        let semantic_features = self
            .semantic_connector
            .forward(&transpose_bct_to_btc(&semantic_latents)?)?;

        Ok(VibeVoiceAsrSession {
            acoustic_features,
            semantic_features,
        })
    }

    pub fn transcribe(
        &mut self,
        prompt_token_ids: &[u32],
        speech_tensor: &Tensor,
        max_new_tokens: usize,
    ) -> Result<String> {
        let _ = prompt_token_ids;
        let _speech = self.encode_speech(speech_tensor)?;
        if max_new_tokens == 0 {
            return Ok(String::new());
        }

        Err(VibeVoiceAsrError::Unsupported(
            "speech-conditioned decoding is not wired yet because Candle's stock Qwen2 path only accepts token ids; this crate already loads the tokenizer encoders/connectors and the decoder, but the multimodal `inputs_embeds` bridge still needs a local Qwen2 fork".to_string(),
        ))
    }
}

fn to_candle_qwen2_config(cfg: &Qwen2DecoderConfig) -> Result<qwen2::Config> {
    Ok(qwen2::Config {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        max_position_embeddings: cfg.max_position_embeddings,
        sliding_window: cfg.sliding_window,
        max_window_layers: cfg.max_window_layers,
        tie_word_embeddings: cfg.tie_word_embeddings,
        rope_theta: cfg.rope_theta,
        rms_norm_eps: cfg.rms_norm_eps,
        use_sliding_window: cfg.use_sliding_window,
        hidden_act: match cfg.hidden_act.as_str() {
            "silu" => candle_nn::Activation::Silu,
            other => {
                return Err(VibeVoiceAsrError::Unsupported(format!(
                    "unsupported qwen2 hidden activation: {other}"
                )))
            }
        },
    })
}

fn transpose_bct_to_btc(xs: &Tensor) -> Result<Tensor> {
    Ok(xs.transpose(1, 2)?)
}

fn find_safetensors(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok().map(|v| v.path()))
        .filter(|path| {
            path.extension()
                .and_then(|v| v.to_str())
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(VibeVoiceAsrError::InvalidInput(format!(
            "no .safetensors files found in {}",
            dir.display()
        )));
    }
    Ok(files)
}
