use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde::Deserialize;
use tokenizers::Tokenizer;
use vibevoice_core::{DTypeName, Qwen2DecoderConfig, VibeVoiceASRConfig};
use vibevoice_tokenizer::{VibeVoiceAcousticTokenizerModel, VibeVoiceSemanticTokenizerModel};

use crate::{
    error::{Result, VibeVoiceAsrError},
    processor::{VibeVoiceAsrInputs, VibeVoiceAsrProcessor},
    qwen2::{self, RmsNorm},
};

pub const DEFAULT_MODEL_REPO: &str = "microsoft/VibeVoice-ASR-HF";

#[derive(Debug)]
pub struct SpeechConnector {
    fc1: Linear,
    norm: RmsNorm,
    fc2: Linear,
}

impl SpeechConnector {
    pub fn load(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(input_dim, output_dim, vb.pp("fc1"))?,
            norm: RmsNorm::new(output_dim, 1e-6, vb.pp("norm"))?,
            fc2: linear(output_dim, output_dim, vb.pp("fc2"))?,
        })
    }

    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(features)?;
        let xs = self.norm.forward(&xs)?;
        Ok(self.fc2.forward(&xs)?)
    }

    pub fn load_hf(vb: VarBuilder, modality: &str, input_dim: usize, output_dim: usize) -> Result<Self> {
        let (linear1, norm, linear2) = match modality {
            "acoustic" => ("acoustic_linear_1", "acoustic_norm", "acoustic_linear_2"),
            "semantic" => ("semantic_linear_1", "semantic_norm", "semantic_linear_2"),
            other => {
                return Err(VibeVoiceAsrError::InvalidInput(format!(
                    "unsupported multimodal projector branch `{other}`"
                )))
            }
        };
        let fc1 = linear(input_dim, output_dim, vb.pp(linear1))?;
        let norm = RmsNorm::new(output_dim, 1e-6, vb.pp(norm))?;
        let fc2 = linear(output_dim, output_dim, vb.pp(linear2))?;
        Ok(Self { fc1, norm, fc2 })
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
    pub model_dtype: DType,
    eos_token_id: Option<u32>,
}

#[derive(Debug)]
pub struct VibeVoiceAsrSession {
    pub acoustic_features: Tensor,
    pub semantic_features: Tensor,
    pub combined_features: Tensor,
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
        let load_dtype = if device.is_cuda() {
            match dtype {
                DType::BF16 | DType::F16 => dtype,
                _ => DType::BF16,
            }
        } else {
            DType::F32
        };

        let weight_files = find_weight_files(model_dir)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, load_dtype, &device)? };

        let acoustic_tokenizer = VibeVoiceAcousticTokenizerModel::load_hf_encoder(
            &config.acoustic_tokenizer_config,
            vb.pp("acoustic_tokenizer_encoder"),
        )?;
        let semantic_tokenizer = VibeVoiceSemanticTokenizerModel::load_hf_encoder(
            &config.semantic_tokenizer_config,
            vb.pp("semantic_tokenizer_encoder"),
        )?;
        let acoustic_connector = SpeechConnector::load_hf(
            vb.pp("multi_modal_projector"),
            "acoustic",
            config.acoustic_vae_dim(),
            config.decoder_config.hidden_size,
        )?;
        let semantic_connector = SpeechConnector::load_hf(
            vb.pp("multi_modal_projector"),
            "semantic",
            config.semantic_vae_dim(),
            config.decoder_config.hidden_size,
        )?;
        let decoder_cfg = to_candle_qwen2_config(&config.decoder_config)?;
        let decoder = qwen2::ModelForCausalLM::new(
            &decoder_cfg,
            vb.pp("language_model.model"),
            Some(vb.pp("language_model.lm_head")),
        )?;
        let eos_token_id = tokenizer
            .token_to_id("<|endoftext|>")
            .or_else(|| tokenizer.get_vocab(true).get("<|endoftext|>").copied());

        Ok(Self {
            config,
            acoustic_tokenizer,
            semantic_tokenizer,
            acoustic_connector,
            semantic_connector,
            decoder,
            decoder_tokenizer: tokenizer,
            device,
            model_dtype: load_dtype,
            eos_token_id,
        })
    }

    pub fn from_hf_hub(repo_id: Option<&str>, device: Device) -> Result<Self> {
        let repo_id = repo_id.unwrap_or(DEFAULT_MODEL_REPO);
        let api = Api::new()?;
        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        if let Ok(index_path) = repo.get("model.safetensors.index.json") {
            let index: SafeTensorIndex = serde_json::from_slice(&fs::read(&index_path)?)?;
            for filename in index.weight_map.into_values().collect::<BTreeSet<_>>() {
                let _ = repo.get(&filename)?;
            }
        } else {
            let _ = repo.get("model.safetensors")?;
        }
        let model_dir = config_path
            .parent()
            .ok_or_else(|| VibeVoiceAsrError::InvalidInput("invalid HF cache path".to_string()))?;
        Self::from_local_dir(model_dir, tokenizer_path, device)
    }

    pub fn encode_speech(&self, speech_tensor: &Tensor) -> Result<VibeVoiceAsrSession> {
        let speech_tensor = speech_tensor.to_dtype(self.model_dtype)?;
        let acoustic_latents = self.acoustic_tokenizer.encode(&speech_tensor)?.sample();
        let acoustic_features = self.acoustic_connector.forward(&acoustic_latents)?;

        let semantic_latents = self.semantic_tokenizer.encode(&speech_tensor)?.sample();
        let semantic_features = self.semantic_connector.forward(&semantic_latents)?;
        let acoustic_shape = acoustic_features.dims3()?;
        let semantic_shape = semantic_features.dims3()?;
        if acoustic_shape != semantic_shape {
            return Err(VibeVoiceAsrError::InvalidInput(format!(
                "acoustic and semantic feature shapes differ: acoustic={acoustic_shape:?}, semantic={semantic_shape:?}"
            )));
        }
        let combined_features = (&acoustic_features + &semantic_features)?;

        Ok(VibeVoiceAsrSession {
            acoustic_features,
            semantic_features,
            combined_features,
        })
    }

    pub fn processor_from_tokenizer(&self) -> Result<VibeVoiceAsrProcessor> {
        VibeVoiceAsrProcessor::new(
            self.decoder_tokenizer.clone(),
            self.config.encoder_ratios_product()?,
        )
    }

    pub fn transcribe_inputs(
        &mut self,
        inputs: &VibeVoiceAsrInputs,
        max_new_tokens: usize,
    ) -> Result<String> {
        if max_new_tokens == 0 {
            return Ok(String::new());
        }
        self.decoder.clear_kv_cache();
        let speech = self.encode_speech(&inputs.speech_tensor)?;
        let input_ids = Tensor::from_vec(
            inputs.prompt_token_ids.clone(),
            (1, inputs.prompt_token_ids.len()),
            &self.device,
        )?;
        let mut input_embeds = self.decoder.embed(&input_ids)?;
        input_embeds = apply_speech_features(
            input_embeds,
            &inputs.acoustic_input_mask,
            &speech.combined_features,
        )?;

        let first_logits = self.decoder.forward_embeds(&input_embeds, 0, None)?;
        let mut generated = Vec::<u32>::new();
        let mut next_token = logits_argmax(&first_logits)?;
        let mut offset = inputs.prompt_token_ids.len();

        while generated.len() < max_new_tokens {
            if Some(next_token) == self.eos_token_id {
                break;
            }
            generated.push(next_token);
            let next = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            let logits = self.decoder.forward(&next, offset)?;
            offset += 1;
            next_token = logits_argmax(&logits)?;
        }

        self.decoder_tokenizer
            .decode(&generated, true)
            .map_err(VibeVoiceAsrError::Tokenizer)
    }

    pub fn transcribe(
        &mut self,
        prompt_token_ids: &[u32],
        speech_tensor: &Tensor,
        max_new_tokens: usize,
    ) -> Result<String> {
        let acoustic_len = self
            .config
            .encoder_ratios_product()
            .map(|ratio| {
                let samples = speech_tensor.dims3().map(|(_, _, samples)| samples).unwrap_or(0);
                samples.div_ceil(ratio)
            })
            .unwrap_or(0);
        let mut acoustic_input_mask = vec![false; prompt_token_ids.len()];
        let take = acoustic_len.min(acoustic_input_mask.len());
        acoustic_input_mask
            .iter_mut()
            .rev()
            .take(take)
            .for_each(|slot| *slot = true);
        self.transcribe_inputs(&VibeVoiceAsrInputs {
            prompt_token_ids: prompt_token_ids.to_vec(),
            acoustic_input_mask,
            speech_tensor: speech_tensor.clone(),
            sample_rate: 24_000,
        }, max_new_tokens)
    }
}

pub fn to_candle_qwen2_config(cfg: &Qwen2DecoderConfig) -> Result<qwen2::Config> {
    Ok(qwen2::Config {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        max_position_embeddings: cfg.max_position_embeddings,
        sliding_window: cfg.sliding_window.unwrap_or(cfg.max_position_embeddings),
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

fn logits_argmax(logits: &Tensor) -> Result<u32> {
    let squeezed = logits.squeeze(0)?.squeeze(0)?;
    Ok(squeezed.argmax(0)?.to_scalar::<u32>()?)
}

fn apply_speech_features(
    token_embeddings: Tensor,
    acoustic_input_mask: &[bool],
    speech_features: &Tensor,
) -> Result<Tensor> {
    let (batch, seq_len, hidden) = token_embeddings.dims3()?;
    if batch != 1 {
        return Err(VibeVoiceAsrError::InvalidInput(
            "only batch size 1 is supported in transcribe_inputs".to_string(),
        ));
    }
    if acoustic_input_mask.len() != seq_len {
        return Err(VibeVoiceAsrError::InvalidInput(format!(
            "acoustic_input_mask length {} does not match sequence length {seq_len}",
            acoustic_input_mask.len()
        )));
    }
    let (_, speech_len, speech_hidden) = speech_features.dims3()?;
    if hidden != speech_hidden {
        return Err(VibeVoiceAsrError::InvalidInput(format!(
            "speech hidden size {speech_hidden} does not match token embedding hidden size {hidden}"
        )));
    }

    let true_positions: Vec<usize> = acoustic_input_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, flag)| flag.then_some(idx))
        .collect();
    if true_positions.is_empty() {
        return Ok(token_embeddings);
    }

    let dtype = token_embeddings.dtype();
    let mut flat = token_embeddings
        .to_dtype(DTypeName::F32.into_candle())?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let speech_values = speech_features
        .to_dtype(DTypeName::F32.into_candle())?
        .flatten_all()?
        .to_vec1::<f32>()?;
    if true_positions.len() != speech_len {
        return Err(VibeVoiceAsrError::InvalidInput(format!(
            "audio placeholder count {} does not match encoded speech feature length {}; prompt/tokenizer ratio mismatch",
            true_positions.len(),
            speech_len
        )));
    }
    for (dst_idx, seq_idx) in true_positions.into_iter().enumerate() {
        let dst_offset = seq_idx * hidden;
        let src_offset = dst_idx * hidden;
        flat[dst_offset..dst_offset + hidden]
            .copy_from_slice(&speech_values[src_offset..src_offset + hidden]);
    }
    Ok(Tensor::from_vec(flat, (batch, seq_len, hidden), token_embeddings.device())?.to_dtype(dtype)?)
}

fn find_weight_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: SafeTensorIndex = serde_json::from_slice(&fs::read(index_path)?)?;
        let files: BTreeSet<PathBuf> = index
            .weight_map
            .into_values()
            .map(|name| dir.join(name))
            .collect();
        return Ok(files.into_iter().collect());
    }
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

#[derive(Debug, Deserialize)]
struct SafeTensorIndex {
    weight_map: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use vibevoice_core::VibeVoiceASRConfig;

    #[test]
    fn speech_connector_preserves_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let connector = SpeechConnector::load(4, 8, vb).unwrap();
        let xs = Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let ys = connector.forward(&xs).unwrap();
        assert_eq!(ys.dims3().unwrap(), (1, 3, 8));
    }

    #[test]
    fn qwen2_config_conversion_works() {
        let cfg = VibeVoiceASRConfig::default();
        let qcfg = to_candle_qwen2_config(&cfg.decoder_config).unwrap();
        assert_eq!(qcfg.hidden_size, cfg.decoder_config.hidden_size);
        assert_eq!(qcfg.vocab_size, cfg.decoder_config.vocab_size);
    }

    #[test]
    fn speech_features_replace_masked_positions() {
        let embeds = Tensor::zeros((1, 4, 3), DType::F32, &Device::Cpu).unwrap();
        let speech = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 2, 3), &Device::Cpu).unwrap();
        let updated = apply_speech_features(embeds, &[false, true, true, false], &speech).unwrap();
        let flat = updated.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(&flat[3..9], &[1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn find_weight_files_prefers_index() {
        let dir = std::env::temp_dir().join(format!("vibevoice-asr-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let files = find_weight_files(&dir).unwrap();
        assert_eq!(files.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_weight_files_falls_back_to_plain_safetensors() {
        let dir = std::env::temp_dir().join(format!("vibevoice-asr-plain-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("a.safetensors"), b"").unwrap();
        fs::write(dir.join("b.safetensors"), b"").unwrap();
        let files = find_weight_files(&dir).unwrap();
        assert_eq!(files.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn find_weight_files_errors_when_missing() {
        let dir = std::env::temp_dir().join(format!("vibevoice-asr-empty-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        assert!(matches!(find_weight_files(&dir), Err(VibeVoiceAsrError::InvalidInput(_))));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn speech_feature_replacement_errors_on_length_mismatch() {
        let embeds = Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let speech = Tensor::zeros((1, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let err = apply_speech_features(embeds, &[true, false, false], &speech).unwrap_err();
        assert!(matches!(err, VibeVoiceAsrError::InvalidInput(_)));
    }

    #[test]
    fn speech_feature_replacement_errors_on_batch_mismatch() {
        let embeds = Tensor::zeros((2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let speech = Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let err = apply_speech_features(embeds, &[true, false, false], &speech).unwrap_err();
        assert!(matches!(err, VibeVoiceAsrError::InvalidInput(_)));
    }

    #[test]
    fn speech_feature_replacement_errors_on_mask_length_mismatch() {
        let embeds = Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let speech = Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let err = apply_speech_features(embeds, &[true, false], &speech).unwrap_err();
        assert!(matches!(err, VibeVoiceAsrError::InvalidInput(_)));
    }

    #[test]
    fn speech_feature_replacement_errors_on_hidden_mismatch() {
        let embeds = Tensor::zeros((1, 3, 5), DType::F32, &Device::Cpu).unwrap();
        let speech = Tensor::zeros((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let err = apply_speech_features(embeds, &[true, false, false], &speech).unwrap_err();
        assert!(matches!(err, VibeVoiceAsrError::InvalidInput(_)));
    }

    #[test]
    fn speech_connector_load_hf_rejects_unknown_modality() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let err = SpeechConnector::load_hf(vb, "video", 4, 8).unwrap_err();
        assert!(matches!(err, VibeVoiceAsrError::InvalidInput(_)));
    }

    #[test]
    fn find_weight_files_deduplicates_index_entries() {
        let dir = std::env::temp_dir().join(format!("vibevoice-asr-dedupe-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"shared.safetensors","b":"shared.safetensors"}}"#,
        )
        .unwrap();
        let files = find_weight_files(&dir).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].file_name().and_then(|v| v.to_str()), Some("shared.safetensors"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn qwen2_config_rejects_unknown_activation() {
        let mut cfg = vibevoice_core::Qwen2DecoderConfig::default();
        cfg.hidden_act = "relu".to_string();
        assert!(matches!(to_candle_qwen2_config(&cfg), Err(VibeVoiceAsrError::Unsupported(_))));
    }
}
