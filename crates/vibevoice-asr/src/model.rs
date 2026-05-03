use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde::Deserialize;
use tokenizers::Tokenizer;
use vibevoice_core::{DTypeName, Qwen2DecoderConfig, VibeVoiceASRConfig};
use vibevoice_tokenizer::{VibeVoiceAcousticTokenizerModel, VibeVoiceSemanticTokenizerModel};

use crate::{
    error::{Result, VibeVoiceAsrError},
    processor::VibeVoiceAsrInputs,
    qwen2::{self, RmsNorm},
};

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

        let weight_files = find_weight_files(model_dir)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)? };

        let acoustic_root = pick_prefix(&vb, &["model.acoustic_tokenizer", "acoustic_tokenizer"])?;
        let semantic_root = pick_prefix(&vb, &["model.semantic_tokenizer", "semantic_tokenizer"])?;
        let acoustic_connector_root = pick_prefix(
            &vb,
            &["model.acoustic_connector", "acoustic_connector"],
        )?;
        let semantic_connector_root = pick_prefix(
            &vb,
            &["model.semantic_connector", "semantic_connector"],
        )?;
        let language_model_root = pick_prefix(
            &vb,
            &["model.language_model", "language_model"],
        )?;

        let acoustic_tokenizer = VibeVoiceAcousticTokenizerModel::load(
            &config.acoustic_tokenizer_config,
            vb.pp(acoustic_root),
        )?;
        let semantic_tokenizer = VibeVoiceSemanticTokenizerModel::load(
            &config.semantic_tokenizer_config,
            vb.pp(semantic_root),
        )?;
        let acoustic_connector = SpeechConnector::load(
            config.acoustic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp(acoustic_connector_root),
        )?;
        let semantic_connector = SpeechConnector::load(
            config.semantic_vae_dim(),
            config.decoder_config.hidden_size,
            vb.pp(semantic_connector_root),
        )?;
        let decoder_cfg = to_candle_qwen2_config(&config.decoder_config)?;
        let decoder = qwen2::ModelForCausalLM::new(&decoder_cfg, vb.pp(language_model_root))?;
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
            eos_token_id,
        })
    }

    pub fn from_hf_hub(repo_id: &str, device: Device) -> Result<Self> {
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
        let acoustic_latents = self.acoustic_tokenizer.encode(speech_tensor)?.sample();
        let acoustic_features = self
            .acoustic_connector
            .forward(&transpose_bct_to_btc(&acoustic_latents)?)?;

        let semantic_latents = self.semantic_tokenizer.encode(speech_tensor)?.sample();
        let semantic_features = self
            .semantic_connector
            .forward(&transpose_bct_to_btc(&semantic_latents)?)?;
        let combined_features = (&acoustic_features + &semantic_features)?;

        Ok(VibeVoiceAsrSession {
            acoustic_features,
            semantic_features,
            combined_features,
        })
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

fn transpose_bct_to_btc(xs: &Tensor) -> Result<Tensor> {
    Ok(xs.transpose(1, 2)?)
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
    for (dst_idx, seq_idx) in true_positions.into_iter().enumerate() {
        if dst_idx >= speech_len {
            break;
        }
        let dst_offset = seq_idx * hidden;
        let src_offset = dst_idx * hidden;
        flat[dst_offset..dst_offset + hidden]
            .copy_from_slice(&speech_values[src_offset..src_offset + hidden]);
    }
    Ok(Tensor::from_vec(flat, (batch, seq_len, hidden), token_embeddings.device())?.to_dtype(dtype)?)
}

fn pick_prefix(vb: &VarBuilder, candidates: &[&str]) -> Result<String> {
    for prefix in candidates {
        let probe = format!("{prefix}.fc1.weight");
        if vb.contains_tensor(&probe) {
            return Ok((*prefix).to_string());
        }
        let probe = format!("{prefix}.encoder.head.weight");
        if vb.contains_tensor(&probe) {
            return Ok((*prefix).to_string());
        }
        let probe = format!("{prefix}.model.embed_tokens.weight");
        if vb.contains_tensor(&probe) {
            return Ok((*prefix).to_string());
        }
    }
    Err(VibeVoiceAsrError::InvalidInput(format!(
        "could not resolve weight prefix from candidates: {candidates:?}"
    )))
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
    use candle_core::Device;
    use candle_nn::{VarBuilder, VarMap};
    use candle_core::DType;
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
}
