use serde::{Deserialize, Serialize};

use crate::error::{Result, VibeVoiceCoreError};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DTypeName {
    #[serde(alias = "float16")]
    F16,
    #[serde(alias = "float32")]
    #[default]
    F32,
    #[serde(alias = "bfloat16")]
    Bf16,
}

impl DTypeName {
    pub fn into_candle(self) -> candle_core::DType {
        match self {
            Self::F16 => candle_core::DType::F16,
            Self::F32 => candle_core::DType::F32,
            Self::Bf16 => candle_core::DType::BF16,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VibeVoiceAcousticTokenizerConfig {
    #[serde(default = "default_channels")]
    pub channels: usize,
    #[serde(default)]
    pub corpus_normalize: f64,
    #[serde(default = "default_true")]
    pub causal: bool,
    #[serde(default = "default_acoustic_vae_dim")]
    pub vae_dim: usize,
    #[serde(default = "default_fix_std")]
    pub fix_std: f64,
    #[serde(default = "default_gaussian")]
    pub std_dist_type: String,
    #[serde(default = "default_depthwise_conv")]
    pub mixer_layer: String,
    #[serde(default = "default_none")]
    pub conv_norm: String,
    #[serde(default = "default_constant")]
    pub pad_mode: String,
    #[serde(default = "default_true")]
    pub disable_last_norm: bool,
    #[serde(default = "default_rmsnorm")]
    pub layernorm: String,
    #[serde(default = "default_layernorm_eps")]
    pub layernorm_eps: f64,
    #[serde(default = "default_true")]
    pub layernorm_elementwise_affine: bool,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default = "default_layer_scale")]
    pub layer_scale_init_value: f64,
    #[serde(default = "default_weight_init")]
    pub weight_init_value: f64,
    #[serde(default = "default_filters")]
    pub encoder_n_filters: usize,
    #[serde(default = "default_encoder_ratios")]
    pub encoder_ratios: Vec<usize>,
    #[serde(default = "default_encoder_depths")]
    pub encoder_depths: String,
    #[serde(default = "default_filters")]
    pub decoder_n_filters: usize,
    #[serde(default)]
    pub decoder_ratios: Option<Vec<usize>>,
    #[serde(default)]
    pub decoder_depths: Option<String>,
}

impl Default for VibeVoiceAcousticTokenizerConfig {
    fn default() -> Self {
        Self {
            channels: default_channels(),
            corpus_normalize: 0.0,
            causal: true,
            vae_dim: default_acoustic_vae_dim(),
            fix_std: default_fix_std(),
            std_dist_type: default_gaussian(),
            mixer_layer: default_depthwise_conv(),
            conv_norm: default_none(),
            pad_mode: default_constant(),
            disable_last_norm: true,
            layernorm: default_rmsnorm(),
            layernorm_eps: default_layernorm_eps(),
            layernorm_elementwise_affine: true,
            conv_bias: true,
            layer_scale_init_value: default_layer_scale(),
            weight_init_value: default_weight_init(),
            encoder_n_filters: default_filters(),
            encoder_ratios: default_encoder_ratios(),
            encoder_depths: default_encoder_depths(),
            decoder_n_filters: default_filters(),
            decoder_ratios: None,
            decoder_depths: None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VibeVoiceSemanticTokenizerConfig {
    #[serde(default = "default_channels")]
    pub channels: usize,
    #[serde(default)]
    pub corpus_normalize: f64,
    #[serde(default = "default_true")]
    pub causal: bool,
    #[serde(default = "default_semantic_vae_dim")]
    pub vae_dim: usize,
    #[serde(default)]
    pub fix_std: f64,
    #[serde(default = "default_none")]
    pub std_dist_type: String,
    #[serde(default = "default_depthwise_conv")]
    pub mixer_layer: String,
    #[serde(default = "default_none")]
    pub conv_norm: String,
    #[serde(default = "default_constant")]
    pub pad_mode: String,
    #[serde(default = "default_true")]
    pub disable_last_norm: bool,
    #[serde(default = "default_rmsnorm")]
    pub layernorm: String,
    #[serde(default = "default_layernorm_eps")]
    pub layernorm_eps: f64,
    #[serde(default = "default_true")]
    pub layernorm_elementwise_affine: bool,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default = "default_layer_scale")]
    pub layer_scale_init_value: f64,
    #[serde(default = "default_weight_init")]
    pub weight_init_value: f64,
    #[serde(default = "default_filters")]
    pub encoder_n_filters: usize,
    #[serde(default = "default_encoder_ratios")]
    pub encoder_ratios: Vec<usize>,
    #[serde(default = "default_encoder_depths")]
    pub encoder_depths: String,
}

impl Default for VibeVoiceSemanticTokenizerConfig {
    fn default() -> Self {
        Self {
            channels: default_channels(),
            corpus_normalize: 0.0,
            causal: true,
            vae_dim: default_semantic_vae_dim(),
            fix_std: 0.0,
            std_dist_type: default_none(),
            mixer_layer: default_depthwise_conv(),
            conv_norm: default_none(),
            pad_mode: default_constant(),
            disable_last_norm: true,
            layernorm: default_rmsnorm(),
            layernorm_eps: default_layernorm_eps(),
            layernorm_elementwise_affine: true,
            conv_bias: true,
            layer_scale_init_value: default_layer_scale(),
            weight_init_value: default_weight_init(),
            encoder_n_filters: default_filters(),
            encoder_ratios: default_encoder_ratios(),
            encoder_depths: default_encoder_depths(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VibeVoiceDiffusionHeadConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,
    #[serde(default = "default_head_ffn_ratio")]
    pub head_ffn_ratio: f64,
    #[serde(default = "default_layernorm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_acoustic_vae_dim")]
    pub latent_size: usize,
    #[serde(default)]
    pub speech_vae_dim: Option<usize>,
    #[serde(default = "default_prediction_type")]
    pub prediction_type: String,
    #[serde(default = "default_diffusion_type")]
    pub diffusion_type: String,
    #[serde(default = "default_ddpm_steps")]
    pub ddpm_num_steps: usize,
    #[serde(default = "default_ddpm_inference_steps")]
    pub ddpm_num_inference_steps: usize,
    #[serde(default = "default_ddpm_schedule")]
    pub ddpm_beta_schedule: String,
    #[serde(default = "default_ddpm_batch_mul")]
    pub ddpm_batch_mul: usize,
}

impl Default for VibeVoiceDiffusionHeadConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            head_layers: default_head_layers(),
            head_ffn_ratio: default_head_ffn_ratio(),
            rms_norm_eps: default_layernorm_eps(),
            latent_size: default_acoustic_vae_dim(),
            speech_vae_dim: None,
            prediction_type: default_prediction_type(),
            diffusion_type: default_diffusion_type(),
            ddpm_num_steps: default_ddpm_steps(),
            ddpm_num_inference_steps: default_ddpm_inference_steps(),
            ddpm_beta_schedule: default_ddpm_schedule(),
            ddpm_batch_mul: default_ddpm_batch_mul(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Qwen2DecoderConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_positions")]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default = "default_max_window_layers")]
    pub max_window_layers: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_qwen_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub initializer_range: f64,
    #[serde(default)]
    pub torch_dtype: Option<DTypeName>,
}

impl Default for Qwen2DecoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: default_vocab_size(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_hidden_layers(),
            num_attention_heads: default_attention_heads(),
            num_key_value_heads: default_kv_heads(),
            max_position_embeddings: default_positions(),
            sliding_window: Some(default_sliding_window()),
            max_window_layers: default_max_window_layers(),
            tie_word_embeddings: false,
            rope_theta: default_rope_theta(),
            rms_norm_eps: default_qwen_eps(),
            use_sliding_window: false,
            hidden_act: default_hidden_act(),
            attention_dropout: 0.0,
            initializer_range: 0.02,
            torch_dtype: Some(DTypeName::Bf16),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct VibeVoiceConfig {
    #[serde(default)]
    pub acoustic_tokenizer_config: VibeVoiceAcousticTokenizerConfig,
    #[serde(default)]
    pub semantic_tokenizer_config: VibeVoiceSemanticTokenizerConfig,
    #[serde(default)]
    pub decoder_config: Qwen2DecoderConfig,
    #[serde(default)]
    pub diffusion_head_config: VibeVoiceDiffusionHeadConfig,
    #[serde(default)]
    pub acoustic_vae_dim: Option<usize>,
    #[serde(default)]
    pub semantic_vae_dim: Option<usize>,
    #[serde(default)]
    pub torch_dtype: Option<DTypeName>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct VibeVoiceASRConfig {
    #[serde(default)]
    pub acoustic_tokenizer_config: VibeVoiceAcousticTokenizerConfig,
    #[serde(default)]
    pub semantic_tokenizer_config: VibeVoiceSemanticTokenizerConfig,
    #[serde(default)]
    pub decoder_config: Qwen2DecoderConfig,
    #[serde(default)]
    pub acoustic_vae_dim: Option<usize>,
    #[serde(default)]
    pub semantic_vae_dim: Option<usize>,
    #[serde(default)]
    pub torch_dtype: Option<DTypeName>,
}

impl VibeVoiceASRConfig {
    pub fn acoustic_vae_dim(&self) -> usize {
        self.acoustic_vae_dim
            .unwrap_or(self.acoustic_tokenizer_config.vae_dim)
    }

    pub fn semantic_vae_dim(&self) -> usize {
        self.semantic_vae_dim
            .unwrap_or(self.semantic_tokenizer_config.vae_dim)
    }

    pub fn from_reader(reader: impl std::io::Read) -> Result<Self> {
        Ok(serde_json::from_reader(reader)?)
    }

    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let data = std::fs::read(path)?;
        Ok(serde_json::from_slice(&data)?)
    }

    pub fn encoder_ratios_product(&self) -> Result<usize> {
        let product = self
            .acoustic_tokenizer_config
            .encoder_ratios
            .iter()
            .copied()
            .try_fold(1usize, |acc, v| acc.checked_mul(v))
            .ok_or_else(|| {
                VibeVoiceCoreError::UnsupportedConfig(
                    "encoder ratio product overflowed usize".to_string(),
                )
            })?;
        Ok(product)
    }
}

fn default_true() -> bool {
    true
}
fn default_channels() -> usize {
    1
}
fn default_acoustic_vae_dim() -> usize {
    64
}
fn default_semantic_vae_dim() -> usize {
    128
}
fn default_fix_std() -> f64 {
    0.5
}
fn default_gaussian() -> String {
    "gaussian".to_string()
}
fn default_depthwise_conv() -> String {
    "depthwise_conv".to_string()
}
fn default_none() -> String {
    "none".to_string()
}
fn default_constant() -> String {
    "constant".to_string()
}
fn default_rmsnorm() -> String {
    "RMSNorm".to_string()
}
fn default_layernorm_eps() -> f64 {
    1e-5
}
fn default_layer_scale() -> f64 {
    1e-6
}
fn default_weight_init() -> f64 {
    1e-2
}
fn default_filters() -> usize {
    32
}
fn default_encoder_ratios() -> Vec<usize> {
    vec![8, 5, 5, 4, 2, 2]
}
fn default_encoder_depths() -> String {
    "3-3-3-3-3-3-8".to_string()
}
fn default_hidden_size() -> usize {
    3584
}
fn default_head_layers() -> usize {
    4
}
fn default_head_ffn_ratio() -> f64 {
    3.0
}
fn default_prediction_type() -> String {
    "v_prediction".to_string()
}
fn default_diffusion_type() -> String {
    "ddpm".to_string()
}
fn default_ddpm_steps() -> usize {
    1000
}
fn default_ddpm_inference_steps() -> usize {
    20
}
fn default_ddpm_schedule() -> String {
    "cosine".to_string()
}
fn default_ddpm_batch_mul() -> usize {
    4
}
fn default_vocab_size() -> usize {
    152064
}
fn default_intermediate_size() -> usize {
    18944
}
fn default_hidden_layers() -> usize {
    28
}
fn default_attention_heads() -> usize {
    28
}
fn default_kv_heads() -> usize {
    4
}
fn default_positions() -> usize {
    32768
}
fn default_sliding_window() -> usize {
    4096
}
fn default_max_window_layers() -> usize {
    28
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_qwen_eps() -> f64 {
    1e-6
}
fn default_hidden_act() -> String {
    "silu".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parses_real_vibevoice_config_json() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp/VibeVoice/vibevoice/configs/qwen2.5_7b_32k.json");
        let config = VibeVoiceASRConfig::from_path(path).unwrap();
        assert_eq!(config.decoder_config.hidden_size, 3584);
        assert_eq!(config.acoustic_vae_dim(), 64);
        assert_eq!(config.semantic_vae_dim(), 128);
        assert_eq!(config.encoder_ratios_product().unwrap(), 3200);
    }

    #[test]
    fn config_round_trip_serialization() {
        let config = VibeVoiceASRConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: VibeVoiceASRConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.decoder_config.hidden_size, config.decoder_config.hidden_size);
        assert_eq!(parsed.acoustic_vae_dim(), config.acoustic_vae_dim());
    }

    #[test]
    fn partial_config_uses_defaults() {
        let parsed: VibeVoiceASRConfig =
            serde_json::from_str(r#"{"decoder_config":{"hidden_size":1024}}"#).unwrap();
        assert_eq!(parsed.decoder_config.hidden_size, 1024);
        assert_eq!(parsed.decoder_config.vocab_size, default_vocab_size());
        assert_eq!(parsed.acoustic_tokenizer_config.encoder_ratios, default_encoder_ratios());
    }

    #[test]
    fn dtype_aliases_deserialize() {
        let bf16: DTypeName = serde_json::from_str(r#""bfloat16""#).unwrap();
        let f16: DTypeName = serde_json::from_str(r#""float16""#).unwrap();
        let f32: DTypeName = serde_json::from_str(r#""float32""#).unwrap();
        assert_eq!(bf16, DTypeName::Bf16);
        assert_eq!(f16, DTypeName::F16);
        assert_eq!(f32, DTypeName::F32);
    }

    #[test]
    fn from_reader_accepts_partial_json() {
        let json = br#"{"semantic_vae_dim":256,"decoder_config":{"torch_dtype":"float16"}}"#;
        let parsed = VibeVoiceASRConfig::from_reader(&json[..]).unwrap();
        assert_eq!(parsed.semantic_vae_dim(), 256);
        assert_eq!(parsed.decoder_config.torch_dtype, Some(DTypeName::F16));
        assert_eq!(parsed.acoustic_vae_dim(), default_acoustic_vae_dim());
    }

    #[test]
    fn encoder_ratio_overflow_is_reported() {
        let config = VibeVoiceASRConfig {
            acoustic_tokenizer_config: VibeVoiceAcousticTokenizerConfig {
                encoder_ratios: vec![usize::MAX, 2],
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            config.encoder_ratios_product(),
            Err(VibeVoiceCoreError::UnsupportedConfig(_))
        ));
    }
}
