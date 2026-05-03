use candle_core::{Shape, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder, conv1d};

use crate::error::{Result, VibeVoiceTokenizerError};
use vibevoice_core::{
    VibeVoiceAcousticTokenizerConfig, VibeVoiceSemanticTokenizerConfig,
};

#[derive(Debug, Clone)]
pub struct TokenizerEncoderOutput {
    pub mean: Tensor,
    pub fixed_std: Option<f64>,
}

impl TokenizerEncoderOutput {
    pub fn sample(&self) -> Tensor {
        self.mean.clone()
    }
}

#[derive(Debug)]
pub struct TokenizerEncoder {
    stem: Conv1d,
    downsamples: Vec<Conv1d>,
    head: Conv1d,
    hop_length: usize,
}

impl TokenizerEncoder {
    pub fn load_acoustic(cfg: &VibeVoiceAcousticTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_common(
            cfg.channels,
            cfg.encoder_n_filters,
            &cfg.encoder_ratios,
            cfg.vae_dim,
            vb,
        )
    }

    pub fn load_semantic(cfg: &VibeVoiceSemanticTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_common(
            cfg.channels,
            cfg.encoder_n_filters,
            &cfg.encoder_ratios,
            cfg.vae_dim,
            vb,
        )
    }

    fn load_common(
        channels: usize,
        filters: usize,
        ratios: &[usize],
        vae_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let stem = conv1d(
            channels,
            filters,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("stem"),
        )?;

        let mut downsamples = Vec::with_capacity(ratios.len());
        for (idx, ratio) in ratios.iter().rev().copied().enumerate() {
            let in_ch = filters * (1usize << idx);
            let out_ch = filters * (1usize << (idx + 1));
            let kernel = ratio * 2;
            let padding = ratio.saturating_sub(1);
            downsamples.push(conv1d(
                in_ch,
                out_ch,
                kernel,
                Conv1dConfig {
                    stride: ratio,
                    padding,
                    ..Default::default()
                },
                vb.pp(format!("downsamples.{idx}")),
            )?);
        }
        let head_in = filters * (1usize << ratios.len());
        let head = conv1d(
            head_in,
            vae_dim,
            7,
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
            vb.pp("head"),
        )?;
        let hop_length = ratios.iter().product();
        Ok(Self {
            stem,
            downsamples,
            head,
            hop_length,
        })
    }

    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    pub fn forward(&self, waveform: &Tensor) -> Result<TokenizerEncoderOutput> {
        let shape = waveform.shape();
        if shape.rank() != 3 {
            return Err(VibeVoiceTokenizerError::InvalidShape(format!(
                "expected [batch, channels, time], got {shape:?}"
            )));
        }
        let mut xs = self.stem.forward(waveform)?;
        for layer in &self.downsamples {
            xs = layer.forward(&xs)?;
        }
        let mean = self.head.forward(&xs)?;
        Ok(TokenizerEncoderOutput {
            mean,
            fixed_std: None,
        })
    }
}

#[derive(Debug)]
pub struct TokenizerDecoder;

impl TokenizerDecoder {
    pub fn decode(&self, _latents: &Tensor) -> Result<Tensor> {
        Err(VibeVoiceTokenizerError::Unsupported(
            "tokenizer decoder is not implemented yet",
        ))
    }
}

#[derive(Debug)]
pub struct VibeVoiceAcousticTokenizerModel {
    encoder: TokenizerEncoder,
    decoder: TokenizerDecoder,
    std_dist_type: String,
}

impl VibeVoiceAcousticTokenizerModel {
    pub fn load(cfg: &VibeVoiceAcousticTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: TokenizerEncoder::load_acoustic(cfg, vb.pp("encoder"))?,
            decoder: TokenizerDecoder,
            std_dist_type: cfg.std_dist_type.clone(),
        })
    }

    pub fn encode(&self, waveform: &Tensor) -> Result<TokenizerEncoderOutput> {
        let mut out = self.encoder.forward(waveform)?;
        if self.std_dist_type == "gaussian" {
            out.fixed_std = Some(0.5);
        }
        Ok(out)
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        self.decoder.decode(latents)
    }

    pub fn hop_length(&self) -> usize {
        self.encoder.hop_length()
    }
}

#[derive(Debug)]
pub struct VibeVoiceSemanticTokenizerModel {
    encoder: TokenizerEncoder,
}

impl VibeVoiceSemanticTokenizerModel {
    pub fn load(cfg: &VibeVoiceSemanticTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: TokenizerEncoder::load_semantic(cfg, vb.pp("encoder"))?,
        })
    }

    pub fn encode(&self, waveform: &Tensor) -> Result<TokenizerEncoderOutput> {
        self.encoder.forward(waveform)
    }

    pub fn hop_length(&self) -> usize {
        self.encoder.hop_length()
    }
}

#[allow(dead_code)]
fn _shape3(shape: &Shape) -> Option<(usize, usize, usize)> {
    let dims = shape.dims();
    if dims.len() == 3 {
        Some((dims[0], dims[1], dims[2]))
    } else {
        None
    }
}

