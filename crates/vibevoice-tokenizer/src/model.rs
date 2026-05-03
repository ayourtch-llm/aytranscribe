use candle_core::{Shape, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder, conv1d};

use crate::error::{Result, VibeVoiceTokenizerError};
use vibevoice_core::{VibeVoiceAcousticTokenizerConfig, VibeVoiceSemanticTokenizerConfig};

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

#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: if vb.contains_tensor("weight") {
                vb.get((dim,), "weight")?
            } else {
                Tensor::ones((dim,), vb.dtype(), vb.device())?
            },
            eps,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let denom = ((xs.sqr()?.sum_keepdim(candle_core::D::Minus1)? / xs.dim(candle_core::D::Minus1)? as f64)?
            + self.eps)?
            .sqrt()?;
        let ys = xs.broadcast_div(&denom)?;
        Ok(ys.broadcast_mul(&self.weight)?)
    }
}

#[derive(Debug, Clone)]
pub struct ConvRmsNorm {
    norm: RmsNorm,
}

impl ConvRmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: RmsNorm::new(dim, eps, vb)?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.transpose(1, 2)?;
        let ys = self.norm.forward(&xs)?;
        Ok(ys.transpose(1, 2)?)
    }
}

pub fn get_extra_padding_for_conv1d(
    input_len: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let n_frames = (input_len.saturating_sub(kernel_size) + padding_total) as f64 / stride as f64 + 1.0;
    let ideal_length = ((n_frames.ceil() as usize).saturating_sub(1)) * stride + (kernel_size - padding_total);
    ideal_length.saturating_sub(input_len)
}

pub fn pad1d(xs: &[f32], left: usize, right: usize, value: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(left + xs.len() + right);
    out.extend(std::iter::repeat_n(value, left));
    out.extend_from_slice(xs);
    out.extend(std::iter::repeat_n(value, right));
    out
}

pub fn unpad1d(xs: &[f32], left: usize, right: usize) -> Vec<f32> {
    let end = xs.len().saturating_sub(right);
    xs[left.min(end)..end].to_vec()
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn encoder_constructs_from_config() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let encoder = TokenizerEncoder::load_acoustic(
            &VibeVoiceAcousticTokenizerConfig::default(),
            vb,
        )
        .unwrap();
        assert_eq!(encoder.hop_length(), 3200);
    }

    #[test]
    fn encoder_forward_has_expected_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let encoder = TokenizerEncoder::load_acoustic(
            &VibeVoiceAcousticTokenizerConfig::default(),
            vb,
        )
        .unwrap();
        let input = Tensor::zeros((1, 1, 3200), DType::F32, &Device::Cpu).unwrap();
        let output = encoder.forward(&input).unwrap();
        let (b, c, _t) = output.mean.dims3().unwrap();
        assert_eq!((b, c), (1, 64));
    }

    #[test]
    fn pad_and_unpad_round_trip() {
        let padded = pad1d(&[1., 2., 3.], 2, 1, 0.);
        assert_eq!(padded, vec![0., 0., 1., 2., 3., 0.]);
        assert_eq!(unpad1d(&padded, 2, 1), vec![1., 2., 3.]);
    }

    #[test]
    fn extra_padding_matches_ceil_behavior() {
        assert_eq!(get_extra_padding_for_conv1d(10, 4, 3, 1), 2);
        assert_eq!(get_extra_padding_for_conv1d(11, 4, 3, 1), 1);
    }

    #[test]
    fn rms_norm_matches_known_values() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let norm = RmsNorm::new(2, 1e-6, vb).unwrap();
        let input = Tensor::from_vec(vec![3f32, 4.0], (1, 2), &Device::Cpu).unwrap();
        let output = norm.forward(&input).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((output[0] - 0.848528).abs() < 1e-4);
        assert!((output[1] - 1.131370).abs() < 1e-4);
    }

    #[test]
    fn conv_rms_norm_preserves_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let norm = ConvRmsNorm::new(2, 1e-6, vb).unwrap();
        let input = Tensor::zeros((1, 2, 5), DType::F32, &Device::Cpu).unwrap();
        let output = norm.forward(&input).unwrap();
        assert_eq!(output.dims3().unwrap(), (1, 2, 5));
    }
}
