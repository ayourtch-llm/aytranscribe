use candle_core::{DType, Shape, Tensor};
use candle_nn::{
    Activation, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear, Module,
    VarBuilder, conv1d, conv_transpose1d, linear,
};

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
        let denom = ((xs.sqr()?.sum_keepdim(candle_core::D::Minus1)?
            / xs.dim(candle_core::D::Minus1)? as f64)?
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
    let length = input_len as f64;
    let kernel_size = kernel_size as f64;
    let stride = stride as f64;
    let padding_total = padding_total as f64;
    let n_frames = (length - kernel_size + padding_total) / stride + 1.0;
    let ideal_length =
        (n_frames.ceil() - 1.0) * stride + (kernel_size - padding_total);
    ideal_length.max(length).round() as usize - input_len
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

#[derive(Debug, Clone)]
struct NormConv1d {
    conv: Conv1d,
}

impl NormConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            ..Default::default()
        };
        let conv = if bias {
            conv1d(in_channels, out_channels, kernel_size, cfg, vb)?
        } else {
            let w = vb.get(
                (out_channels, in_channels / groups, kernel_size),
                "weight",
            )?;
            Conv1d::new(w, None, cfg)
        };
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct SConv1d {
    conv: NormConv1d,
    causal: bool,
    pad_value: f32,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding_total: usize,
}

impl SConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv = NormConv1d::new(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
            vb,
        )?;
        let padding_total = (kernel_size - 1) * dilation - (stride - 1);
        Ok(Self {
            conv,
            causal,
            pad_value: 0.0,
            kernel_size,
            stride,
            dilation,
            padding_total,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, t) = xs.dims3()?;
        let extra_padding =
            get_extra_padding_for_conv1d(t, self.kernel_size, self.stride, self.padding_total);
        let padded = pad_tensor1d(
            xs,
            if self.causal {
                self.padding_total
            } else {
                self.padding_total - self.padding_total / 2
            },
            if self.causal {
                extra_padding
            } else {
                self.padding_total / 2 + extra_padding
            },
            self.pad_value,
        )?;
        self.conv.forward(&padded)
    }
}

#[derive(Debug, Clone)]
struct NormConvTranspose1d {
    convtr: ConvTranspose1d,
}

impl NormConvTranspose1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            stride,
            ..Default::default()
        };
        let convtr = if bias {
            conv_transpose1d(in_channels, out_channels, kernel_size, cfg, vb)?
        } else {
            let w = vb.get((in_channels, out_channels, kernel_size), "weight")?;
            ConvTranspose1d::new(w, None, cfg)
        };
        Ok(Self { convtr })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.convtr.forward(xs).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct SConvTranspose1d {
    convtr: NormConvTranspose1d,
    causal: bool,
    trim_right_ratio: f64,
    padding_total: usize,
}

impl SConvTranspose1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        bias: bool,
        causal: bool,
        trim_right_ratio: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            convtr: NormConvTranspose1d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                bias,
                vb,
            )?,
            causal,
            trim_right_ratio,
            padding_total: kernel_size - stride,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.convtr.forward(xs)?;
        let (padding_left, padding_right) = if self.causal {
            let padding_right = (self.padding_total as f64 * self.trim_right_ratio).ceil() as usize;
            (self.padding_total - padding_right, padding_right)
        } else {
            let padding_right = self.padding_total / 2;
            (self.padding_total - padding_right, padding_right)
        };
        trim_tensor1d(&ys, padding_left, padding_right)
    }
}

#[derive(Debug, Clone)]
struct Ffn {
    linear1: Linear,
    linear2: Linear,
}

impl Ffn {
    fn new(embed_dim: usize, ffn_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let linear1 = linear(embed_dim, ffn_dim, vb.pp("linear1"))?;
        let linear2 = linear(ffn_dim, embed_dim, vb.pp("linear2"))?;
        let _ = bias;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear1.forward(xs)?;
        let xs = xs.apply(&Activation::Gelu)?;
        self.linear2.forward(&xs).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
struct Convlayer {
    conv: SConv1d,
}

impl Convlayer {
    fn new(dim: usize, kernel_size: usize, groups: usize, causal: bool, bias: bool, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            conv: SConv1d::new(
                dim, dim, kernel_size, 1, 1, groups, bias, causal, vb,
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Block1D {
    norm: ConvRmsNorm,
    ffn_norm: ConvRmsNorm,
    mixer: Convlayer,
    ffn: Ffn,
    gamma: Option<Tensor>,
    ffn_gamma: Option<Tensor>,
}

impl Block1D {
    fn new(dim: usize, kernel_size: usize, causal: bool, bias: bool, depthwise: bool, vb: VarBuilder) -> Result<Self> {
        let groups = if depthwise { dim } else { 1 };
        Ok(Self {
            norm: ConvRmsNorm::new(dim, 1e-5, vb.pp("norm"))?,
            ffn_norm: ConvRmsNorm::new(dim, 1e-5, vb.pp("ffn_norm"))?,
            mixer: Convlayer::new(dim, kernel_size, groups, causal, bias, vb.pp("mixer").pp("conv"))?,
            ffn: Ffn::new(dim, dim * 4, bias, vb.pp("ffn"))?,
            gamma: if vb.contains_tensor("gamma") {
                Some(vb.get((dim,), "gamma")?)
            } else {
                None
            },
            ffn_gamma: if vb.contains_tensor("ffn_gamma") {
                Some(vb.get((dim,), "ffn_gamma")?)
            } else {
                None
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let mut ys = self.norm.forward(xs)?;
        ys = self.mixer.forward(&ys)?;
        if let Some(gamma) = &self.gamma {
            ys = ys.broadcast_mul(&gamma.reshape((1, gamma.dim(0)?, 1))?)?;
        }
        let xs = (residual + ys)?;

        let residual = &xs;
        let mut ys = self.ffn_norm.forward(&xs)?;
        ys = ys.transpose(1, 2)?;
        ys = self.ffn.forward(&ys)?;
        ys = ys.transpose(1, 2)?;
        if let Some(gamma) = &self.ffn_gamma {
            ys = ys.broadcast_mul(&gamma.reshape((1, gamma.dim(0)?, 1))?)?;
        }
        Ok((residual + ys)?)
    }
}

#[derive(Debug)]
struct EncoderStem {
    conv: SConv1d,
    stage: Vec<Block1D>,
}

#[derive(Debug)]
struct EncoderConvLayer {
    conv: SConv1d,
    stage: Vec<Block1D>,
}

#[derive(Debug)]
struct EncoderHead {
    conv: SConv1d,
}

#[derive(Debug)]
pub struct TokenizerEncoder {
    stem: EncoderStem,
    conv_layers: Vec<EncoderConvLayer>,
    head: EncoderHead,
    hop_length: usize,
}

impl TokenizerEncoder {
    pub fn load_acoustic(cfg: &VibeVoiceAcousticTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_hf_encoder_layout(
            cfg.channels,
            cfg.encoder_n_filters,
            &cfg.encoder_ratios,
            cfg.vae_dim,
            cfg.causal,
            cfg.conv_bias,
            vb,
        )
    }

    pub fn load_semantic(cfg: &VibeVoiceSemanticTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_hf_encoder_layout(
            cfg.channels,
            cfg.encoder_n_filters,
            &cfg.encoder_ratios,
            cfg.vae_dim,
            cfg.causal,
            cfg.conv_bias,
            vb,
        )
    }

    pub fn load_hf_encoder_layout(
        channels: usize,
        filters: usize,
        ratios: &[usize],
        vae_dim: usize,
        causal: bool,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let stage_depths = [3usize, 3, 3, 3, 3, 3, 8];
        let ratios_rev: Vec<usize> = ratios.iter().copied().rev().collect();

        let stem = EncoderStem {
            conv: SConv1d::new(
                channels,
                filters,
                7,
                1,
                1,
                1,
                bias,
                causal,
                vb.pp("stem").pp("conv").pp("conv"),
            )?,
            stage: load_stage(
                filters,
                stage_depths[0],
                causal,
                bias,
                true,
                vb.pp("stem").pp("stage"),
            )?,
        };

        let mut conv_layers = Vec::with_capacity(ratios_rev.len());
        for (idx, ratio) in ratios_rev.iter().copied().enumerate() {
            let in_ch = filters * (1usize << idx);
            let out_ch = filters * (1usize << (idx + 1));
            conv_layers.push(EncoderConvLayer {
                conv: SConv1d::new(
                    in_ch,
                    out_ch,
                    ratio * 2,
                    ratio,
                    1,
                    1,
                    bias,
                    causal,
                    vb.pp("conv_layers").pp(idx).pp("conv").pp("conv"),
                )?,
                stage: load_stage(
                    out_ch,
                    stage_depths[idx + 1],
                    causal,
                    bias,
                    true,
                    vb.pp("conv_layers").pp(idx).pp("stage"),
                )?,
            });
        }

        let head = EncoderHead {
            conv: SConv1d::new(
                filters * (1usize << ratios.len()),
                vae_dim,
                7,
                1,
                1,
                1,
                bias,
                causal,
                vb.pp("head").pp("conv"),
            )?,
        };

        Ok(Self {
            stem,
            conv_layers,
            head,
            hop_length: ratios.iter().product(),
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
        let mut xs = self.stem.conv.forward(waveform)?;
        for block in &self.stem.stage {
            xs = block.forward(&xs)?;
        }
        for layer in &self.conv_layers {
            xs = layer.conv.forward(&xs)?;
            for block in &layer.stage {
                xs = block.forward(&xs)?;
            }
        }
        let mean = self.head.conv.forward(&xs)?.transpose(1, 2)?;
        Ok(TokenizerEncoderOutput {
            mean,
            fixed_std: None,
        })
    }
}

fn load_stage(
    dim: usize,
    depth: usize,
    causal: bool,
    bias: bool,
    depthwise: bool,
    vb: VarBuilder,
) -> Result<Vec<Block1D>> {
    let mut stage = Vec::with_capacity(depth);
    for idx in 0..depth {
        stage.push(Block1D::new(
            dim,
            7,
            causal,
            bias,
            depthwise,
            vb.pp(idx),
        )?);
    }
    Ok(stage)
}

#[derive(Debug)]
pub struct TokenizerDecoder {
    upsample_layers: Vec<DecoderLayerOp>,
    stages: Vec<Vec<Block1D>>,
    norm: Option<ConvRmsNorm>,
    head: Option<SConv1d>,
    loaded: bool,
}

#[derive(Debug)]
enum DecoderLayerOp {
    Conv(SConv1d),
    ConvTr(SConvTranspose1d),
}

impl DecoderLayerOp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(layer) => layer.forward(xs),
            Self::ConvTr(layer) => layer.forward(xs),
        }
    }
}

impl TokenizerDecoder {
    pub fn load(cfg: &VibeVoiceAcousticTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let encoder_depths = parse_depths(&cfg.encoder_depths);
        let decoder_depths = cfg
            .decoder_depths
            .as_ref()
            .map(|v| parse_depths(v))
            .unwrap_or_else(|| encoder_depths.iter().copied().rev().collect());
        let ratios = cfg
            .decoder_ratios
            .clone()
            .unwrap_or_else(|| cfg.encoder_ratios.clone());
        let mut upsample_layers = Vec::with_capacity(ratios.len() + 1);
        upsample_layers.push(DecoderLayerOp::Conv(SConv1d::new(
            cfg.vae_dim,
            cfg.decoder_n_filters * (1usize << (decoder_depths.len() - 1)),
            7,
            1,
            1,
            1,
            cfg.conv_bias,
            cfg.causal,
            vb.pp("upsample_layers").pp(0).pp(0),
        )?));
        for i in 0..ratios.len() {
            let in_ch = cfg.decoder_n_filters * (1usize << (decoder_depths.len() - 1 - i));
            let out_ch = cfg.decoder_n_filters * (1usize << (decoder_depths.len() - 2 - i));
            upsample_layers.push(DecoderLayerOp::ConvTr(SConvTranspose1d::new(
                in_ch,
                out_ch,
                ratios[i] * 2,
                ratios[i],
                cfg.conv_bias,
                cfg.causal,
                1.0,
                vb.pp("upsample_layers").pp(i + 1).pp(0),
            )?));
        }

        let mut stages = Vec::with_capacity(decoder_depths.len());
        for (i, depth) in decoder_depths.iter().copied().enumerate() {
            let dim = cfg.decoder_n_filters * (1usize << (decoder_depths.len() - 1 - i));
            let mut stage = Vec::with_capacity(depth);
            for j in 0..depth {
                stage.push(Block1D::new(
                    dim,
                    7,
                    cfg.causal,
                    cfg.conv_bias,
                    cfg.mixer_layer == "depthwise_conv",
                    vb.pp("stages").pp(i).pp(j),
                )?);
            }
            stages.push(stage);
        }
        let final_dim = cfg.decoder_n_filters;
        let norm = if cfg.disable_last_norm {
            None
        } else {
            Some(ConvRmsNorm::new(final_dim, 1e-5, vb.pp("norm"))?)
        };
        let head = SConv1d::new(final_dim, cfg.channels, 7, 1, 1, 1, cfg.conv_bias, cfg.causal, vb.pp("head"))?;
        Ok(Self {
            upsample_layers,
            stages,
            norm,
            head: Some(head),
            loaded: true,
        })
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        if !self.loaded {
            return Err(VibeVoiceTokenizerError::Unsupported(
                "tokenizer decoder weights are not loaded",
            ));
        }
        let mut xs = if latents.dims3()?.2 == 1 || latents.dims3()?.2 == 64 || latents.dims3()?.2 == 128 {
            latents.transpose(1, 2)?
        } else {
            latents.clone()
        };
        for i in 0..self.stages.len() {
            xs = self.upsample_layers[i].forward(&xs)?;
            for block in &self.stages[i] {
                xs = block.forward(&xs)?;
            }
        }
        if let Some(norm) = &self.norm {
            xs = norm.forward(&xs)?;
        }
        self.head
            .as_ref()
            .ok_or(VibeVoiceTokenizerError::Unsupported(
                "tokenizer decoder head is not loaded",
            ))?
            .forward(&xs)
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
        let decoder = if vb.contains_tensor("decoder.upsample_layers.0.0.conv.conv.weight")
            || vb.contains_tensor("decoder.head.conv.conv.weight")
        {
            TokenizerDecoder::load(cfg, vb.pp("decoder"))?
        } else {
            TokenizerDecoder {
                upsample_layers: Vec::new(),
                stages: Vec::new(),
                norm: None,
                head: None,
                loaded: false,
            }
        };
        Ok(Self {
            encoder: TokenizerEncoder::load_acoustic(cfg, vb.pp("encoder"))?,
            decoder,
            std_dist_type: cfg.std_dist_type.clone(),
        })
    }

    pub fn load_hf_encoder(cfg: &VibeVoiceAcousticTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: TokenizerEncoder::load_hf_encoder_layout(
                cfg.channels,
                cfg.encoder_n_filters,
                &cfg.encoder_ratios,
                cfg.vae_dim,
                cfg.causal,
                cfg.conv_bias,
                vb,
            )?,
            decoder: TokenizerDecoder {
                upsample_layers: Vec::new(),
                stages: Vec::new(),
                norm: None,
                head: None,
                loaded: false,
            },
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

    pub fn load_hf_encoder(cfg: &VibeVoiceSemanticTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            encoder: TokenizerEncoder::load_hf_encoder_layout(
                cfg.channels,
                cfg.encoder_n_filters,
                &cfg.encoder_ratios,
                cfg.vae_dim,
                cfg.causal,
                cfg.conv_bias,
                vb,
            )?,
        })
    }

    pub fn encode(&self, waveform: &Tensor) -> Result<TokenizerEncoderOutput> {
        self.encoder.forward(waveform)
    }

    pub fn hop_length(&self) -> usize {
        self.encoder.hop_length()
    }
}

fn parse_depths(depths: &str) -> Vec<usize> {
    depths
        .split('-')
        .filter_map(|v| v.parse::<usize>().ok())
        .collect()
}

fn pad_tensor1d(xs: &Tensor, left: usize, right: usize, value: f32) -> Result<Tensor> {
    let (b, c, t) = xs.dims3()?;
    let dtype = xs.dtype();
    let data = xs
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut out = vec![value; b * c * (t + left + right)];
    let new_t = t + left + right;
    for bi in 0..b {
        for ci in 0..c {
            let src_offset = (bi * c + ci) * t;
            let dst_offset = (bi * c + ci) * new_t + left;
            out[dst_offset..dst_offset + t].copy_from_slice(&data[src_offset..src_offset + t]);
        }
    }
    Ok(Tensor::from_vec(out, (b, c, new_t), xs.device())?.to_dtype(dtype)?)
}

fn trim_tensor1d(xs: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let (_, _, t) = xs.dims3()?;
    if left + right >= t {
        return Err(VibeVoiceTokenizerError::InvalidShape(
            "invalid trim exceeding tensor length".to_string(),
        ));
    }
    Ok(xs.narrow(2, left, t - left - right)?)
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
        let (b, t, c) = output.mean.dims3().unwrap();
        assert_eq!((b, c), (1, 64));
        assert!(t > 0);
    }

    #[test]
    fn decoder_forward_has_expected_channels() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let decoder =
            TokenizerDecoder::load(&VibeVoiceAcousticTokenizerConfig::default(), vb).unwrap();
        let input = Tensor::zeros((1, 1, 64), DType::F32, &Device::Cpu).unwrap();
        let output = decoder.decode(&input).unwrap();
        assert_eq!(output.dims3().unwrap().1, 1);
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
        let output = norm
            .forward(&input)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
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

    #[test]
    fn ffn_preserves_last_dimension() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let ffn = Ffn::new(8, 16, true, vb).unwrap();
        let input = Tensor::zeros((1, 4, 8), DType::F32, &Device::Cpu).unwrap();
        let output = ffn.forward(&input).unwrap();
        assert_eq!(output.dims3().unwrap(), (1, 4, 8));
    }

    #[test]
    fn block1d_forward_preserves_shape() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let block = Block1D::new(8, 7, true, true, true, vb).unwrap();
        let input = Tensor::zeros((1, 8, 17), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims3().unwrap(), (1, 8, 17));
    }

    #[test]
    fn block1d_forward_handles_short_sequences() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let block = Block1D::new(8, 7, true, true, true, vb).unwrap();
        let input = Tensor::zeros((1, 8, 2), DType::F32, &Device::Cpu).unwrap();
        let output = block.forward(&input).unwrap();
        assert_eq!(output.dims3().unwrap(), (1, 8, 2));
    }

    #[test]
    fn pad_tensor1d_preserves_dtype_and_shape() {
        let input = Tensor::zeros((1, 2, 3), DType::F32, &Device::Cpu).unwrap();
        let output = pad_tensor1d(&input, 2, 1, 0.0).unwrap();
        assert_eq!(output.dims3().unwrap(), (1, 2, 6));
        assert_eq!(output.dtype(), DType::F32);
    }

    #[test]
    fn trim_tensor1d_errors_on_invalid_trim() {
        let input = Tensor::zeros((1, 2, 3), DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(trim_tensor1d(&input, 2, 2), Err(VibeVoiceTokenizerError::InvalidShape(_))));
    }

    #[test]
    fn trim_tensor1d_keeps_expected_window() {
        let input =
            Tensor::from_vec((0..10).map(|v| v as f32).collect::<Vec<_>>(), (1, 1, 10), &Device::Cpu)
                .unwrap();
        let output = trim_tensor1d(&input, 2, 3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(output, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn acoustic_model_sets_gaussian_fixed_std() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let model = VibeVoiceAcousticTokenizerModel::load(
            &VibeVoiceAcousticTokenizerConfig::default(),
            vb,
        )
        .unwrap();
        let input = Tensor::zeros((1, 1, 3200), DType::F32, &Device::Cpu).unwrap();
        let output = model.encode(&input).unwrap();
        assert_eq!(output.fixed_std, Some(0.5));
    }

    #[test]
    fn acoustic_model_decode_without_decoder_weights_errors() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let model = VibeVoiceAcousticTokenizerModel::load_hf_encoder(
            &VibeVoiceAcousticTokenizerConfig::default(),
            vb,
        )
        .unwrap();
        let input = Tensor::zeros((1, 1, 64), DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            model.decode(&input),
            Err(VibeVoiceTokenizerError::Unsupported(_))
        ));
    }

    #[test]
    fn semantic_model_hop_length_matches_config() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let model = VibeVoiceSemanticTokenizerModel::load(
            &VibeVoiceSemanticTokenizerConfig::default(),
            vb,
        )
        .unwrap();
        assert_eq!(model.hop_length(), 3200);
    }

    #[test]
    fn encoder_rejects_invalid_rank() {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &Device::Cpu);
        let encoder = TokenizerEncoder::load_acoustic(
            &VibeVoiceAcousticTokenizerConfig::default(),
            vb,
        )
        .unwrap();
        let bad = Tensor::zeros((1, 3200), DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            encoder.forward(&bad),
            Err(VibeVoiceTokenizerError::InvalidShape(_))
        ));
    }
}
