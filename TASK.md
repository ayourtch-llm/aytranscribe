# Task: Port VibeVoice ASR and TTS Models to Rust

Port the VibeVoice ASR and TTS models from Python to Rust using the Candle ML framework. The reference Python code is in ./tmp/VibeVoice/.

## Project Structure

Convert this project into a Cargo workspace with these crates:

1. `vibevoice-core` — shared types, configs, and utilities (audio loading, resampling)
2. `vibevoice-tokenizer` — the acoustic and semantic VAE tokenizer (encoder+decoder)
3. `vibevoice-asr` — the ASR model (tokenizer encoders + SpeechConnector + Qwen2 decoder)
4. `vibevoice-tts` — the TTS model (Qwen2 + DiffusionHead + tokenizer decoders)
5. `aytranscribe` — the existing binary crate, which will use vibevoice-asr as a library

## Key Reference Files to Port

Read these carefully before coding:

- `tmp/VibeVoice/vibevoice/modular/configuration_vibevoice.py` — Config classes (VibeVoiceAcousticTokenizerConfig, VibeVoiceSemanticTokenizerConfig, VibeVoiceASRConfig, VibeVoiceDiffusionHeadConfig)
- `tmp/VibeVoice/vibevoice/modular/modular_vibevoice_tokenizer.py` — VAE tokenizer implementation (~1200 lines). Key classes: TokenizerEncoder, TokenizerDecoder, SConv1d, NormConv1d, NormConvTranspose1d, RMSNorm, ConvRMSNorm, pad1d/unpad1d. Encoder ratios: [8,5,5,4,2,2], acoustic vae_dim=64, semantic vae_dim=128.
- `tmp/VibeVoice/vibevoice/modular/modeling_vibevoice_asr.py` — ASR model: VibeVoiceASRModel with acoustic_connector + semantic_connector (SpeechConnector: Linear->RMSNorm->Linear) feeding into Qwen2 decoder
- `tmp/VibeVoice/vibevoice/modular/modeling_vibevoice.py` — Full model with SpeechConnector definition
- `tmp/VibeVoice/vibevoice/modular/modular_vibevoice_diffusion_head.py` — DiffusionHead with TimestepEmbedder, HeadLayer (adaLN modulation), FeedForwardNetwork (SwiGLU), FinalLayer
- `tmp/VibeVoice/vibevoice/configs/qwen2.5_7b_32k.json` — Full model config JSON
- `tmp/VibeVoice/vibevoice/processor/audio_utils.py` — FFmpeg audio loading
- `tmp/VibeVoice/vibevoice/processor/vibevoice_asr_processor.py` — ASR processor with special tokens

## Architecture Summary

Audio pipeline: Raw 24kHz waveform -> AcousticTokenizerEncoder (depthwise conv downsampling 3200x, dim=64) -> SemanticTokenizerEncoder (dim=128) -> SpeechConnector (project to Qwen2 hidden_size=3584) -> Qwen2-7B decoder -> text tokens (ASR) or -> DiffusionHead -> AcousticTokenizerDecoder -> waveform (TTS).

## Technical Requirements

- Use `candle-core`, `candle-nn`, `candle-transformers` (Qwen2 is already in candle-transformers)
- Use `hf-hub` for model weight downloads from HuggingFace
- Use `tokenizers` crate for the Qwen2 text tokenizer
- Use `symphonia` for audio decoding (mp3/flac/wav/ogg) and implement resampling to 24kHz
- Load weights from safetensors format
- Support both CPU and CUDA devices
- Make the workspace Cargo.toml reference all member crates

## Important Notes

- The existing Cargo.toml is at the root — convert it to a workspace Cargo.toml
- The existing src/main.rs becomes the binary in the aytranscribe crate
- For Qwen2, reuse candle-transformers' existing Qwen2 implementation where possible
- Start with ASR as the priority, TTS can have stub implementations
- Include proper error types using thiserror in each crate
