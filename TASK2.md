# Task 2: Complete VibeVoice ASR Implementation + Tests

## Status Update

The workspace scaffold from TASK.md is done and compiles cleanly. I made two small fixes after your pass:
1. Added `use symphonia::core::audio::Signal;` import in `crates/vibevoice-core/src/audio.rs` (needed for `.chan()`)
2. Added `[features] cuda = []` to `crates/vibevoice-core/Cargo.toml` to suppress cfg warnings
3. Added `Io(#[from] std::io::Error)` variant to `crates/vibevoice-asr/src/error.rs`
4. Fixed moved-value bug in `crates/vibevoice-asr/src/processor.rs` (captured `audio.samples.len()` before moving)

Everything compiles and is committed.

## What Needs to Be Done Now

### 1. Custom Qwen2 with `inputs_embeds` Support (Critical Path)

Candle's stock Qwen2 in `candle-transformers` only accepts token IDs. VibeVoice ASR needs to:
- Encode audio via the VAE tokenizer encoders to get speech embeddings
- Project them via SpeechConnector (Linear -> RMSNorm -> Linear) to Qwen2's hidden_size (3584)
- Concatenate prompt token embeddings + speech embeddings + continuation token embeddings
- Feed the combined `inputs_embeds` tensor into Qwen2's transformer layers (bypassing the embedding lookup)
- Run autoregressive generation from there

You need to either:
a) Copy and adapt the Qwen2 model from candle-transformers to accept `inputs_embeds` alongside or instead of `input_ids`, OR
b) Create a wrapper that calls into Qwen2's internal layers directly

Reference: `tmp/VibeVoice/vibevoice/modular/modeling_vibevoice_asr.py` — see how `VibeVoiceASRModel.forward()` constructs `inputs_embeds` and passes it to `self.language_model`.

### 2. Weight Loading from SafeTensors

Implement loading VibeVoice model weights from HuggingFace. The model ID is likely `fixie-ai/VibeVoice-ASR-7B` or similar. You'll need to:
- Map Python state_dict keys to our Rust struct field names
- Handle the split between acoustic_tokenizer, semantic_tokenizer, and language_model weights
- Reference `tmp/VibeVoice/vibevoice/configs/qwen2.5_7b_32k.json` for the full config

### 3. Test-Driven Development (IMPORTANT)

Please follow TDD methodology. We want **~85% code coverage** as measured by `cargo tarpaulin`. For each module:

- Write unit tests FIRST, then implement/fix the code to pass them
- Add `cargo-tarpaulin` as a dev dependency or document how to run it
- Each crate should have its own `#[cfg(test)] mod tests` sections

Specific tests needed:

**vibevoice-core:**
- Config parsing from JSON (use the real config from `tmp/VibeVoice/vibevoice/configs/qwen2.5_7b_32k.json`)
- Audio loading and resampling (create small synthetic WAV test fixtures)
- Device selection logic
- Error type conversions

**vibevoice-tokenizer:**
- Encoder construction from config
- Forward pass with synthetic tensor input (correct output shapes)
- Padding/unpadding utilities (pad1d, unpad1d, get_extra_padding_for_conv1d)
- RMSNorm and ConvRMSNorm correctness with known values

**vibevoice-asr:**
- SpeechConnector forward pass (input/output dimensions)
- Processor prompt construction with and without context
- Processor speech tensor shape validation
- Model config conversion from VibeVoice config to Qwen2 config
- End-to-end transcription with a tiny/mock model if feasible

**vibevoice-tts:**
- Basic stub tests (construction, error on unimplemented)

**aytranscribe:**
- CLI argument parsing
- Error handling for missing files

### 4. Generation Loop

Implement the autoregressive decoding loop:
- Greedy decoding (temperature=0)
- Stop on EOS token (`<|endoftext|>`)
- Respect max_new_tokens limit
- Return decoded text via the Qwen2 tokenizer

## Guidelines

- Run `cargo check` and `cargo test` after each significant change
- Keep the code compiling at all times
- Use `#[allow(unused)]` sparingly — prefer actually wiring things up
- The goal is a working ASR pipeline, even if slow — optimization comes later
