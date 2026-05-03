# Task 3: End-to-End Testing, Coverage, and TTS Decoder

## Status Update

Everything compiles and all 23 tests pass. Great work so far!

## Phase 1: End-to-End Test with Real Model Weights

The HuggingFace model ID is `microsoft/VibeVoice-ASR` (there's also `microsoft/VibeVoice-ASR-HF` for transformers integration).

There is a test audio file at `tmp/small-test.m4a` — the expected transcription is: "This is a small test to see how a recognition works"

Please:

1. Make sure the model loading code in `crates/vibevoice-asr/src/model.rs` uses `microsoft/VibeVoice-ASR` as the default HuggingFace model ID.

2. Verify the weight key mapping is correct. Check what keys are in the VibeVoice safetensors by looking at `tmp/VibeVoice/vibevoice/modular/modeling_vibevoice_asr.py` — specifically the `__init__` method to see attribute names like `self.acoustic_tokenizer`, `self.semantic_tokenizer`, `self.acoustic_connector`, `self.semantic_connector`, `self.language_model`, etc. These become the weight prefixes in the safetensors files.

3. Make the `aytranscribe` CLI actually work end-to-end:
   ```
   cargo run --release -- tmp/small-test.m4a output.txt
   ```
   It should download the model weights from HF on first run, load them, process the audio, run inference, and write the transcription to output.txt.

4. The tokenizer for text decoding should come from the HF repo as well (tokenizer.json).

## Phase 2: Test Coverage to ~85%

Run `cargo tarpaulin --workspace --out Html` (or just check what the current coverage is) and add tests to get close to 85%. Focus on:

- Audio loading edge cases (stereo files, different sample rates, short audio)
- Config serialization/deserialization round trips
- Processor prompt building with various audio lengths
- Error paths (missing files, invalid inputs, etc.)
- Model weight file discovery logic
- The custom Qwen2 forward pass (more shape tests, attention mask handling)
- CLI argument validation

Add an integration test in `crates/vibevoice-asr/tests/` that uses a tiny synthetic audio input to test the full pipeline (without real model weights — use random weights via VarMap).

## Phase 3: TTS Tokenizer Decoder (Stub → Real)

Port the `TokenizerDecoder` from `tmp/VibeVoice/vibevoice/modular/modular_vibevoice_tokenizer.py` into `crates/vibevoice-tokenizer/src/model.rs`. It's the inverse of the encoder — uses `NormConvTranspose1d` for upsampling with the reversed ratios [2,2,4,5,5,8]. This is needed for TTS but also validates the tokenizer architecture is complete.

## Notes

- You CAN run cargo commands now — there's a `.cargo/config.toml` with the aarch64 fp16 rustflags
- The test audio is at `tmp/small-test.m4a`, expected text: "This is a small test to see how a recognition works"
- Keep all tests passing as you make changes
