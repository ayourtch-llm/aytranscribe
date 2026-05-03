# aytranscribe

A Rust CLI for speech-to-text transcription using the
[VibeVoice ASR](https://github.com/microsoft/VibeVoice) model from Microsoft.

Built on [Candle](https://github.com/huggingface/candle) for native inference
-- no Python runtime required. Supports CUDA and Metal acceleration.

## Features

- Automatic model download from Hugging Face Hub
- Chunked processing for long audio files (default 15-minute chunks with 30s overlap)
- Per-chunk backup JSON files for resilience during long transcriptions
- Streaming token output during generation
- Speaker-diarized, timestamped JSON output
- Reads WAV, MP3, AAC/M4A, FLAC, OGG/Vorbis out of the box (via Symphonia)

## Quick start

```bash
# CPU-only
cargo install --path crates/aytranscribe

# With CUDA (NVIDIA GPU)
cargo install --path crates/aytranscribe --features cuda

# With Metal (Apple GPU)
cargo install --path crates/aytranscribe --features metal
```

The model weights (~6 GB) are downloaded automatically from
[microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF)
on first run.

## Usage

```bash
aytranscribe input.m4a output.json
```

For long files, chunk backup files are written alongside the output
(e.g. `output.chunk_001.json`, `output.chunk_002.json`, ...) so partial
results are preserved even if the process is interrupted.

### Options

```
aytranscribe [OPTIONS] <AUDIO> <OUTPUT>

Arguments:
  <AUDIO>    Path to the input audio file
  <OUTPUT>   Path for the JSON output

Options:
      --max-new-tokens <N>       Max tokens to generate per chunk [default: 128]
      --context <TEXT>            Optional context hint for the model
      --device <DEVICE>          auto | cpu | cuda:N [default: auto]
      --model-dir <DIR>          Use a local model directory instead of HF Hub
      --tokenizer <FILE>         Path to tokenizer.json (required with --model-dir)
      --model-repo <REPO>        HF repo ID [default: microsoft/VibeVoice-ASR-HF]
```

### Example

```bash
# Transcribe a 2-hour meeting recording on GPU
aytranscribe meeting.m4a meeting.json --max-new-tokens 32768 --device cuda:0
```

### Output format

```json
[
  {
    "Start": 0.0,
    "End": 34.7,
    "Speaker": 0,
    "Content": "Hello, today we will be talking about..."
  },
  {
    "Start": 34.7,
    "End": 72.3,
    "Speaker": 0,
    "Content": "Another item is, how well it works..."
  }
]
```

## Project structure

```
crates/
  aytranscribe/          CLI binary
  vibevoice-asr/         ASR model, chunked processing, processor
  vibevoice-core/        Shared types, audio loading, device abstraction
  vibevoice-tokenizer/   Acoustic & semantic tokenizer (VAE encoder)
  vibevoice-tts/         TTS model (work in progress)
```

## Requirements

- Rust 2024 edition (1.85+)
- For CUDA: CUDA toolkit and cuDNN
- For Metal: macOS with Metal-capable GPU

## Credits

This project is a Rust port of [VibeVoice](https://github.com/microsoft/VibeVoice)
by Microsoft, licensed under the MIT License. The original Python implementation
provides both ASR and TTS capabilities. The model weights are published at
[microsoft/VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF).

## License

MIT
