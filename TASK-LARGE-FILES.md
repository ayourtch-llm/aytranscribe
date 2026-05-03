# Task: Large File Support for ASR

## Context

VibeVoice ASR handles up to ~60 minutes in a single pass (24kHz audio → 7.5Hz token rate → ~27K tokens for 1 hour, fits in 64K context). For files longer than that, we need chunked processing.

The CLI currently works: `cargo run --release --features cuda -- tmp/small-test.m4a output.txt`

## What to Implement

### 1. Chunked Audio Processing

In `crates/vibevoice-asr/src/lib.rs` (or a new `chunked.rs`):

- Split audio longer than a configurable max duration (default: 55 minutes, leaving headroom for the 64K context)
- Use overlapping chunks (e.g., 5 seconds overlap) to avoid cutting words at boundaries
- Merge transcription results from overlapping chunks (deduplicate based on timestamps)
- Adjust timestamps in later chunks to be relative to the full file

### 2. Progress Callbacks (Library API)

Add a callback trait for library users:

```rust
pub trait TranscriptionProgress: Send + Sync {
    fn on_model_loaded(&self) {}
    fn on_chunk_start(&self, chunk_index: usize, total_chunks: usize, start_time_secs: f64) {}
    fn on_chunk_complete(&self, chunk_index: usize, partial_result: &str) {}
    fn on_complete(&self, full_result: &str) {}
}
```

- The `transcribe()` method should accept an optional `&dyn TranscriptionProgress`
- Default no-op implementation so it's not breaking

### 3. CLI Progress Display

In the CLI (`crates/aytranscribe/src/main.rs`):
- Show progress when processing multiple chunks: `[2/5] Processing 10:00-20:00...`
- Show total elapsed time at the end
- For single-chunk files (< 55 min), just show a simple "Processing..." message

### 4. Streaming JSON Output

For large files, write results incrementally:
- Each chunk's results are appended to the output file as they complete
- The final output is still valid JSON (array of segments)
- This way partial results are available even if the process is interrupted

### 5. Tests

- Test chunking logic with synthetic audio at various lengths
- Test timestamp adjustment across chunks
- Test overlap deduplication
- Test callback invocation order
- Test that single-chunk files work unchanged

## Important Notes

- Do NOT try to run cargo from the sandbox
- Keep the existing single-file transcribe path working (no regression)
- The chunking should be transparent to the user — same CLI, just handles large files automatically
- Use async-friendly design where possible (the callback could be used from async code later)
