pub mod chunked;
pub mod error;
pub mod model;
pub mod processor;
mod qwen2;

pub use chunked::{
    AudioChunkRange, ChunkedTranscriptionOptions, DEFAULT_CHUNK_OVERLAP_SECS,
    DEFAULT_MAX_CHUNK_DURATION_SECS, TranscriptionProgress, TranscriptionSegment,
    chunk_backup_path, merge_transcription_segments, parse_transcription_segments,
    split_audio_into_chunks,
};
pub use error::{Result, VibeVoiceAsrError};
pub use model::{DEFAULT_MODEL_REPO, SpeechConnector, VibeVoiceAsrModel, VibeVoiceAsrSession};
pub use processor::{SYSTEM_PROMPT, VibeVoiceAsrInputs, VibeVoiceAsrProcessor};
