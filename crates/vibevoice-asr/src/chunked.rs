use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use vibevoice_core::{AudioBuffer, TARGET_SAMPLE_RATE, load_audio_file};

use crate::{
    Result, VibeVoiceAsrError, VibeVoiceAsrModel, VibeVoiceAsrProcessor,
};

pub const DEFAULT_MAX_CHUNK_DURATION_SECS: f64 = 15.0 * 60.0;
pub const DEFAULT_CHUNK_OVERLAP_SECS: f64 = 30.0;

pub trait TranscriptionProgress: Send + Sync {
    fn on_model_loaded(&self) {}
    fn on_chunk_start(&self, _chunk_index: usize, _total_chunks: usize, _start_time_secs: f64) {}
    fn on_generation_progress(&self, _tokens_generated: usize, _estimated_total: usize) {}
    fn on_token(&self, _text: &str) {}
    fn on_chunk_complete(&self, _chunk_index: usize, _partial_result: &str) {}
    fn on_complete(&self, _full_result: &str) {}
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkedTranscriptionOptions<'a> {
    pub max_chunk_duration_secs: f64,
    pub overlap_secs: f64,
    pub max_new_tokens: usize,
    pub context_info: Option<&'a str>,
}

impl<'a> Default for ChunkedTranscriptionOptions<'a> {
    fn default() -> Self {
        Self {
            max_chunk_duration_secs: DEFAULT_MAX_CHUNK_DURATION_SECS,
            overlap_secs: DEFAULT_CHUNK_OVERLAP_SECS,
            max_new_tokens: 128,
            context_info: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioChunkRange {
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub start_sample: usize,
    pub end_sample: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionSegment {
    #[serde(rename = "Start")]
    pub start: f64,
    #[serde(rename = "End")]
    pub end: f64,
    #[serde(rename = "Speaker")]
    pub speaker: i64,
    #[serde(rename = "Content")]
    pub content: String,
}

impl TranscriptionSegment {
    fn with_offset(mut self, start_time_secs: f64) -> Self {
        self.start += start_time_secs;
        self.end += start_time_secs;
        self
    }
}

impl VibeVoiceAsrModel {
    pub fn transcribe_audio_file(
        &mut self,
        processor: &VibeVoiceAsrProcessor,
        path: impl AsRef<std::path::Path>,
        device: &Device,
        options: &ChunkedTranscriptionOptions<'_>,
        progress: Option<&dyn TranscriptionProgress>,
    ) -> Result<Vec<TranscriptionSegment>> {
        let audio = load_audio_file(path, true)?;
        self.transcribe_audio_buffer(processor, &audio, device, options, progress)
    }

    pub fn transcribe_audio_buffer(
        &mut self,
        processor: &VibeVoiceAsrProcessor,
        audio: &AudioBuffer,
        device: &Device,
        options: &ChunkedTranscriptionOptions<'_>,
        progress: Option<&dyn TranscriptionProgress>,
    ) -> Result<Vec<TranscriptionSegment>> {
        if audio.sample_rate != TARGET_SAMPLE_RATE {
            return Err(VibeVoiceAsrError::InvalidInput(format!(
                "expected {TARGET_SAMPLE_RATE} Hz audio, got {} Hz",
                audio.sample_rate
            )));
        }
        if options.max_new_tokens == 0 {
            if let Some(progress) = progress {
                progress.on_model_loaded();
                progress.on_complete("[]");
            }
            return Ok(Vec::new());
        }
        let chunk_ranges = split_audio_into_chunks(
            audio.samples.len(),
            audio.sample_rate,
            options.max_chunk_duration_secs,
            options.overlap_secs,
        );
        if let Some(progress) = progress {
            progress.on_model_loaded();
        }
        let mut merged = Vec::new();
        for chunk in &chunk_ranges {
            let start_time_secs = chunk.start_sample as f64 / audio.sample_rate as f64;
            if let Some(progress) = progress {
                progress.on_chunk_start(chunk.chunk_index, chunk.total_chunks, start_time_secs);
            }
            let chunk_samples = &audio.samples[chunk.start_sample..chunk.end_sample];
            let speech_tensor =
                Tensor::from_vec(chunk_samples.to_vec(), (1, 1, chunk_samples.len()), device)?;
            let inputs =
                processor.prepare_audio_tensor(speech_tensor, chunk_samples.len(), options.context_info)?;
            let chunk_json =
                self.transcribe_inputs_with_progress(&inputs, options.max_new_tokens, progress)?;
            let chunk_segments =
                parse_transcription_segments(&chunk_json, start_time_secs)?;
            let appended = merge_transcription_segments(&mut merged, chunk_segments);
            if let Some(progress) = progress {
                let partial_json = serde_json::to_string(&appended)?;
                progress.on_chunk_complete(chunk.chunk_index, &partial_json);
            }
        }
        if let Some(progress) = progress {
            let final_json = serde_json::to_string(&merged)?;
            progress.on_complete(&final_json);
        }
        Ok(merged)
    }
}

pub fn split_audio_into_chunks(
    total_samples: usize,
    sample_rate: u32,
    max_chunk_duration_secs: f64,
    overlap_secs: f64,
) -> Vec<AudioChunkRange> {
    if total_samples == 0 {
        return vec![AudioChunkRange {
            chunk_index: 1,
            total_chunks: 1,
            start_sample: 0,
            end_sample: 0,
        }];
    }
    let max_chunk_samples = ((max_chunk_duration_secs.max(1.0)) * sample_rate as f64).round() as usize;
    let overlap_samples = (overlap_secs.max(0.0) * sample_rate as f64).round() as usize;
    if total_samples <= max_chunk_samples || max_chunk_samples <= overlap_samples {
        return vec![AudioChunkRange {
            chunk_index: 1,
            total_chunks: 1,
            start_sample: 0,
            end_sample: total_samples,
        }];
    }
    let step = max_chunk_samples - overlap_samples;
    let mut raw = Vec::new();
    let mut start_sample = 0usize;
    while start_sample < total_samples {
        let end_sample = (start_sample + max_chunk_samples).min(total_samples);
        raw.push((start_sample, end_sample));
        if end_sample == total_samples {
            break;
        }
        start_sample += step;
    }
    let total_chunks = raw.len();
    raw.into_iter()
        .enumerate()
        .map(|(idx, (start_sample, end_sample))| AudioChunkRange {
            chunk_index: idx + 1,
            total_chunks,
            start_sample,
            end_sample,
        })
        .collect()
}

pub fn parse_transcription_segments(
    json: &str,
    start_time_offset_secs: f64,
) -> Result<Vec<TranscriptionSegment>> {
    let trimmed = json.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let segments: Vec<TranscriptionSegment> = serde_json::from_str(trimmed).map_err(|err| {
        VibeVoiceAsrError::InvalidInput(format!(
            "transcription output was not valid JSON segment array: {err}"
        ))
    })?;
    Ok(segments
        .into_iter()
        .map(|segment| segment.with_offset(start_time_offset_secs))
        .collect())
}

pub fn merge_transcription_segments(
    merged: &mut Vec<TranscriptionSegment>,
    incoming: Vec<TranscriptionSegment>,
) -> Vec<TranscriptionSegment> {
    let mut appended = Vec::new();
    for mut segment in incoming {
        if segment.end <= segment.start {
            continue;
        }
        if let Some(last) = merged.last() {
            if is_duplicate_or_overlapping(last, &segment) {
                continue;
            }
            if segment.start < last.end {
                segment.start = last.end;
                if segment.end <= segment.start {
                    continue;
                }
            }
        }
        merged.push(segment.clone());
        appended.push(segment);
    }
    appended
}

fn is_duplicate_or_overlapping(
    previous: &TranscriptionSegment,
    candidate: &TranscriptionSegment,
) -> bool {
    let content_matches = previous.content.trim() == candidate.content.trim();
    let speaker_matches = previous.speaker == candidate.speaker;
    let near_same_start = (previous.start - candidate.start).abs() < 1.0;
    let fully_covered = candidate.end <= previous.end + 0.5;
    (speaker_matches && content_matches && near_same_start) || (candidate.start < previous.end && fully_covered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_chunk_for_short_audio() {
        let chunks = split_audio_into_chunks(24_000 * 60, 24_000, 45.0 * 60.0, 30.0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start_sample, 0);
        assert_eq!(chunks[0].end_sample, 24_000 * 60);
    }

    #[test]
    fn long_audio_uses_overlap() {
        let chunks = split_audio_into_chunks(
            24_000 * 100 * 60,
            24_000,
            45.0 * 60.0,
            30.0,
        );
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[1].start_sample, 24_000 * (45 * 60 - 30));
    }

    #[test]
    fn timestamp_offset_is_applied() {
        let parsed = parse_transcription_segments(
            r#"[{"Start":1.0,"End":2.5,"Speaker":0,"Content":"hello"}]"#,
            120.0,
        )
        .unwrap();
        assert_eq!(parsed[0].start, 121.0);
        assert_eq!(parsed[0].end, 122.5);
    }

    #[test]
    fn overlap_dedup_skips_duplicate_boundary_segments() {
        let mut merged = vec![TranscriptionSegment {
            start: 10.0,
            end: 12.0,
            speaker: 0,
            content: "hello".to_string(),
        }];
        let appended = merge_transcription_segments(
            &mut merged,
            vec![
                TranscriptionSegment {
                    start: 10.2,
                    end: 11.8,
                    speaker: 0,
                    content: "hello".to_string(),
                },
                TranscriptionSegment {
                    start: 12.0,
                    end: 14.0,
                    speaker: 0,
                    content: "world".to_string(),
                },
            ],
        );
        assert_eq!(merged.len(), 2);
        assert_eq!(appended.len(), 1);
        assert_eq!(appended[0].content, "world");
    }

    #[test]
    fn overlapping_new_segment_is_trimmed_forward() {
        let mut merged = vec![TranscriptionSegment {
            start: 0.0,
            end: 5.0,
            speaker: 0,
            content: "first".to_string(),
        }];
        let appended = merge_transcription_segments(
            &mut merged,
            vec![TranscriptionSegment {
                start: 4.5,
                end: 7.0,
                speaker: 1,
                content: "second".to_string(),
            }],
        );
        assert_eq!(appended[0].start, 5.0);
    }

    #[test]
    fn empty_audio_still_returns_single_empty_chunk() {
        let chunks = split_audio_into_chunks(0, 24_000, 45.0 * 60.0, 30.0);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].end_sample, 0);
    }
}
