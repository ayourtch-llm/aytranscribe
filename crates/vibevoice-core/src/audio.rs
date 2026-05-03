use std::{fs::File, path::Path};

use rubato::{FftFixedInOut, Resampler};
use symphonia::core::{
    audio::{AudioBufferRef, SampleBuffer, Signal},
    codecs::DecoderOptions,
    errors::Error as SymphoniaError,
    formats::FormatOptions,
    io::MediaSourceStream,
    meta::MetadataOptions,
    probe::Hint,
};

use crate::error::{Result, VibeVoiceCoreError};

pub const TARGET_SAMPLE_RATE: u32 = 24_000;

#[derive(Clone, Debug)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Clone, Debug)]
pub struct AudioNormalizer {
    target_db_fs: f32,
    eps: f32,
}

impl Default for AudioNormalizer {
    fn default() -> Self {
        Self {
            target_db_fs: -25.0,
            eps: 1e-6,
        }
    }
}

impl AudioNormalizer {
    pub fn normalize(&self, input: &[f32]) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        let rms = (input.iter().map(|v| v * v).sum::<f32>() / input.len() as f32).sqrt();
        let scalar = 10f32.powf(self.target_db_fs / 20.0) / (rms + self.eps);
        let mut out: Vec<f32> = input.iter().map(|v| v * scalar).collect();
        let peak = out
            .iter()
            .fold(0.0f32, |acc, v| if v.abs() > acc { v.abs() } else { acc });
        if peak > 1.0 {
            out.iter_mut().for_each(|v| *v /= peak + self.eps);
        }
        out
    }
}

pub fn load_audio_file(path: impl AsRef<Path>, normalize: bool) -> Result<AudioBuffer> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let source = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|v| v.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| VibeVoiceCoreError::AudioDecode(e.to_string()))?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| VibeVoiceCoreError::AudioDecode("no default audio track".to_string()))?;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| VibeVoiceCoreError::AudioDecode(e.to_string()))?;

    let mut mono = Vec::<f32>::new();
    let sample_rate = track
        .codec_params
        .sample_rate
        .ok_or_else(|| VibeVoiceCoreError::AudioDecode("missing sample rate".to_string()))?;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(_)) => break,
            Err(err) => return Err(VibeVoiceCoreError::AudioDecode(err.to_string())),
        };

        let decoded = decoder
            .decode(&packet)
            .map_err(|e| VibeVoiceCoreError::AudioDecode(e.to_string()))?;

        match decoded {
            AudioBufferRef::F32(buf) => {
                push_interleaved_mono(buf.spec().channels.count(), buf.chan(0), &mut mono);
            }
            other => {
                let spec = *other.spec();
                let duration = other.capacity() as u64;
                let mut sample_buf = SampleBuffer::<f32>::new(duration, spec);
                sample_buf.copy_interleaved_ref(other);
                let channels = spec.channels.count();
                push_interleaved_mono_from_interleaved(channels, sample_buf.samples(), &mut mono);
            }
        }
    }

    let samples = if sample_rate == TARGET_SAMPLE_RATE {
        mono
    } else {
        resample_to_24khz(&mono, sample_rate)?
    };

    let samples = if normalize {
        AudioNormalizer::default().normalize(&samples)
    } else {
        samples
    };

    Ok(AudioBuffer {
        samples,
        sample_rate: TARGET_SAMPLE_RATE,
    })
}

fn push_interleaved_mono(channels: usize, first_channel: &[f32], mono: &mut Vec<f32>) {
    if channels <= 1 {
        mono.extend_from_slice(first_channel);
        return;
    }
    mono.extend_from_slice(first_channel);
}

fn push_interleaved_mono_from_interleaved(channels: usize, samples: &[f32], mono: &mut Vec<f32>) {
    if channels <= 1 {
        mono.extend_from_slice(samples);
        return;
    }
    for frame in samples.chunks(channels) {
        let sum = frame.iter().copied().sum::<f32>();
        mono.push(sum / channels as f32);
    }
}

fn resample_to_24khz(input: &[f32], input_sample_rate: u32) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let chunk = 1024usize;
    let mut resampler = FftFixedInOut::<f32>::new(
        input_sample_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        chunk,
        1,
    )
    .map_err(|e| VibeVoiceCoreError::AudioDecode(e.to_string()))?;

    let mut padded = input.to_vec();
    let rem = padded.len() % chunk;
    if rem != 0 {
        padded.resize(padded.len() + (chunk - rem), 0.0);
    }

    let mut output = Vec::new();
    for frame in padded.chunks(chunk) {
        let out = resampler
            .process(&[frame.to_vec()], None)
            .map_err(|e| VibeVoiceCoreError::AudioDecode(e.to_string()))?;
        output.extend_from_slice(&out[0]);
    }
    Ok(output)
}

