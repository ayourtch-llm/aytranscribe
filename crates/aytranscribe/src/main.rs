use std::{
    fs::{self, File},
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    sync::Mutex,
    time::Instant,
};

use anyhow::{Result, anyhow};
use clap::Parser;
use vibevoice_asr::{
    ChunkedTranscriptionOptions, DEFAULT_CHUNK_OVERLAP_SECS, DEFAULT_MAX_CHUNK_DURATION_SECS,
    DEFAULT_MODEL_REPO, TranscriptionProgress, TranscriptionSegment, VibeVoiceAsrModel,
    VibeVoiceAsrProcessor, chunk_backup_path,
};
use vibevoice_core::{DeviceSpec, TARGET_SAMPLE_RATE, load_audio_file};

#[derive(Debug, Parser)]
struct Cli {
    audio: PathBuf,

    output: PathBuf,

    #[arg(long, default_value_t = 128)]
    max_new_tokens: usize,

    #[arg(long)]
    context: Option<String>,

    #[arg(long, default_value = "auto")]
    device: String,

    #[arg(long)]
    model_dir: Option<PathBuf>,

    #[arg(long)]
    tokenizer: Option<PathBuf>,

    #[arg(long, default_value = DEFAULT_MODEL_REPO)]
    model_repo: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    run(cli)
}

fn run(cli: Cli) -> Result<()> {
    let started_at = Instant::now();
    let device = parse_device(&cli.device)?.resolve()?;
    let audio = load_audio_file(&cli.audio, true)?;
    if audio.sample_rate != TARGET_SAMPLE_RATE {
        anyhow::bail!(
            "expected decoded audio at {TARGET_SAMPLE_RATE} Hz, got {} Hz",
            audio.sample_rate
        );
    }

    let (processor, mut model) = match (&cli.model_dir, &cli.tokenizer) {
        (Some(model_dir), Some(tokenizer)) => {
            let processor = VibeVoiceAsrProcessor::from_file(tokenizer)?;
            let model = VibeVoiceAsrModel::from_local_dir(model_dir, tokenizer, device.clone())?;
            (processor, model)
        }
        _ => {
            let model = VibeVoiceAsrModel::from_hf_hub(Some(&cli.model_repo), device.clone())?;
            let processor = model.processor_from_tokenizer()?;
            (processor, model)
        }
    };

    let total_duration_secs = audio.samples.len() as f64 / audio.sample_rate as f64;
    let progress = CliProgress::new(
        cli.output.clone(),
        total_duration_secs,
        DEFAULT_MAX_CHUNK_DURATION_SECS,
    );
    let options = ChunkedTranscriptionOptions {
        max_chunk_duration_secs: DEFAULT_MAX_CHUNK_DURATION_SECS,
        overlap_secs: DEFAULT_CHUNK_OVERLAP_SECS,
        max_new_tokens: cli.max_new_tokens,
        context_info: cli.context.as_deref(),
        output_path: Some(&cli.output),
    };
    let segments =
        model.transcribe_audio_buffer(&processor, &audio, &device, &options, Some(&progress))?;
    if let Some(err) = progress.take_error() {
        return Err(anyhow!(err));
    }

    let output = serde_json::to_string(&segments)?;
    fs::write(&cli.output, &output)?;
    println!("{output}");
    println!("Elapsed: {:.2}s", started_at.elapsed().as_secs_f64());
    Ok(())
}

fn parse_device(spec: &str) -> Result<DeviceSpec> {
    if spec.eq_ignore_ascii_case("auto") {
        return Ok(DeviceSpec::Auto);
    }
    if spec.eq_ignore_ascii_case("cpu") {
        return Ok(DeviceSpec::Cpu);
    }
    if let Some(rest) = spec.strip_prefix("cuda:") {
        return Ok(DeviceSpec::Cuda(rest.parse()?));
    }
    anyhow::bail!("unsupported device spec `{spec}`, use `auto`, `cpu` or `cuda:N`")
}

struct CliProgress {
    total_duration_secs: f64,
    chunk_duration_secs: f64,
    output_path: PathBuf,
    writer: Mutex<JsonStreamWriter>,
    error: Mutex<Option<String>>,
}

impl CliProgress {
    fn new(output_path: PathBuf, total_duration_secs: f64, chunk_duration_secs: f64) -> Self {
        Self {
            total_duration_secs,
            chunk_duration_secs,
            output_path,
            writer: Mutex::new(JsonStreamWriter::default()),
            error: Mutex::new(None),
        }
    }

    fn take_error(&self) -> Option<String> {
        self.error.lock().ok().and_then(|mut guard| guard.take())
    }

    fn record_error(&self, err: impl std::fmt::Display) {
        if let Ok(mut guard) = self.error.lock() {
            if guard.is_none() {
                *guard = Some(err.to_string());
            }
        }
    }
}

impl TranscriptionProgress for CliProgress {
    fn on_model_loaded(&self) {
        println!("Model loaded.");
    }

    fn on_chunk_start(&self, chunk_index: usize, total_chunks: usize, start_time_secs: f64) {
        let end_time_secs = (start_time_secs + self.chunk_duration_secs).min(self.total_duration_secs);
        if total_chunks <= 1 {
            println!("Processing...");
            return;
        }
        println!(
            "[{chunk_index}/{total_chunks}] Processing {}-{}...",
            format_clock(start_time_secs),
            format_clock(end_time_secs),
        );
    }

    fn on_token(&self, text: &str) {
        eprint!("{text}");
    }

    fn on_chunk_complete(&self, chunk_index: usize, partial_result: &str) {
        eprintln!();
        if let Err(err) = self
            .writer
            .lock()
            .map_err(|_| anyhow!("progress writer mutex poisoned"))
            .and_then(|mut writer| writer.append_segments(&self.output_path, partial_result))
        {
            self.record_error(err);
        }
        if !partial_result.trim().is_empty() && partial_result.trim() != "[]" {
            let chunk_path = chunk_backup_path(&self.output_path, chunk_index);
            if let Err(err) = fs::write(&chunk_path, partial_result) {
                self.record_error(err);
            } else {
                println!("Chunk {chunk_index} complete → {}", chunk_path.display());
            }
        }
    }

    fn on_complete(&self, full_result: &str) {
        eprintln!();
        if let Err(err) = self
            .writer
            .lock()
            .map_err(|_| anyhow!("progress writer mutex poisoned"))
            .and_then(|mut writer| writer.finish(&self.output_path, full_result))
        {
            self.record_error(err);
        }
    }
}

#[derive(Default)]
struct JsonStreamWriter {
    file: Option<File>,
    wrote_any: bool,
}

impl JsonStreamWriter {
    fn append_segments(&mut self, path: &PathBuf, partial_result: &str) -> Result<()> {
        let segments: Vec<TranscriptionSegment> = if partial_result.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(partial_result)?
        };
        if segments.is_empty() {
            return Ok(());
        }
        self.ensure_open(path)?;
        let wrote_any = &mut self.wrote_any;
        let file = self.file.as_mut().expect("file initialized");
        file.seek(SeekFrom::End(-1))?;
        for (idx, segment) in segments.into_iter().enumerate() {
            if *wrote_any || idx > 0 {
                write!(file, ",")?;
            }
            serde_json::to_writer(&mut *file, &segment)?;
            *wrote_any = true;
        }
        file.write_all(b"]")?;
        file.flush()?;
        Ok(())
    }

    fn finish(&mut self, path: &PathBuf, full_result: &str) -> Result<()> {
        self.file.take();
        fs::write(path, full_result)?;
        Ok(())
    }

    fn ensure_open(&mut self, path: &PathBuf) -> Result<()> {
        if self.file.is_none() {
            let mut file = File::create(path)?;
            file.write_all(b"[]")?;
            self.file = Some(file);
        }
        Ok(())
    }
}

fn format_clock(seconds: f64) -> String {
    let total = seconds.floor() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let secs = total % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes:02}:{secs:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tokenizers::{
        AddedToken, Tokenizer, models::wordlevel::WordLevel,
        pre_tokenizers::whitespace::Whitespace,
    };

    fn write_test_wav(path: &std::path::Path) {
        let samples = [0i16; 16];
        let sample_rate = 24_000u32;
        let byte_rate = sample_rate * 2;
        let block_align = 2u16;
        let data_len = (samples.len() * 2) as u32;
        let riff_len = 36 + data_len;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&riff_len.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        fs::write(path, bytes).unwrap();
    }

    fn write_test_tokenizer(path: &std::path::Path) {
        let vocab = [
            ("[UNK]", 0u32),
            ("<|speech_start|>", 1),
            ("<|speech_pad|>", 2),
            ("<|speech_end|>", 3),
            ("<|im_start|>", 4),
            ("<|im_end|>", 5),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|speech_start|>", true),
            AddedToken::from("<|speech_pad|>", true),
            AddedToken::from("<|speech_end|>", true),
            AddedToken::from("<|im_start|>", true),
            AddedToken::from("<|im_end|>", true),
        ]);
        tokenizer.save(path, false).unwrap();
    }

    #[test]
    fn cli_parses_required_args() {
        let cli = Cli::try_parse_from([
            "aytranscribe",
            "sample.wav",
            "out.txt",
        ])
        .unwrap();
        assert_eq!(cli.max_new_tokens, 128);
        assert_eq!(cli.device, "auto");
    }

    #[test]
    fn parse_device_rejects_invalid_value() {
        assert!(parse_device("gpu").is_err());
    }

    #[test]
    fn cli_rejects_missing_output_arg() {
        assert!(Cli::try_parse_from(["aytranscribe", "sample.wav"]).is_err());
    }

    #[test]
    fn parse_device_accepts_auto_and_cuda() {
        assert!(matches!(parse_device("auto").unwrap(), DeviceSpec::Auto));
        assert!(matches!(parse_device("cuda:1").unwrap(), DeviceSpec::Cuda(1)));
    }

    #[test]
    fn parse_device_accepts_cpu() {
        assert!(matches!(parse_device("cpu").unwrap(), DeviceSpec::Cpu));
    }

    #[test]
    fn parse_device_rejects_malformed_cuda_index() {
        assert!(parse_device("cuda:not-a-number").is_err());
    }

    #[test]
    fn format_clock_formats_hours_and_minutes() {
        assert_eq!(format_clock(65.0), "01:05");
        assert_eq!(format_clock(3665.0), "01:01:05");
    }

    #[test]
    fn json_stream_writer_produces_valid_array() {
        let temp_path = std::env::temp_dir()
            .join(format!("aytranscribe-stream-{}.json", std::process::id()));
        let mut writer = JsonStreamWriter::default();
        writer
            .append_segments(
                &temp_path,
                r#"[{"Start":0.0,"End":1.0,"Speaker":0,"Content":"a"}]"#,
            )
            .unwrap();
        writer
            .append_segments(
                &temp_path,
                r#"[{"Start":1.0,"End":2.0,"Speaker":0,"Content":"b"}]"#,
            )
            .unwrap();
        writer
            .finish(
                &temp_path,
                r#"[{"Start":0.0,"End":1.0,"Speaker":0,"Content":"a"},{"Start":1.0,"End":2.0,"Speaker":0,"Content":"b"}]"#,
            )
            .unwrap();
        let saved = fs::read_to_string(&temp_path).unwrap();
        let parsed: Vec<TranscriptionSegment> = serde_json::from_str(&saved).unwrap();
        assert_eq!(parsed.len(), 2);
        let _ = fs::remove_file(temp_path);
    }

    #[test]
    fn run_errors_on_invalid_device() {
        let cli = Cli {
            audio: PathBuf::from("missing.wav"),
            output: PathBuf::from("out.txt"),
            max_new_tokens: 8,
            context: None,
            device: "gpu".to_string(),
            model_dir: Some(PathBuf::from("unused-model")),
            tokenizer: Some(PathBuf::from("unused-tokenizer")),
            model_repo: DEFAULT_MODEL_REPO.to_string(),
        };
        assert!(run(cli).is_err());
    }

    #[test]
    fn run_errors_when_local_tokenizer_is_missing() {
        let temp_dir = std::env::temp_dir().join(format!("aytranscribe-run-missing-tok-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        let audio_path = temp_dir.join("sample.wav");
        write_test_wav(&audio_path);
        let cli = Cli {
            audio: audio_path,
            output: temp_dir.join("out.txt"),
            max_new_tokens: 8,
            context: None,
            device: "cpu".to_string(),
            model_dir: Some(temp_dir.clone()),
            tokenizer: Some(temp_dir.join("missing-tokenizer.json")),
            model_repo: DEFAULT_MODEL_REPO.to_string(),
        };
        assert!(run(cli).is_err());
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn run_errors_when_local_weights_are_missing() {
        let temp_dir = std::env::temp_dir().join(format!("aytranscribe-run-missing-weights-{}", std::process::id()));
        let _ = fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        fs::write(
            temp_dir.join("config.json"),
            serde_json::to_vec(&vibevoice_core::VibeVoiceASRConfig::default()).unwrap(),
        )
        .unwrap();
        let tokenizer_path = temp_dir.join("tokenizer.json");
        write_test_tokenizer(&tokenizer_path);
        let audio_path = temp_dir.join("sample.wav");
        write_test_wav(&audio_path);
        let cli = Cli {
            audio: audio_path,
            output: temp_dir.join("out.txt"),
            max_new_tokens: 8,
            context: None,
            device: "cpu".to_string(),
            model_dir: Some(temp_dir.clone()),
            tokenizer: Some(tokenizer_path),
            model_repo: DEFAULT_MODEL_REPO.to_string(),
        };
        assert!(run(cli).is_err());
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
