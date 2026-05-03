use std::{fs, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use vibevoice_asr::{DEFAULT_MODEL_REPO, VibeVoiceAsrModel, VibeVoiceAsrProcessor};
use vibevoice_core::DeviceSpec;

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
    let device = parse_device(&cli.device)?.resolve()?;
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
    let inputs = processor.prepare_audio_file(&cli.audio, &device, cli.context.as_deref())?;

    let output = model.transcribe_inputs(&inputs, cli.max_new_tokens)?;
    fs::write(&cli.output, &output)?;
    println!("{output}");
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tokenizers::{
        AddedToken, Tokenizer, models::wordlevel::WordLevel,
        pre_tokenizers::whitespace::Whitespace,
    };

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
        let cli = Cli {
            audio: temp_dir.join("missing.wav"),
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
        let cli = Cli {
            audio: temp_dir.join("missing.wav"),
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
